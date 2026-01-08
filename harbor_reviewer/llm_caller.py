"""
LLM Caller Module
--------------------
Purpose:
- Sends prompts to LLM and retrieves raw responses
- Handles retries and respects rate limits from LLM provider
- Supports asynchronous and parallel execution
- Provides mechanisms to collect and return all raw responses for further processing

This module is responsible ONLY for LLM communication, not evaluation logic.
Evaluation logic is handled by evaluation_engine.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import litellm

# Add project root to path for shared module imports
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_cwd = Path.cwd().resolve()
if str(_cwd) not in sys.path and _cwd != _project_root:
    sys.path.insert(0, str(_cwd))

from shared.utils.error_handler import get_error_handler
from shared.utils.log import setup_logger

from .input_loader import InputLoader
from .models import LLMBatch, LLMRequest, LLMResponse

logger = setup_logger(__name__)


class LLMCaller:
    """LLM communication layer - handles only LLM calls, retries, and rate limiting"""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the LLM caller

        Args:
            config_dir: Path to the configuration directory
        """
        self.input_loader = InputLoader(config_dir)
        self.config = self.input_loader.load_evaluator_config()

        # Get max parallel workers for concurrency control
        # Todo: Add a module to determine this dynamically based on the system's resources and API rate limits.
        self.max_parallel_workers = self.config.processing.max_parallel_workers

        # Initialize error handler
        self.error_handler = get_error_handler(config_dir)

        logger.info("LLM Caller initialized successfully")

    async def call_llm_single(
        self, prompt: Dict[str, str], model_name: str, request_id: Optional[str] = None
    ) -> LLMResponse:
        """
        Make a single LLM call with retry logic and rate limiting (In case a single call is needed)

        Args:
            prompt: Formatted prompt with system and user messages
            model_name: Name of the model to use
            request_id: Optional identifier for tracking

        Returns:
            LLMResponse with the result
        """
        request = LLMRequest(
            prompt=prompt, model_name=model_name, request_id=request_id
        )

        return await self._process_llm_request(request)

    async def call_llm_batch(self, batch: LLMBatch) -> LLMBatch:
        """
        Make multiple LLM calls with parallel processing (Main method for batch processing)

        Args:
            batch: LLMBatch containing requests to process

        Returns:
            LLMBatch with responses added
        """
        if not batch.requests:
            # Return batch with empty responses list for consistency
            return LLMBatch(
                requests=batch.requests,
                responses=[],
                batch_id=batch.batch_id,
                created_at=batch.created_at,
            )

        # Create tasks for parallel execution
        tasks = []
        for request in batch.requests:
            task = asyncio.create_task(self._process_llm_request(request))
            tasks.append(task)

        # Execute all tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.max_parallel_workers)

        async def limited_task(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_task(task) for task in tasks]

        # Wait for all tasks to complete
        responses = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Process results
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Handle task exceptions with structured error handling
                error_info = self.error_handler.get_error_info(
                    "processing.batch_processing", details=str(response)
                )
                error_response = LLMResponse(
                    raw_response="",
                    model_used=batch.requests[i].model_name,
                    latency_ms=0.0,
                    error=error_info.message,
                    error_code=error_info.code,
                    error_details=error_info.details,
                    request_id=batch.requests[i].request_id,
                )
                results.append(error_response)
            else:
                results.append(response)

        # Return updated batch with responses
        return LLMBatch(
            requests=batch.requests,
            responses=results,
            batch_id=batch.batch_id,
            created_at=batch.created_at,
        )

    async def _process_llm_request(self, request: LLMRequest) -> LLMResponse:
        """
        Call LLM with retry logic and rate limiting (Implementation Details).

        Args:
            request: LLM request

        Returns:
            LLMResponse with the result
        """
        model_config = self.input_loader.get_model_config(request.model_name)
        if not model_config:
            error_info = self.error_handler.get_error_info(
                "model.not_found", model_name=request.model_name
            )
            return LLMResponse(
                raw_response="",
                model_used=request.model_name,
                latency_ms=0.0,
                error=error_info.message,
                error_code=error_info.code,
                error_details=error_info.details,
                request_id=request.request_id,
            )

        start_time = time.time()
        retry_count = 0

        while retry_count <= model_config.max_retries:
            try:
                # Prepare LiteLLM call
                litellm_model = self.input_loader.get_litellm_model_name(
                    request.model_name
                )
                litellm_config = self.input_loader.get_litellm_config(
                    request.model_name
                )

                # Set API key
                api_key = self.input_loader.get_api_key(request.model_name)
                if api_key:
                    litellm.set_verbose = False
                    litellm.api_key = api_key

                # Make the call
                messages = request.prompt

                response = await litellm.acompletion(
                    model=litellm_model, messages=messages, **litellm_config
                )

                # Extract response content
                raw_response = response.choices[0].message.content

                if not raw_response or not raw_response.strip():
                    logger.warning("No response from LLM, proceeding to retry")
                    retry_count += 1
                    await asyncio.sleep(model_config.retry_delay)
                    continue

                # Extract token usage metrics
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                finish_reason = None

                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                    completion_tokens = getattr(
                        response.usage, "completion_tokens", None
                    )
                    total_tokens = getattr(response.usage, "total_tokens", None)

                # Extract finish reason from first choice
                if hasattr(response.choices[0], "finish_reason"):
                    finish_reason = response.choices[0].finish_reason

                # Calculate response size
                response_size_chars = len(raw_response) if raw_response else 0

                # Parse JSON response if possible
                parsed_response = None
                try:
                    parsed_response = json.loads(raw_response)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse JSON response for request {request.request_id}"
                    )

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                return LLMResponse(
                    raw_response=raw_response,
                    parsed_response=parsed_response,
                    model_used=request.model_name,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    request_id=request.request_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    response_size_chars=response_size_chars,
                    finish_reason=finish_reason,
                )

            except Exception as e:
                retry_count += 1
                error_msg = str(e)

                # Check if it's a rate limit error
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    logger.warning(
                        f"Rate limit hit for {request.model_name}, waiting before retry"
                    )
                    await asyncio.sleep(
                        model_config.retry_delay * 2
                    )  # Double delay for rate limits
                else:
                    logger.warning(
                        f"Error calling {request.model_name} (attempt {retry_count}): {error_msg}"
                    )
                    await asyncio.sleep(model_config.retry_delay)

                # If this was the last retry, return error response
                if retry_count > model_config.max_retries:
                    logger.error(f"Max retries exceeded for {request.model_name}")
                    # Use generic API error - let downstream components handle specifics
                    error_info = self.error_handler.get_error_info(
                        "api.service_unavailable",
                        model_name=request.model_name,
                        details=error_msg,
                    )

                    return LLMResponse(
                        raw_response="",
                        model_used=request.model_name,
                        latency_ms=(time.time() - start_time) * 1000,
                        error=error_info.message,
                        error_code=error_info.code,
                        error_details=error_info.details,
                        retry_count=retry_count,
                        request_id=request.request_id,
                    )

        # Should not reach here - fallback error handling
        error_info = self.error_handler.get_error_info(
            "processing.unexpected_error", model_name=request.model_name
        )
        return LLMResponse(
            raw_response="",
            model_used=request.model_name,
            latency_ms=(time.time() - start_time) * 1000,
            error=error_info.message,
            error_code=error_info.code,
            error_details=error_info.details,
            request_id=request.request_id,
        )

    async def close(self):
        """Clean up resources"""
        try:
            # Note: LiteLLM handles its own connection cleanup
            # Note: InputLoader doesn't manage resources that need cleanup
            # Note: Semaphores are automatically cleaned up by asyncio
            # Todo: Add cleanup for any other resources that might be used as a result of future updates.

            logger.info("LLM Caller closed successfully")

        except Exception as e:
            logger.error(f"Error during LLM Caller cleanup: {e}")
            raise
