"""
Response Parser Module
---------------------
Purpose:
- Parses raw LLM responses into structured ratings + justifications per criterion
- Handles various response formats and edge cases
- Provides validation and error handling for malformed responses
- Extracts evaluation scores and reasoning from LLM outputs
- Implements retry logic before using fallback parsing methods
"""

import asyncio
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from shared.utils.log import setup_logger

from .models import CriterionResult, EvaluationResult

logger = setup_logger(__name__)


class EvaluationScore(Enum):
    """Evaluation score levels"""

    VERY_POOR = 1
    POOR = 2
    FAIR = 3
    GOOD = 4
    EXCELLENT = 5


class ResponseParser:
    def __init__(self, config_dir: str = "configs"):
        # Initialize regex patterns for fallback parsing
        self.score_patterns = [
            r'score["\s]*:["\s]*(\d+(?:\.\d+)?)',
            r'rating["\s]*:["\s]*(\d+(?:\.\d+)?)',
            r'grade["\s]*:["\s]*(\d+(?:\.\d+)?)',
            r'"s"["\s]*:["\s]*(\d+(?:\.\d+)?)',  # Support legacy short keys
            r'"score"["\s]*:["\s]*(\d+(?:\.\d+)?)',  # Quoted keys
        ]
        self.justification_patterns = [
            r'justification["\s]*:["\s]*["\']([^"\']*)["\']',
            r'reasoning["\s]*:["\s]*["\']([^"\']*)["\']',
            r'explanation["\s]*:["\s]*["\']([^"\']*)["\']',
            r'comment["\s]*:["\s]*["\']([^"\']*)["\']',
            r'"j"["\s]*:["\s]*["\']([^"\']*)["\']',  # Support legacy short keys
            r'"justification"["\s]*:["\s]*["\']([^"\']*)["\']',  # Quoted keys
            # More flexible pattern for multiline text
            r'justification["\s]*:["\s]*["\']([^"\']*)',
            r'"justification"["\s]*:["\s]*["\']([^"\']*)',
        ]
        self.confidence_patterns = [
            r'confidence["\s]*:["\s]*(\d+(?:\.\d+)?)',
            r'"c"["\s]*:["\s]*(\d+(?:\.\d+)?)',  # Support legacy short keys
            r'"confidence"["\s]*:["\s]*(\d+(?:\.\d+)?)',  # Quoted keys
        ]

        # Load configuration for retry settings
        try:
            from .input_loader import InputLoader

            self.input_loader = InputLoader(config_dir)
            self.config = self.input_loader.load_evaluator_config()
        except Exception as e:
            logger.warning(f"Failed to load configuration for retry settings: {e}")
            self.config = None

    def parse_response(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> CriterionResult:
        """
        Parse the response from the LLM with graceful degradation.

        This is the main method that tries multiple parsing approaches:
        1. Try parsing JSON (structured data)
        2. Fallback to structured text parsing
        3. Final fallback: regex extraction of known keys

        Args:
            raw_response: String response from the LLM to parse
            criterion_name: Name of the criterion being evaluated
            request_id: Optional request identifier for logging

        Returns:
            CriterionResult with parsed score, justification, and confidence
        """
        if not raw_response or not raw_response.strip():
            return CriterionResult(
                criterion_name=criterion_name,
                score=None,
                justification="No response received",
                raw_response=raw_response,
                parsing_errors=["Empty or missing response"],
            )

        # Step 1: Try parsing JSON
        json_result = self._try_json_parsing(raw_response, criterion_name, request_id)
        if json_result and json_result.is_valid():
            return json_result

        # Step 2: Try structured text parsing
        text_result = self._try_structured_text_parsing(
            raw_response, criterion_name, request_id
        )
        if text_result and text_result.is_valid():
            return text_result

        # Step 3: Final fallback - regex extraction
        regex_result = self._try_regex_parsing(raw_response, criterion_name, request_id)
        if regex_result and regex_result.is_valid():
            return regex_result

        # If all parsing methods fail, return error result
        request_info = f" (request_id: {request_id})" if request_id else ""
        logger.error(
            f"Failed to parse response for {criterion_name}{request_info} with all available methods"
        )
        return CriterionResult(
            criterion_name=criterion_name,
            score=None,
            justification="Parsing failed - response format unclear",
            raw_response=raw_response,
            parsing_errors=["All parsing methods failed"],
        )

    async def parse_response_with_retries(
        self,
        llm_caller,
        prompt: Dict[str, str],
        model_name: str,
        criterion_name: str,
        request_id: Optional[str] = None,
    ) -> CriterionResult:
        """
        Parse response with optimized retry logic.

        This method will:
        1. Make the initial LLM request
        2. Try ALL fallback parsing methods first
        3. Only retry the LLM request if ALL parsing methods fail
        4. Repeat this cycle until retries are exhausted

        Args:
            llm_caller: LLMCaller instance for making requests
            prompt: Formatted prompt with system and user messages
            model_name: Name of the model to use
            criterion_name: Name of the criterion being evaluated
            request_id: Optional request identifier for logging

        Returns:
            CriterionResult with parsed score, justification, and confidence
        """
        if not self.config:
            # Fallback to original parsing if config is not available
            logger.warning("Configuration not available, using original parsing logic")
            return self.parse_response("", criterion_name, request_id)

        # Get model configuration for retry settings
        try:
            model_config = self.input_loader.get_model_config(model_name)
            max_retries = model_config.max_retries
            retry_delay = model_config.retry_delay
        except Exception as e:
            logger.warning(f"Failed to get model config for {model_name}: {e}")
            max_retries = 3  # Default fallback
            retry_delay = 1.0

        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Make LLM request
                llm_response = await llm_caller.call_llm_single(
                    prompt, model_name, request_id
                )

                # Check if request was successful
                if llm_response.error:
                    logger.warning(
                        f"LLM request failed for {criterion_name} (attempt {retry_count + 1}): {llm_response.error}"
                    )
                    retry_count += 1
                    if retry_count <= max_retries:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        # All retries exhausted, return error result
                        return CriterionResult(
                            criterion_name=criterion_name,
                            score=None,
                            justification=f"LLM request failed after {max_retries} retries: {llm_response.error}",
                            raw_response=llm_response.raw_response,
                            parsing_errors=[
                                f"LLM request failed: {llm_response.error}"
                            ],
                        )

                # Try ALL fallback parsing methods first
                parsed_result = self.parse_response(
                    llm_response.raw_response, criterion_name, request_id
                )

                # If ANY parsing method succeeded, return the result immediately
                if parsed_result.is_valid():
                    logger.debug(
                        f"Parsing succeeded for {criterion_name} on attempt {retry_count + 1}"
                    )
                    return parsed_result

                # If ALL parsing methods failed but we have retries left, retry the request
                if retry_count < max_retries:
                    logger.warning(
                        f"All parsing methods failed for {criterion_name} (attempt {retry_count + 1}), retrying request..."
                    )
                    retry_count += 1
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # All retries exhausted, return the failed parsing result
                    logger.error(
                        f"All retries exhausted for {criterion_name}, returning failed parsing result"
                    )
                    return parsed_result

            except Exception as e:
                logger.error(
                    f"Unexpected error during retry for {criterion_name} (attempt {retry_count + 1}): {e}"
                )
                retry_count += 1
                if retry_count <= max_retries:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # All retries exhausted, return error result
                    return CriterionResult(
                        criterion_name=criterion_name,
                        score=None,
                        justification=f"Unexpected error after {max_retries} retries: {str(e)}",
                        raw_response="",
                        parsing_errors=[f"Unexpected error: {str(e)}"],
                    )

        # Should not reach here, but just in case
        return CriterionResult(
            criterion_name=criterion_name,
            score=None,
            justification="Retry logic failed unexpectedly",
            raw_response="",
            parsing_errors=["Retry logic failed unexpectedly"],
        )

    def parse_response_with_fallback_only(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> CriterionResult:
        """
        Parse response using ALL fallback methods without making additional LLM calls.

        This method tries all available parsing methods in order:
        1. JSON parsing
        2. Structured text parsing
        3. Regex extraction

        Args:
            raw_response: Raw response from LLM
            criterion_name: Name of the criterion being evaluated
            request_id: Optional request identifier for logging

        Returns:
            CriterionResult with parsed score, justification, and confidence
        """
        # Try all parsing methods - this is what parse_response already does
        return self.parse_response(raw_response, criterion_name, request_id)

    def parse_batch_responses(
        self,
        responses: Dict[str, str],
        record_id: str,
        model_used: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Parse multiple criterion responses for a single record

        Args:
            responses: Dictionary of criterion_name -> raw_response
            record_id: Identifier for the record being evaluated
            model_used: Model used for evaluation

        Returns:
            EvaluationResult with all parsed criterion results
        """
        criterion_results = {}
        parsing_errors = []

        for criterion_name, raw_response in responses.items():
            try:
                result = self.parse_response(raw_response, criterion_name)
                criterion_results[criterion_name] = result

                if not result.is_valid():
                    parsing_errors.extend(result.parsing_errors)

            except Exception as e:
                logger.error(f"Error parsing response for {criterion_name}: {e}")
                parsing_errors.append(f"Parsing error for {criterion_name}: {str(e)}")

                criterion_results[criterion_name] = CriterionResult(
                    criterion_name=criterion_name,
                    score=None,
                    justification="Parsing error occurred",
                    raw_response=raw_response,
                    parsing_errors=[str(e)],
                )

        # Calculate overall score if all results are valid
        overall_score = None
        overall_justification = None

        valid_results = [r for r in criterion_results.values() if r.is_valid()]
        if valid_results:
            overall_score = sum(r.score for r in valid_results) / len(valid_results)
            overall_justification = (
                f"Average score across {len(valid_results)} criteria"
            )

        return EvaluationResult(
            record_id=record_id,
            criterion_results=criterion_results,
            overall_score=overall_score,
            overall_justification=overall_justification,
            model_used=model_used,
            parsing_errors=parsing_errors,
        )

    def _try_json_parsing(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> Optional[CriterionResult]:
        """
        Try to parse response as JSON with graceful error handling.
        """
        try:
            # Try existing JSON extraction method
            json_data = self.extract_json_string_from_response(raw_response)
            if json_data:
                structured_data = self._structure_response(json_data)
                if structured_data:
                    return CriterionResult(
                        criterion_name=criterion_name,
                        score=int(structured_data["score"]),
                        justification=structured_data["justification"],
                        confidence=structured_data.get("confidence"),
                        raw_response=raw_response,
                    )
        except Exception as e:
            request_info = f" (request_id: {request_id})" if request_id else ""
            logger.debug(
                f"JSON parsing failed for {criterion_name}{request_info}: {str(e)}"
            )
        return None

    def _try_structured_text_parsing(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> Optional[CriterionResult]:
        """
        Try to parse response as structured text (legacy compatibility).
        """
        try:
            # Extract score
            score = self._extract_with_patterns(
                raw_response, self.score_patterns, float
            )
            if not score:
                return None

            # Extract justification
            justification = self._extract_with_patterns(
                raw_response, self.justification_patterns, str
            )
            if not justification:
                return None

            # Extract confidence
            confidence = self._extract_with_patterns(
                raw_response, self.confidence_patterns, float
            )

            return CriterionResult(
                criterion_name=criterion_name,
                score=int(score),
                justification=justification,
                confidence=confidence,
                raw_response=raw_response,
            )

        except Exception as e:
            request_info = f" (request_id: {request_id})" if request_id else ""
            logger.debug(
                f"Structured text parsing failed for {criterion_name}{request_info}: {e}"
            )
            return None

    def _try_regex_parsing(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> Optional[CriterionResult]:
        """
        Final fallback: extract key information using regex patterns.
        """
        try:
            # Extract score
            score = self._extract_with_patterns(
                raw_response, self.score_patterns, float
            )
            if not score:
                return None

            # Extract justification
            justification = self._extract_with_patterns(
                raw_response, self.justification_patterns, str
            )
            if not justification:
                return None

            # Extract confidence
            confidence = self._extract_with_patterns(
                raw_response, self.confidence_patterns, float
            )

            return CriterionResult(
                criterion_name=criterion_name,
                score=int(score),
                justification=justification,
                confidence=confidence,
                raw_response=raw_response,
                parsing_errors=["Used regex fallback parsing"],
            )

        except Exception as e:
            request_info = f" (request_id: {request_id})" if request_id else ""
            logger.debug(
                f"Regex parsing failed for {criterion_name}{request_info}: {e}"
            )
            return None

    def _extract_with_patterns(
        self, text: str, patterns: List[str], value_type: type
    ) -> Optional[Any]:
        """
        Extract value using a list of regex patterns.
        """
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    value = match.group(1).strip()
                    return value_type(value)
                except (ValueError, TypeError) as e:
                    logger.debug(
                        f"Failed to convert '{value}' to {value_type.__name__}: {e}"
                    )
                    continue
        return None

    def _structure_response(self, response: dict) -> Optional[Dict[str, Any]]:
        """
        Structure and validate the response data.
        Supports both legacy short keys (s, j, c) and full keys.
        """
        if not isinstance(response, dict):
            logger.error(f"Response must be dictionary, got {type(response)}")
            return None

        structured = {}

        # Handle score (supports both 'score' and legacy 's' key)
        score_value = response.get("score", response.get("s"))
        if score_value is not None:
            try:
                structured["score"] = float(score_value)
            except (ValueError, TypeError):
                logger.error("Score must be a number")
                return None
        else:
            logger.error("Missing required field: score")
            return None

        # Handle justification (supports both 'justification' and legacy 'j' key)
        justification_value = response.get("justification", response.get("j"))
        if justification_value:
            if isinstance(justification_value, str):
                structured["justification"] = justification_value
            else:
                logger.error("Justification must be a string")
                return None
        else:
            logger.error("Missing required field: justification")
            return None

        # Handle confidence (supports both 'confidence' and legacy 'c' key)
        # Confidence is optional, so don't fail if missing
        confidence_value = response.get("confidence", response.get("c"))
        if confidence_value is not None:
            try:
                confidence = float(confidence_value)
                if not 0 <= confidence <= 1:
                    logger.warning(
                        "Confidence should be between 0 and 1, got: %s", confidence
                    )
                structured["confidence"] = confidence
            except (ValueError, TypeError):
                logger.warning("Confidence must be a number, ignoring invalid value")
                # Don't fail the entire parsing for confidence issues

        return structured

    def extract_json_string_from_response(
        self, raw_response: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from between ```json ... ``` markers in the response.

        Args:
            raw_response: String containing the full response with JSON markers

        Returns:
            Parsed JSON object (dict) if found and valid, otherwise None.
        """
        # First try to find JSON between ```json and ``` markers
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, raw_response, re.DOTALL)

        if match:
            fenced_json_str = match.group(1).strip()
            try:
                return json.loads(fenced_json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"Invalid JSON found inside fenced code block: {e}")
                # Continue to try other parsing methods
            except Exception as e:
                logger.debug(f"Unexpected error parsing fenced JSON: {e}")

        # If no markers found, try to parse the raw string as JSON
        try:
            # Verify it can be parsed as valid JSON
            response = json.loads(raw_response)
            return response
        except json.JSONDecodeError as e:
            logger.debug(f"Raw response is not valid JSON: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error parsing raw response as JSON: {e}")

        return None

    def validate_score(self, score: int) -> bool:
        """Validate that a score is within acceptable range"""
        return 1 <= score <= 5

    def get_score_description(self, score: int) -> str:
        """Get human-readable description for a score"""
        score_map = {1: "Very Poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
        return score_map.get(score, "Unknown")
