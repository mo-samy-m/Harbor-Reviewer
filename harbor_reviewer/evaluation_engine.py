"""
Evaluation Engine Module
------------------------
Purpose:
- Orchestrates the evaluation flow for each criterion
- Builds prompts → sends to LLM → parses rating & justification → checks early-exit logic
- Manages evaluation criteria and scoring logic
- Coordinates between prompt builder, LLM caller, and response parser
- Handles evaluation strategies and early termination rules
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union

from shared.utils.log import setup_logger

from .aggregator import ScoreAggregator
from .input_loader import InputLoader
from .llm_caller import LLMCaller
from .models import (
    CriterionResult,
    EvaluationContext,
    EvaluationOutput,
    EvaluationProgress,
    EvaluationResult,
    EvaluationStrategy,
    IndividualRecordManifest,
    LLMBatch,
    LLMRequest,
)
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser
from .harbor_response_parser import HarborResponseParser

logger = setup_logger(__name__)


class EvaluationEngine:
    """Main evaluation engine that orchestrates the evaluation process"""

    def __init__(self, config_dir: str = "configs", input_loader: InputLoader = None, disagreement_loader=None):
        """
        Initialize the evaluation engine

        Args:
            config_dir: Path to the configuration directory
            input_loader: Optional InputLoader instance to use (for consistency with runner)
            disagreement_loader: Optional DisagreementLoader instance for disagreement analysis
        """
        if input_loader:
            self.input_loader = input_loader
        else:
            self.input_loader = InputLoader(config_dir)

        self.prompt_builder = PromptBuilder(config_dir)
        self.llm_caller = LLMCaller(config_dir)
        self.response_parser = HarborResponseParser(config_dir)  # Use Harbor parser for pass/fail
        self.score_aggregator = ScoreAggregator(config_dir)
        self.disagreement_loader = disagreement_loader

        # Load configuration
        self.evaluator_config = self.input_loader.load_evaluator_config()
        self.criteria_config = self.input_loader.load_criteria_config()

        # Note: Statistics are calculated from EvaluationResult data when needed

        # Note: Run-level tracking is handled by runner.py
        # This engine focuses on individual record evaluation only

        logger.info("Evaluation Engine initialized successfully")

    async def evaluate_record(
        self,
        record_data: Dict[str, Any],
        record_id: str,
        criteria: Optional[List[str]] = None,
        strategy: Union[str, EvaluationStrategy] = EvaluationStrategy.FULL,
    ) -> EvaluationOutput:
        """
        Evaluate a single record against specified criteria

        Args:
            record_data: Data for the record to evaluate
            record_id: Unique identifier for the record
            criteria: List of criteria to evaluate (None for all)
            strategy: Evaluation strategy (full, selective, adaptive)

        Returns:
            EvaluationOutput containing result and manifest
        """
        # Validate and convert strategy
        if isinstance(strategy, EvaluationStrategy):
            strategy_value = strategy
        elif isinstance(strategy, str):
            try:
                strategy_value = EvaluationStrategy(strategy)
            except ValueError:
                valid_strategies = [s.value for s in EvaluationStrategy]
                raise ValueError(
                    f"Invalid strategy '{strategy}'. Valid strategies: {valid_strategies}"
                )
        else:
            raise TypeError(
                f"Strategy must be string or EvaluationStrategy, got {type(strategy)}"
            )

        # Determine criteria to evaluate
        if criteria is None:
            criteria = list(self.criteria_config.criteria.keys())

        # Create evaluation context
        context = EvaluationContext(
            record_id=record_id,
            record_data=record_data,
            criteria_to_evaluate=criteria,
            evaluation_strategy=strategy_value,
            early_exit_rules=self._get_early_exit_rules(strategy_value),
        )

        logger.info(
            f"Starting evaluation for record {record_id} with {len(criteria)} criteria"
        )

        # Execute evaluation
        result, manifest = await self._execute_evaluation(context)

        return EvaluationOutput(result=result, manifest=manifest)

    async def evaluate_batch(
        self,
        records: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None,
        strategy: Union[str, EvaluationStrategy] = EvaluationStrategy.FULL,
        max_parallel: Optional[int] = None,
    ) -> tuple[List[EvaluationResult], List[IndividualRecordManifest]]:
        """
        Evaluate multiple records in parallel

        Args:
            records: List of records to evaluate
            criteria: List of criteria to evaluate (None for all)
            strategy: Evaluation strategy
            max_parallel: Maximum parallel evaluations

        Returns:
            List of EvaluationResult objects and manifests
        """
        if not records:
            return [], []

        # Validate and convert strategy
        if isinstance(strategy, EvaluationStrategy):
            strategy_value = strategy
        elif isinstance(strategy, str):
            try:
                strategy_value = EvaluationStrategy(strategy)
            except ValueError:
                valid_strategies = [s.value for s in EvaluationStrategy]
                raise ValueError(
                    f"Invalid strategy '{strategy}'. Valid strategies: {valid_strategies}"
                )
        else:
            raise TypeError(
                f"Strategy must be string or EvaluationStrategy, got {type(strategy)}"
            )

        # Validate max_parallel
        if max_parallel is not None:
            if not isinstance(max_parallel, int):
                raise TypeError(
                    f"max_parallel must be an integer, got {type(max_parallel)}"
                )
            if max_parallel <= 0:
                raise ValueError(f"max_parallel must be positive, got {max_parallel}")

        # Determine criteria to evaluate
        if criteria is None:
            criteria = list(self.criteria_config.criteria.keys())

        # Create a high-level batch for tracking the entire evaluation session
        session_batch = LLMBatch(requests=[])
        logger.info(
            f"Starting evaluation session {session_batch.batch_id} for {len(records)} records"
        )

        # Create evaluation coroutines (not tasks)
        coroutines = []
        for i, record_data in enumerate(records):
            record_id = record_data.get("record_id", str(uuid.uuid4()))
            coroutine = self.evaluate_record(
                record_data, record_id, criteria, strategy_value
            )
            coroutines.append(coroutine)

        # Execute with concurrency limit
        if max_parallel:
            semaphore = asyncio.Semaphore(max_parallel)

            async def limited_coroutine(coro):
                async with semaphore:
                    return await coro

            limited_coroutines = [limited_coroutine(coro) for coro in coroutines]
            results = await asyncio.gather(*limited_coroutines, return_exceptions=True)
        else:
            results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        evaluation_results = []
        record_manifests = []
        successful_evaluations = 0
        failed_evaluations = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed for record {i}: {result}")
                failed_evaluations += 1
                # Create error result and manifest
                error_result = EvaluationResult(
                    record_id=str(uuid.uuid4()),
                    criterion_results={},
                    parsing_errors=[f"Evaluation failed: {str(result)}"],
                )
                error_manifest = IndividualRecordManifest(
                    record_id=error_result.record_id,
                )
                evaluation_results.append(error_result)
                record_manifests.append(error_manifest)
            else:
                successful_evaluations += 1
                evaluation_output = result  # Unpack the EvaluationOutput
                evaluation_results.append(evaluation_output.result)
                record_manifests.append(evaluation_output.manifest)

        # Log session completion
        logger.info(
            f"Completed evaluation session {session_batch.batch_id}: {successful_evaluations} successful, {failed_evaluations} failed"
        )

        return evaluation_results, record_manifests

    async def _execute_evaluation(
        self, context: EvaluationContext
    ) -> tuple[EvaluationResult, IndividualRecordManifest]:
        """
        Execute evaluation for a single record

        Args:
            context: Evaluation context

        Returns:
            EvaluationResult with all criterion evaluations
        """
        import time

        # Start timing for individual record
        start_time = time.time()

        criterion_results = {}
        raw_responses = {}
        model_used = None
        llm_requests_made = 0
        llm_responses_errors = 0
        total_tokens_used = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_response_size_chars = 0
        early_exit_triggered = False
        early_exit_reason = None

        # Track progress
        progress = EvaluationProgress(
            total_criteria=len(context.criteria_to_evaluate),
            completed_criteria=0,
            failed_criteria=0,
        )

        # Store disagreement opinions per criterion for later use in CriterionResult
        # Initialize before the if/else so it's available in both code paths
        criterion_opinions = {}

        # Process criteria based on evaluation strategy
        if context.evaluation_strategy == EvaluationStrategy.FULL:
            # Build all prompts first for batch processing
            llm_requests = []
            criterion_to_request_map = {}
            criterion_to_prompt_map = {}
            
            for criterion_name in context.criteria_to_evaluate:
                # Get disagreement opinions if loader is available
                disagreement_opinions = None
                if self.disagreement_loader:
                    record_id = context.record_data.get("record_id") or context.record_data.get("input_record_id")
                    repo_name = context.record_data.get("repo_name") or context.record_data.get("input_repo_name") or context.record_data.get("repo")
                    pr_number = context.record_data.get("pr_number") or context.record_data.get("input_pr_number") or context.record_data.get("pr_id")
                    instance_id = context.record_data.get("instance_id") or context.record_data.get("input_instance_id")
                    
                    disagreement_opinions = self.disagreement_loader.get_opinions(
                        record_id=record_id,
                        repo_name=repo_name,
                        pr_number=pr_number,
                        instance_id=instance_id,
                        criterion_name=criterion_name,
                    )
                    # Store for later use in CriterionResult (even if None, so columns are populated)
                    criterion_opinions[criterion_name] = disagreement_opinions
                
                # Build prompt for criterion
                prompt_obj = self.prompt_builder.build_prompt(
                    criterion_name, context.record_data, disagreement_opinions=disagreement_opinions
                )
                if not prompt_obj:
                    logger.warning(
                        f"Failed to build prompt for criterion {criterion_name}"
                    )
                    continue

                # Convert Prompt object to expected format
                prompt = prompt_obj.to_llm_format()

                # Create LLM request
                request = LLMRequest(
                    prompt=prompt,
                    model_name=self.evaluator_config.llm.primary_model,
                    request_id=f"{context.record_id}_{criterion_name}",
                )
                llm_requests.append(request)
                criterion_to_request_map[criterion_name] = request
                criterion_to_prompt_map[criterion_name] = prompt
                llm_requests_made += 1

            # Process all requests in batch
            if llm_requests:
                # Create LLMBatch for tracking and logging
                batch = LLMBatch(requests=llm_requests)
                logger.info(
                    f"Processing batch {batch.batch_id} for record {context.record_id} with {len(llm_requests)} criteria"
                )

                # Send batch to LLM caller
                completed_batch = await self.llm_caller.call_llm_batch(batch)

                # Process responses using paired request-response data
                for i, (request, llm_response) in enumerate(
                    completed_batch.get_request_response_pairs()
                ):
                    # Get criterion name from the original list (maintains order)
                    criterion_name = context.criteria_to_evaluate[i]
                    progress.current_criterion = criterion_name

                    try:
                        # Track model used
                        if model_used is None:
                            model_used = llm_response.model_used

                        # Parse response
                        if llm_response.error:
                            logger.error(
                                f"LLM error for {criterion_name}: {llm_response.error}"
                            )
                            llm_responses_errors += 1
                            criterion_result = CriterionResult(
                                criterion_name=criterion_name,
                                score=None,
                                justification=f"LLM error: {llm_response.error}",
                                raw_response=llm_response.raw_response,
                                parsing_errors=[llm_response.error],
                            )
                            progress.failed_criteria += 1
                        else:
                            # For batch processing, we already have the response, so just parse it
                            # No additional LLM calls needed - use all fallback methods
                            criterion_result = (
                                self.response_parser.parse_response_with_fallback_only(
                                    llm_response.raw_response,
                                    criterion_name,
                                    request.request_id,
                                )
                            )
                            # Add disagreement opinions to result if available
                            if criterion_name in criterion_opinions:
                                opinions = criterion_opinions[criterion_name]
                                if opinions is not None:  # Only set if opinions dict exists (not None)
                                    criterion_result.opinion_1_score = opinions.get("opinion_1_score")
                                    criterion_result.opinion_1_justification = opinions.get("opinion_1_justification")
                                    criterion_result.opinion_2_score = opinions.get("opinion_2_score")
                                    criterion_result.opinion_2_justification = opinions.get("opinion_2_justification")
                            progress.completed_criteria += 1

                        # Store results
                        criterion_results[criterion_name] = criterion_result
                        raw_responses[criterion_name] = llm_response.raw_response

                        # Debug LLM response token information
                        logger.debug(
                            f"LLM Response for {criterion_name}: total_tokens={llm_response.total_tokens}, prompt_tokens={llm_response.prompt_tokens}, completion_tokens={llm_response.completion_tokens}"
                        )

                        # Aggregate token usage and response metrics
                        if llm_response.total_tokens is not None:
                            total_tokens_used += llm_response.total_tokens
                        if llm_response.prompt_tokens is not None:
                            total_prompt_tokens += llm_response.prompt_tokens
                        if llm_response.completion_tokens is not None:
                            total_completion_tokens += llm_response.completion_tokens
                        if llm_response.response_size_chars is not None:
                            total_response_size_chars += (
                                llm_response.response_size_chars
                            )

                        # Log progress
                        logger.debug(
                            f"Completed {criterion_name} for {context.record_id}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Error evaluating {criterion_name} for {context.record_id}: {e}"
                        )
                        criterion_results[criterion_name] = CriterionResult(
                            criterion_name=criterion_name,
                            score=None,
                            justification=f"Evaluation error: {str(e)}",
                            parsing_errors=[str(e)],
                        )
                        progress.failed_criteria += 1

                # Log batch completion
                logger.info(
                    f"Completed batch {completed_batch.batch_id} for record {context.record_id}"
                )

        else:
            # Process criteria sequentially for early exit strategies
            for criterion_name in context.criteria_to_evaluate:
                # Check early exit conditions before processing each criterion
                should_exit, exit_reason = self._should_exit_early(
                    criterion_results, context.early_exit_rules
                )
                if should_exit:
                    logger.info(
                        f"Early exit triggered for record {context.record_id} before processing {criterion_name}"
                    )
                    early_exit_triggered = True
                    early_exit_reason = exit_reason
                    break

                # Get disagreement opinions if loader is available
                disagreement_opinions = None
                if self.disagreement_loader:
                    record_id = context.record_data.get("record_id") or context.record_data.get("input_record_id")
                    repo_name = context.record_data.get("repo_name") or context.record_data.get("input_repo_name") or context.record_data.get("repo")
                    pr_number = context.record_data.get("pr_number") or context.record_data.get("input_pr_number") or context.record_data.get("pr_id")
                    instance_id = context.record_data.get("instance_id") or context.record_data.get("input_instance_id")
                    
                    disagreement_opinions = self.disagreement_loader.get_opinions(
                        record_id=record_id,
                        repo_name=repo_name,
                        pr_number=pr_number,
                        instance_id=instance_id,
                        criterion_name=criterion_name,
                    )
                    # Store for later use in CriterionResult (even if None, so columns are populated)
                    criterion_opinions[criterion_name] = disagreement_opinions
                
                # Build prompt for criterion
                prompt_obj = self.prompt_builder.build_prompt(
                    criterion_name, context.record_data, disagreement_opinions=disagreement_opinions
                )
                if not prompt_obj:
                    logger.warning(
                        f"Failed to build prompt for criterion {criterion_name}"
                    )
                    continue

                # Convert Prompt object to expected format
                prompt = prompt_obj.to_llm_format()

                # Create LLM request
                request = LLMRequest(
                    prompt=prompt,
                    model_name=self.evaluator_config.llm.primary_model,
                    request_id=f"{context.record_id}_{criterion_name}",
                )
                llm_requests_made += 1

                try:
                    # Make single LLM call
                    llm_response = await self.llm_caller.call_llm_single(
                        prompt,
                        self.evaluator_config.llm.primary_model,
                        request.request_id,
                    )
                    progress.current_criterion = criterion_name

                    # Track model used
                    if model_used is None:
                        model_used = llm_response.model_used

                    # Parse response
                    if llm_response.error:
                        logger.error(
                            f"LLM error for {criterion_name}: {llm_response.error}"
                        )
                        llm_responses_errors += 1
                        criterion_result = CriterionResult(
                            criterion_name=criterion_name,
                            score=None,
                            justification=f"LLM error: {llm_response.error}",
                            raw_response=llm_response.raw_response,
                            parsing_errors=[llm_response.error],
                        )
                        progress.failed_criteria += 1
                    else:
                        # Use retry-enabled parsing method
                        criterion_result = (
                            await self.response_parser.parse_response_with_retries(
                                self.llm_caller,
                                prompt,
                                self.evaluator_config.llm.primary_model,
                                criterion_name,
                                request.request_id,
                            )
                        )
                        # Add disagreement opinions to result if available
                        if criterion_name in criterion_opinions:
                            opinions = criterion_opinions[criterion_name]
                            if opinions is not None:  # Only set if opinions dict exists (not None)
                                criterion_result.opinion_1_score = opinions.get("opinion_1_score")
                                criterion_result.opinion_1_justification = opinions.get("opinion_1_justification")
                                criterion_result.opinion_2_score = opinions.get("opinion_2_score")
                                criterion_result.opinion_2_justification = opinions.get("opinion_2_justification")
                        progress.completed_criteria += 1

                    # Store results
                    criterion_results[criterion_name] = criterion_result
                    raw_responses[criterion_name] = llm_response.raw_response

                    # Check early exit conditions after processing each response
                    should_exit, exit_reason = self._should_exit_early(
                        criterion_results, context.early_exit_rules
                    )
                    if should_exit:
                        logger.info(
                            f"Early exit triggered for record {context.record_id} after processing {criterion_name}"
                        )
                        early_exit_triggered = True
                        early_exit_reason = exit_reason
                        break

                    # Debug LLM response token information
                    logger.debug(
                        f"LLM Response for {criterion_name}: total_tokens={llm_response.total_tokens}, prompt_tokens={llm_response.prompt_tokens}, completion_tokens={llm_response.completion_tokens}"
                    )

                    # Aggregate token usage and response metrics
                    if llm_response.total_tokens is not None:
                        total_tokens_used += llm_response.total_tokens
                    if llm_response.prompt_tokens is not None:
                        total_prompt_tokens += llm_response.prompt_tokens
                    if llm_response.completion_tokens is not None:
                        total_completion_tokens += llm_response.completion_tokens
                    if llm_response.response_size_chars is not None:
                        total_response_size_chars += llm_response.response_size_chars

                    # Log progress
                    logger.debug(f"Completed {criterion_name} for {context.record_id}")

                except Exception as e:
                    logger.error(
                        f"Error evaluating {criterion_name} for {context.record_id}: {e}"
                    )
                    criterion_results[criterion_name] = CriterionResult(
                        criterion_name=criterion_name,
                        score=None,
                        justification=f"Evaluation error: {str(e)}",
                        parsing_errors=[str(e)],
                    )
                    progress.failed_criteria += 1

        # Generate manual review notes if there are failures
        # Solution-related criteria to exclude from manual_review_notes
        solution_criteria = {"solution_does_not_depend_on_runtime_internet", "solution_passes_all_tests"}
        
        # Check for failures (excluding solution criteria for manual_review_notes)
        non_solution_failures = self._get_failures(criterion_results, exclude_criteria=solution_criteria)
        all_failures = self._get_failures(criterion_results, exclude_criteria=set())
        
        # Generate manual_review_notes if there are non-solution failures
        if non_solution_failures:
            notes_result = await self._generate_manual_review_notes(
                context, criterion_results, exclude_criteria=solution_criteria, criterion_name="manual_review_notes"
            )
            if notes_result:
                criterion_results["manual_review_notes"] = notes_result
                llm_requests_made += 1
        
        # Generate manual_review_notes_all if there are any failures
        if all_failures:
            notes_all_result = await self._generate_manual_review_notes(
                context, criterion_results, exclude_criteria=set(), criterion_name="manual_review_notes_all"
            )
            if notes_all_result:
                criterion_results["manual_review_notes_all"] = notes_all_result
                llm_requests_made += 1

        # Calculate evaluation duration
        evaluation_duration_seconds = time.time() - start_time

        # Store entire original record data for output
        source_info = {}
        if context.record_data:
            # Store ALL original input data to preserve in output
            source_info = dict(context.record_data)
            
            # Handle multiple issue IDs (list) for backward compatibility
            issue_id = context.record_data.get("issue_number")
            if isinstance(issue_id, list):
                # Convert list to comma-separated string for source field
                source_info["issue_id"] = ",".join(map(str, issue_id))
            else:
                source_info["issue_id"] = issue_id

        # Debug token information
        logger.info(
            f"Record {context.record_id} - Tokens: {total_tokens_used}, Model: {model_used}"
        )
        logger.info(
            f"Token breakdown - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}"
        )

        # Get compatibility data if disagreement loader is available
        human_compatibility = None
        llm_compatibility = None
        if self.disagreement_loader:
            record_id_for_lookup = context.record_data.get("record_id") or context.record_data.get("input_record_id")
            repo_name_for_lookup = context.record_data.get("repo_name") or context.record_data.get("input_repo_name") or context.record_data.get("repo")
            pr_number_for_lookup = context.record_data.get("pr_number") or context.record_data.get("input_pr_number") or context.record_data.get("pr_id")
            instance_id_for_lookup = context.record_data.get("instance_id") or context.record_data.get("input_instance_id")
            
            compatibility_data = self.disagreement_loader.get_compatibility(
                record_id=record_id_for_lookup,
                repo_name=repo_name_for_lookup,
                pr_number=pr_number_for_lookup,
                instance_id=instance_id_for_lookup,
            )
            if compatibility_data:
                human_compatibility = compatibility_data.get("human_phase2_compatibility")
                llm_compatibility = compatibility_data.get("llm_phase2_compatibility")

        # Create evaluation result with full metadata
        evaluation_result = EvaluationResult(
            record_id=context.record_id,
            criterion_results=criterion_results,
            model_used=model_used,
            parsing_errors=[],
            source=source_info,
            criteria_evaluated=list(criterion_results.keys()),
            evaluation_duration_seconds=evaluation_duration_seconds,
            early_exit_triggered=early_exit_triggered,
            early_exit_reason=early_exit_reason,
            llm_requests_made=llm_requests_made,
            llm_responses_errors=llm_responses_errors,
            total_tokens_used=total_tokens_used,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_response_size_chars=total_response_size_chars,
            cost_estimate=self._estimate_cost(total_tokens_used, model_used),
            human_phase2_compatibility=human_compatibility,
            llm_phase2_compatibility=llm_compatibility,
        )

        # Create individual record manifest as per documentation
        record_manifest = IndividualRecordManifest(
            record_id=context.record_id,
            source=source_info,
            model_used=model_used,
            criteria_evaluated=list(criterion_results.keys()),
        )

        return evaluation_result, record_manifest

    def _should_exit_early(
        self,
        criterion_results: Dict[str, CriterionResult],
        early_exit_rules: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Check if evaluation should exit early based on deciding factors

        Args:
            criterion_results: Current criterion results
            early_exit_rules: Rules for early exit based on deciding factors

        Returns:
            Tuple of (should_exit, reason) where reason is None if no exit
        """
        if not early_exit_rules or not criterion_results:
            return False, None

        # Check deciding criteria failures
        deciding_criteria = early_exit_rules.get("deciding_criteria", [])
        deciding_thresholds = early_exit_rules.get("deciding_thresholds", {})
        max_failures = early_exit_rules.get("max_failures", 0)

        if not deciding_criteria or max_failures <= 0:
            return False, None

        # Count deciding factor failures and collect details
        deciding_failures = 0
        failed_criteria_details = []

        for criterion in deciding_criteria:
            if criterion in criterion_results:
                result = criterion_results[criterion]
                threshold = deciding_thresholds.get(criterion, 0)

                # Check if the criterion failed (score below threshold or invalid)
                if not result.is_valid() or result.score < threshold:
                    deciding_failures += 1
                    failure_reason = (
                        "invalid result"
                        if not result.is_valid()
                        else f"score {result.score} < threshold {threshold}"
                    )
                    failed_criteria_details.append(f"{criterion} ({failure_reason})")
                    logger.info(
                        f"Deciding factor {criterion} failed (score: {result.score} < {threshold})"
                    )

        # Exit early if we've reached the maximum number of deciding factor failures
        if deciding_failures >= max_failures:
            strategy = "SELECTIVE" if max_failures == 1 else "ADAPTIVE"
            reason = f"Early exit ({strategy}): {deciding_failures} deciding factors failed - {', '.join(failed_criteria_details)}"
            logger.info(reason)
            return True, reason

        return False, None

    def _get_early_exit_rules(self, strategy: EvaluationStrategy) -> Dict[str, Any]:
        """Get early exit rules based on evaluation strategy and deciding factors from criteria"""
        if strategy == EvaluationStrategy.FULL:
            # Full evaluation - no early exit
            return {}

        # Get deciding factors from criteria configuration
        deciding_criteria = []
        deciding_thresholds = {}

        for criterion_name, criterion_config in self.criteria_config.criteria.items():
            if getattr(criterion_config, "is_deciding_factor", False):
                deciding_criteria.append(criterion_name)
                threshold = getattr(criterion_config, "deciding_threshold", None)
                if threshold is not None:
                    deciding_thresholds[criterion_name] = threshold

        if not deciding_criteria:
            # No deciding factors configured, no early exit
            return {}

        # Build rules based on strategy
        if strategy == EvaluationStrategy.SELECTIVE:
            # Selective: exit early if any deciding factor fails
            return {
                "deciding_criteria": deciding_criteria,
                "deciding_thresholds": deciding_thresholds,
                "max_failures": 1,  # Exit on first deciding factor failure
            }
        elif strategy == EvaluationStrategy.ADAPTIVE:
            # Adaptive: exit early if multiple deciding factors fail
            return {
                "deciding_criteria": deciding_criteria,
                "deciding_thresholds": deciding_thresholds,
                "max_failures": 2,  # Exit after 2 deciding factor failures
            }
        else:
            # Unknown strategy, no early exit
            return {}

    def _estimate_cost(
        self, total_tokens: Optional[int], model_name: Optional[str]
    ) -> Optional[float]:
        """Estimate cost based on token usage and model pricing from configuration"""
        if not total_tokens or not model_name:
            logger.warning(
                f"Missing tokens ({total_tokens}) or model name ({model_name}) for cost estimation"
            )
            return None

        # Get cost estimation configuration
        cost_config = getattr(self.evaluator_config, "cost_estimation", None)
        if not cost_config:
            logger.warning("Cost estimation configuration not found")
            return None

        # Get token distribution ratios
        input_ratio = getattr(cost_config, "token_distribution", {}).get(
            "input_ratio", 0.7
        )
        output_ratio = getattr(cost_config, "token_distribution", {}).get(
            "output_ratio", 0.3
        )
        precision = getattr(cost_config, "precision", 6)

        # Get default pricing as fallback
        default_pricing = getattr(cost_config, "default_pricing", {})
        default_input_price = default_pricing.get("input", 0.01)
        default_output_price = default_pricing.get("output", 0.02)

        # Find model pricing from configuration
        model_pricing = None
        models_config = getattr(self.evaluator_config.llm, "models", {})

        for model_key, model_config in models_config.items():
            if model_key.lower() in model_name.lower():
                model_pricing = getattr(model_config, "pricing", None)
                break

        # Use default pricing if model pricing not found
        if not model_pricing:
            logger.warning(
                f"Model pricing not found for {model_name}, using default pricing"
            )
            model_pricing = {
                "input": default_input_price,
                "output": default_output_price,
            }

        # Calculate token distribution
        input_tokens = int(total_tokens * input_ratio)
        output_tokens = int(total_tokens * output_ratio)

        # Calculate cost
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost

        logger.debug(
            f"Cost calculation for {model_name}: {total_tokens} tokens = ${total_cost:.6f}"
        )
        return round(total_cost, precision)

    def create_batch_for_tracking(self, requests: List[LLMRequest] = None) -> LLMBatch:
        """
        Create a new LLMBatch for tracking purposes

        Args:
            requests: Optional list of requests to include in the batch

        Returns:
            LLMBatch with unique batch_id and timestamp
        """
        if requests is None:
            requests = []
        return LLMBatch(requests=requests)

    def create_batch_with_requests(self, requests: List[LLMRequest]) -> LLMBatch:
        """
        Create a new LLMBatch with requests for processing

        Args:
            requests: List of requests to include in the batch

        Returns:
            LLMBatch with requests and unique batch_id
        """
        return LLMBatch(requests=requests)

    def log_batch_progress(self, batch: LLMBatch, message: str, **kwargs):
        """
        Log batch-related information with consistent formatting

        Args:
            batch: The LLMBatch being tracked
            message: Log message
            **kwargs: Additional context for logging
        """
        context = (
            f"batch_id={batch.batch_id}, created_at={batch.created_at.isoformat()}"
        )
        if batch.is_complete():
            response_count = len(batch.responses) if batch.responses else 0
            context += f", completed=True, response_count={response_count}"
        else:
            context += f", request_count={len(batch.requests)}"

        if kwargs:
            context += f", {', '.join(f'{k}={v}' for k, v in kwargs.items())}"

        logger.info(f"[{context}] {message}")

    def _get_failures(
        self, 
        criterion_results: Dict[str, CriterionResult], 
        exclude_criteria: set = None
    ) -> Dict[str, CriterionResult]:
        """
        Get all failures from criterion results, optionally excluding certain criteria
        
        Args:
            criterion_results: Dictionary of criterion results
            exclude_criteria: Set of criterion names to exclude from failure check
            
        Returns:
            Dictionary of failed criterion results
        """
        if exclude_criteria is None:
            exclude_criteria = set()
        
        failures = {}
        for criterion_name, result in criterion_results.items():
            # Skip excluded criteria and manual review notes themselves
            if criterion_name in exclude_criteria or criterion_name in {"manual_review_notes", "manual_review_notes_all"}:
                continue
            
            # Check if this is a failure (score is 0 or None/invalid for pass/fail criteria)
            # For pass/fail criteria, score of 0 means fail, None/invalid also means fail
            if result.score == 0 or (result.score is None and result.is_valid() is False):
                failures[criterion_name] = result
        
        return failures
    
    def _format_evaluation_results_summary(
        self, 
        criterion_results: Dict[str, CriterionResult],
        exclude_criteria: set = None
    ) -> str:
        """
        Format evaluation results summary for manual review notes prompt
        
        Args:
            criterion_results: Dictionary of criterion results
            exclude_criteria: Set of criterion names to exclude from summary
            
        Returns:
            Formatted string summary of evaluation results
        """
        if exclude_criteria is None:
            exclude_criteria = set()
        
        summary_lines = []
        for criterion_name, result in criterion_results.items():
            # Skip excluded criteria and manual review notes themselves
            if criterion_name in exclude_criteria or criterion_name in {"manual_review_notes", "manual_review_notes_all"}:
                continue
            
            # Only include failures in the summary
            if result.score == 0 or (result.score is None and result.is_valid() is False):
                status = "FAIL"
                justification = result.justification or "No justification provided"
                summary_lines.append(f"- {criterion_name}: {status}")
                summary_lines.append(f"  Justification: {justification}")
        
        if not summary_lines:
            return "No failures found."
        
        return "\n".join(summary_lines)
    
    async def _generate_manual_review_notes(
        self,
        context: EvaluationContext,
        criterion_results: Dict[str, CriterionResult],
        exclude_criteria: set,
        criterion_name: str
    ) -> Optional[CriterionResult]:
        """
        Generate manual review notes based on evaluation failures
        
        Args:
            context: Evaluation context
            criterion_results: Dictionary of criterion results
            exclude_criteria: Set of criterion names to exclude from notes
            criterion_name: Name of the criterion to generate (manual_review_notes or manual_review_notes_all)
            
        Returns:
            CriterionResult with notes, or None if generation failed
        """
        try:
            # Format evaluation results summary
            results_summary = self._format_evaluation_results_summary(
                criterion_results, exclude_criteria=exclude_criteria
            )
            
            # Create context data
            # Start with a copy of record_data, or empty dict if None
            notes_context_data = dict(context.record_data) if context.record_data else {}
            
            # Ensure task_name is present (use record_id as fallback)
            if "task_name" not in notes_context_data:
                notes_context_data["task_name"] = context.record_id
            
            # Build the base prompt first
            try:
                prompt_obj = self.prompt_builder.build_prompt(
                    criterion_name, notes_context_data
                )
            except ValueError as e:
                logger.error(f"Failed to build prompt for {criterion_name}: {e}")
                return None
            
            if not prompt_obj:
                logger.warning(f"Failed to build prompt for {criterion_name}")
                return None
            
            # Inject the evaluation results summary into the user message
            # Append it to the context section of the user message
            user_message = prompt_obj.user_message.content
            if results_summary and results_summary != "No failures found.":
                # Find where to insert the summary (after Task Name in the context)
                summary_section = f"\nEvaluation Results:\n{results_summary}"
                # Insert before the response format section or at the end of context
                if "```json" in user_message:
                    # Insert before the JSON response format
                    user_message = user_message.replace("```json", summary_section + "\n\n```json")
                else:
                    # Append at the end
                    user_message = user_message + summary_section
                
                # Update the prompt with the modified user message
                from .prompt_builder import Message
                prompt_obj.user_message = Message(role="user", content=user_message)
            
            # Convert Prompt object to expected format
            prompt = prompt_obj.to_llm_format()
            
            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                model_name=self.evaluator_config.llm.primary_model,
                request_id=f"{context.record_id}_{criterion_name}",
            )
            
            # Make LLM call
            llm_response = await self.llm_caller.call_llm_single(
                prompt,
                self.evaluator_config.llm.primary_model,
                request.request_id,
            )
            
            # Parse response
            if llm_response.error:
                logger.error(f"LLM error for {criterion_name}: {llm_response.error}")
                return CriterionResult(
                    criterion_name=criterion_name,
                    score=0,
                    justification=f"LLM error: {llm_response.error}",
                    raw_response=llm_response.raw_response,
                    parsing_errors=[llm_response.error],
                )
            
            # Parse the response
            criterion_result = await self.response_parser.parse_response_with_retries(
                self.llm_caller,
                prompt,
                self.evaluator_config.llm.primary_model,
                criterion_name,
                request.request_id,
            )
            
            # For manual review notes, result is the text content
            # If parsing failed, try to extract the result field directly
            if not criterion_result or not criterion_result.result:
                # Try to extract from raw response
                import json
                import re
                try:
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response.raw_response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                        notes_text = data.get("result", "").strip()
                        if notes_text:
                            criterion_result = CriterionResult(
                                criterion_name=criterion_name,
                                result=notes_text,
                                score=0,  # Set score to 0 for manual review notes
                                justification=notes_text,
                                raw_response=llm_response.raw_response,
                            )
                except Exception as e:
                    logger.debug(f"Failed to extract notes from raw response: {e}")
            
            return criterion_result
            
        except Exception as e:
            logger.error(f"Error generating {criterion_name}: {e}")
            return CriterionResult(
                criterion_name=criterion_name,
                score=0,
                justification=f"Error generating notes: {str(e)}",
                parsing_errors=[str(e)],
            )

    async def close(self):
        """Clean up resources"""
        await self.llm_caller.close()
        logger.info("Evaluation Engine closed")


# Convenience functions for common evaluation patterns
async def evaluate_single_record(
    record_data: Dict[str, Any],
    record_id: str,
    criteria: Optional[List[str]] = None,
    config_dir: str = "configs",
    strategy: Union[str, EvaluationStrategy] = EvaluationStrategy.FULL,
) -> EvaluationOutput:
    """
    Convenience function to evaluate a single record

    Args:
        record_data: Data for the record to evaluate
        record_id: Unique identifier for the record
        criteria: List of criteria to evaluate (None for all)
        config_dir: Configuration directory path
        strategy: Evaluation strategy

    Returns:
        EvaluationOutput containing result and manifest
    """
    engine = EvaluationEngine(config_dir)
    try:
        return await engine.evaluate_record(record_data, record_id, criteria, strategy)
    finally:
        await engine.close()


async def evaluate_multiple_records(
    records: List[Dict[str, Any]],
    criteria: Optional[List[str]] = None,
    config_dir: str = "configs",
    max_parallel: Optional[int] = None,
    strategy: Union[str, EvaluationStrategy] = EvaluationStrategy.FULL,
) -> tuple[List[EvaluationResult], List[IndividualRecordManifest]]:
    """
    Convenience function to evaluate multiple records

    Args:
        records: List of records to evaluate
        criteria: List of criteria to evaluate (None for all)
        config_dir: Configuration directory path
        max_parallel: Maximum parallel evaluations
        strategy: Evaluation strategy

    Returns:
        List of EvaluationResult objects and manifests
    """
    engine = EvaluationEngine(config_dir)
    try:
        return await engine.evaluate_batch(records, criteria, strategy, max_parallel)
    finally:
        await engine.close()
