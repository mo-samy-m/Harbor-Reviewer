"""
Result Formatter Module
----------------------
Purpose:
- Formats evaluation results for output
- Determines Phase 2 compatibility status
- Handles rewrite criteria checking and binary flag setting
- Provides centralized result formatting logic for Step 3 evaluation

Architecture:
- Configuration-driven: Output columns are dynamically generated from criteria.yaml
- No hardcoded mappings: Eliminates brittle manual field mappings
- Maintainable: Adding new criteria requires only config changes, not code changes
- Consistent: All criteria follow {criterion_name}_score and {criterion_name}_justification pattern
"""

from datetime import datetime, timezone
from typing import Any, Dict

from shared.utils.log import setup_logger

from .models import Phase2Compatibility

logger = setup_logger(__name__)


class ResultFormatter:
    """Handles evaluation result formatting and compatibility determination"""

    def __init__(self, criteria_config: Any, disagreement_loader: Any = None):
        """
        Initialize the result formatter

        Args:
            criteria_config: Configuration object containing criteria and thresholds
            disagreement_loader: Optional disagreement loader for opinion tracking
        """
        self.criteria_config = criteria_config
        self.disagreement_loader = disagreement_loader
        logger.info("ResultFormatter initialized successfully")

    def check_rewrite_criteria(
        self, result: Any, rewrite_thresholds: Any
    ) -> Dict[str, bool]:
        """
        Check if any criteria fall below rewrite thresholds and determine rewrite flags.

        This method centralizes the logic for checking rewrite criteria to avoid duplication
        between format_evaluation_result and determine_compatibility methods.

        Args:
            result: Evaluation result containing criterion_results
            rewrite_thresholds: Configuration object containing rewrite threshold values

        Returns:
            Dictionary containing rewrite flags:
            - needs_rewrite_problem: True if problem_clarity score is below threshold OR validity_problem_test_alignment indicates problem rewrite needed
            - needs_rewrite_unit_tests: True if any unit test validity score is below threshold OR validity_problem_test_alignment indicates test rewrite needed
            - needs_rewrite_gold_patch: True if validity_gold_patch_alignment score is below threshold
            - any_rewrite_needed: True if any rewrite criteria is below threshold
        """
        rewrite_flags = {
            "needs_rewrite_problem": False,
            "needs_rewrite_unit_tests": False,
            "needs_rewrite_gold_patch": False,
            "any_rewrite_needed": False,
        }

        if not result or not result.criterion_results:
            return rewrite_flags

        # Track FP/FN unit test issues for smart logic
        fp_fn_flagged_unit_tests = False
        alignment_score = None

        # First pass: collect FP/FN and alignment data
        for criterion_name, criterion_result in result.criterion_results.items():
            if criterion_result.score is None:
                continue

            if criterion_name == "unit_test_validity_false_positives":
                if criterion_result.score < rewrite_thresholds.unit_test_validity_fp:
                    fp_fn_flagged_unit_tests = True
                    logger.debug(
                        f"Unit test FP validity below threshold: {criterion_result.score} < {rewrite_thresholds.unit_test_validity_fp}"
                    )

            elif criterion_name == "unit_test_validity_false_negatives":
                if criterion_result.score < rewrite_thresholds.unit_test_validity_fn:
                    fp_fn_flagged_unit_tests = True
                    logger.debug(
                        f"Unit test FN validity below threshold: {criterion_result.score} < {rewrite_thresholds.unit_test_validity_fn}"
                    )

            elif criterion_name == "validity_problem_test_alignment":
                # Always capture alignment score for smart logic, regardless of threshold
                alignment_score = criterion_result.score
                if (
                    criterion_result.score
                    < rewrite_thresholds.validity_problem_test_alignment
                ):
                    logger.debug(
                        f"Problem-test alignment validity below threshold: {criterion_result.score} < {rewrite_thresholds.validity_problem_test_alignment}"
                    )

        # Second pass: apply smart rewrite logic
        for criterion_name, criterion_result in result.criterion_results.items():
            if criterion_result.score is None:
                continue

            if criterion_name == "problem_clarity":
                if criterion_result.score < rewrite_thresholds.problem_clarity:
                    rewrite_flags["needs_rewrite_problem"] = True
                    rewrite_flags["any_rewrite_needed"] = True
                    logger.debug(
                        f"Problem clarity below threshold: {criterion_result.score} < {rewrite_thresholds.problem_clarity}"
                    )

            elif criterion_name == "validity_gold_patch_alignment":
                if (
                    criterion_result.score
                    < rewrite_thresholds.validity_gold_patch_alignment
                ):
                    rewrite_flags["needs_rewrite_gold_patch"] = True
                    rewrite_flags["any_rewrite_needed"] = True
                    logger.debug(
                        f"Gold patch alignment validity below threshold: {criterion_result.score} < {rewrite_thresholds.validity_gold_patch_alignment}"
                    )

            # Note: unit_test_validity_false_positives, unit_test_validity_false_negatives,
            # and validity_problem_test_alignment are handled in the smart logic below

        # ============================================================================
        # SMART REWRITE LOGIC (Prioritizes Alignment System)
        # ============================================================================
        # This logic resolves conflicts between old FP/FN rewrite flags and new
        # alignment-based rewrite flags. The alignment system takes precedence
        # for unit test rewrites because it provides more specific guidance.

        if fp_fn_flagged_unit_tests and alignment_score is not None:
            # FP/FN flagged unit tests - use alignment to determine what to actually rewrite
            if alignment_score in [0, 2, 4]:  # Rewrite tests
                rewrite_flags["needs_rewrite_unit_tests"] = True
                rewrite_flags["any_rewrite_needed"] = True
                logger.debug(
                    f"FP/FN flagged unit tests, alignment confirms: rewrite tests (score {alignment_score})"
                )
            elif alignment_score in [1, 3]:  # Rewrite problem
                rewrite_flags["needs_rewrite_problem"] = True
                rewrite_flags["any_rewrite_needed"] = True
                logger.debug(
                    f"FP/FN flagged unit tests, alignment says: rewrite problem (score {alignment_score})"
                )
            elif alignment_score == 5:  # No issue
                # If alignment says no issue but FP/FN flagged unit tests,
                # and problem clarity is high, then the issue is likely with tests, not problem
                if not rewrite_flags["needs_rewrite_problem"]:
                    # Check if problem clarity is high - if so, flag tests instead of problem
                    problem_clarity_score = None
                    for (
                        criterion_name,
                        criterion_result,
                    ) in result.criterion_results.items():
                        if (
                            criterion_name == "problem_clarity"
                            and criterion_result.score is not None
                        ):
                            problem_clarity_score = criterion_result.score
                            break

                    if problem_clarity_score is not None and problem_clarity_score >= 3:
                        # Problem is clear, so FP/FN issues are likely test problems
                        rewrite_flags["needs_rewrite_unit_tests"] = True
                        rewrite_flags["any_rewrite_needed"] = True
                        logger.debug(
                            f"FP/FN flagged unit tests, alignment says no issue, problem clarity high ({problem_clarity_score}) - flagging unit tests (score {alignment_score})"
                        )
                    else:
                        # Problem not clear, so flag it (something must be wrong)
                        rewrite_flags["needs_rewrite_problem"] = True
                        rewrite_flags["any_rewrite_needed"] = True
                        logger.debug(
                            f"FP/FN flagged unit tests, alignment says no issue, problem clarity low ({problem_clarity_score}) - flagging problem (score {alignment_score})"
                        )
                else:
                    logger.debug(
                        f"FP/FN flagged unit tests, alignment says no issue, problem already flagged (score {alignment_score})"
                    )
        elif fp_fn_flagged_unit_tests:
            # FP/FN flagged but no alignment data - use old logic
            rewrite_flags["needs_rewrite_unit_tests"] = True
            rewrite_flags["any_rewrite_needed"] = True
            logger.debug(
                "FP/FN flagged unit tests, no alignment data - using old logic"
            )
        elif alignment_score is not None:
            # No FP/FN issues, use alignment system normally
            if alignment_score == 0:
                # Total mismatch - rewrite both
                rewrite_flags["needs_rewrite_problem"] = True
                rewrite_flags["needs_rewrite_unit_tests"] = True
            elif alignment_score in [1, 3]:
                # Misaligned or partially misaligned - problem
                rewrite_flags["needs_rewrite_problem"] = True
            elif alignment_score in [2, 4]:
                # Misaligned or partially misaligned - tests
                rewrite_flags["needs_rewrite_unit_tests"] = True
            # Score 5 = fully aligned, no rewrite needed

            if alignment_score < 5:
                rewrite_flags["any_rewrite_needed"] = True
                logger.debug(
                    f"No FP/FN issues, using alignment system: score {alignment_score}"
                )

        return rewrite_flags

    def format_evaluation_result(self, result: Any, analysis: Any) -> Dict[str, Any]:
        """Format evaluation result for output"""
        # Calculate confidence score as average of individual criterion confidences
        confidence_scores = [
            cr.confidence
            for cr in result.criterion_results.values()
            if cr.confidence is not None and isinstance(cr.confidence, (int, float))
        ]
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else None
        )

        # Fix source data types and extract key fields as separate columns
        source_data = {}
        repo_name = None
        pr_number = None

        if result.source:
            repo_name = result.source.get("repo")
            pr_number = (
                str(result.source.get("pr_number"))
                if result.source.get("pr_number") is not None
                else None
            )

            source_data = {
                "repo": repo_name,
                "pr_number": pr_number,
                "issue_id": (
                    str(result.source.get("issue_id"))
                    if result.source.get("issue_id") is not None
                    else None
                ),
                "commit_hash": result.source.get("commit_hash"),
                "file_path": result.source.get("file_path"),
            }

        # Start with ALL original input columns (prefixed with "input_")
        formatted = {}
        if result.source:
            for key, value in result.source.items():
                # Prefix original input columns to avoid collisions with evaluation fields
                formatted[f"input_{key}"] = value
        
        # Add evaluation metadata fields
        formatted.update({
            "record_id": result.record_id,
            "repo_name": repo_name,
            "pr_number": pr_number,
            "evaluation_status": "success",
            "model_used": result.model_used,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "evaluation_duration_seconds": result.evaluation_duration_seconds,
            "early_exit_triggered": result.early_exit_triggered,
            "early_exit_reason": result.early_exit_reason,
            "llm_requests_made": result.llm_requests_made,
            "llm_responses_errors": result.llm_responses_errors,
            "total_tokens_used": result.total_tokens_used,
            "total_prompt_tokens": result.total_prompt_tokens,
            "total_completion_tokens": result.total_completion_tokens,
            "total_response_size_chars": result.total_response_size_chars,
            "cost_estimate": result.cost_estimate,
            "source": source_data,
            "confidence_score": (
                round(avg_confidence, 3)
                if avg_confidence is not None
                and isinstance(avg_confidence, (int, float))
                else None
            ),
            "criteria_evaluated": len(result.criterion_results),
            "criteria_missing": 0,
        })

        # ============================================================================
        # DYNAMIC FIELD GENERATION (Configuration-Driven Architecture)
        # ============================================================================
        # This replaces the old hardcoded approach where we manually mapped each
        # criterion to specific field names. Now the output columns are automatically
        # generated from the criteria configuration, making the system maintainable
        # and eliminating the need to update this code when adding new criteria.

        document_fields = {
            # ============================================================================
            # REPOSITORY METADATA FIELDS (Required by SWE-Bench Data Curation)
            # ============================================================================
            # These fields are required by the output specification but not yet implemented.
            # They should be populated by repository analysis logic in a future iteration.
            "repo_english_only": None,  # TODO: Implement English-only repository detection
            "repo_spoken_languages": None,  # TODO: Implement programming language detection
            "repo_topic": None,  # TODO: Implement repository topic classification
        }

        # Dynamically generate criterion score and justification fields from configuration
        # This automatically creates {criterion_name}_score and {criterion_name}_justification
        # fields for ALL criteria defined in criteria.yaml
        for criterion_name in self.criteria_config.criteria.keys():
            document_fields[f"{criterion_name}_score"] = None
            document_fields[f"{criterion_name}_justification"] = ""
            
            # Add opinion columns if disagreement loader is available
            # Use "human" and "llm" terminology in CSV (not "opinion_1" and "opinion_2")
            if self.disagreement_loader:
                document_fields[f"{criterion_name}_human_score"] = None
                document_fields[f"{criterion_name}_human_justification"] = ""
                document_fields[f"{criterion_name}_llm_score"] = None
                document_fields[f"{criterion_name}_llm_justification"] = ""
                document_fields[f"{criterion_name}_correct_opinion"] = ""  # "human", "llm", or "neither"

        # Add analysis fields
        document_fields.update(
            {
                "eval_avg_repo_score": analysis.eval_category_avgs.get("repo", 0.0),
                "eval_avg_problem_score": analysis.eval_category_avgs.get(
                    "problem", 0.0
                ),
                "eval_avg_gold_patch_score": analysis.eval_category_avgs.get(
                    "patch", 0.0
                ),
                "eval_avg_test_score": analysis.eval_category_avgs.get("tests", 0.0),
                "eval_overall_avg_score_unweighted": getattr(
                    analysis, "eval_overall_avg_score_unweighted", None
                ),
                "eval_overall_avg_score_weighted": getattr(
                    analysis, "eval_overall_avg_score_weighted", None
                ),
                "phase_2_compatibility": (
                    self.determine_compatibility(
                        analysis.eval_overall_avg_score_weighted, result
                    ).value
                    if hasattr(analysis, "eval_overall_avg_score_weighted")
                    and analysis.eval_overall_avg_score_weighted is not None
                    else None
                ),
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "analyzed_by_llm": True,  # This is LLM evaluation
                "analyzed_with_model": result.model_used,
                "needs_rewrite_problem": False,
                "needs_rewrite_unit_tests": False,
            }
        )
        
        # Add compatibility columns if disagreement loader is available
        if self.disagreement_loader:
            document_fields["human_phase2_compatibility"] = None
            document_fields["llm_phase2_compatibility"] = None
            document_fields["correct_compatibility"] = ""  # "human", "llm", or "neither"

        # ============================================================================
        # DYNAMIC CRITERION POPULATION (Replaces Hardcoded Mapping)
        # ============================================================================
        # OLD APPROACH (REMOVED): Had a massive hardcoded criterion_mapping dictionary
        # that manually mapped each criterion to specific field names. This was:
        # - Brittle: Had to manually update for each new criterion
        # - Error-prone: Easy to forget to add new criteria
        # - Unmaintainable: Violated DRY principle
        #
        # NEW APPROACH: Dynamically populate based on configuration
        # - Automatic: Works with any criterion defined in criteria.yaml
        # - Consistent: Follows {criterion_name}_score pattern
        # - Maintainable: Single source of truth (the config file)

        for criterion_name, criterion_result in result.criterion_results.items():
            score_col = f"{criterion_name}_score"
            justification_col = f"{criterion_name}_justification"
            document_fields[score_col] = criterion_result.score
            document_fields[justification_col] = criterion_result.justification

            # Add opinion columns if disagreement loader is available
            # Always populate these columns (even if None) for all criteria
            if self.disagreement_loader:
                # Get opinions from CriterionResult (stored as opinion_1/opinion_2 internally)
                human_score = getattr(criterion_result, 'opinion_1_score', None)  # opinion_1 = human
                human_justification = getattr(criterion_result, 'opinion_1_justification', None) or ""
                llm_score = getattr(criterion_result, 'opinion_2_score', None)  # opinion_2 = llm
                llm_justification = getattr(criterion_result, 'opinion_2_justification', None) or ""
                
                # Use "human" and "llm" terminology in CSV output
                document_fields[f"{criterion_name}_human_score"] = human_score
                document_fields[f"{criterion_name}_human_justification"] = human_justification
                document_fields[f"{criterion_name}_llm_score"] = llm_score
                document_fields[f"{criterion_name}_llm_justification"] = llm_justification
                
                # Determine which opinion was correct (if any)
                correct_opinion = "neither"
                if criterion_result.score is not None:
                    if human_score is not None and criterion_result.score == human_score:
                        correct_opinion = "human"
                    elif llm_score is not None and criterion_result.score == llm_score:
                        correct_opinion = "llm"
                document_fields[f"{criterion_name}_correct_opinion"] = correct_opinion

            # Handle additional fields for language detection
            if criterion_name == "repo_language_usage":
                if criterion_result.primary_language is not None:
                    document_fields["primary_spoken_language"] = (
                        criterion_result.primary_language
                    )
                if criterion_result.all_languages is not None:
                    # Convert list to comma-separated string for CSV output
                    document_fields["spoken_languages"] = ",".join(
                        criterion_result.all_languages
                    )

        # Add document fields to formatted result
        formatted.update(document_fields)

        # Set binary flags for needs_rewrite based on individual criterion scores
        # This must run AFTER phase_2_compatibility is set
        rewrite_thresholds = self.criteria_config.overall_assessment.rewrite_thresholds

        # Use centralized rewrite criteria checking logic
        rewrite_flags = self.check_rewrite_criteria(result, rewrite_thresholds)
        formatted["needs_rewrite_problem"] = rewrite_flags["needs_rewrite_problem"]
        formatted["needs_rewrite_unit_tests"] = rewrite_flags[
            "needs_rewrite_unit_tests"
        ]
        formatted["needs_rewrite_gold_patch"] = rewrite_flags["needs_rewrite_gold_patch"]

        # Add compatibility data if disagreement loader is available
        if self.disagreement_loader:
            human_compatibility = getattr(result, 'human_phase2_compatibility', None)
            llm_compatibility = getattr(result, 'llm_phase2_compatibility', None)
            formatted["human_phase2_compatibility"] = human_compatibility
            formatted["llm_phase2_compatibility"] = llm_compatibility
            
            # Determine which compatibility was correct
            correct_compatibility = "neither"
            current_compatibility = formatted.get("phase_2_compatibility")
            if current_compatibility is not None:
                if human_compatibility is not None and current_compatibility == human_compatibility:
                    correct_compatibility = "human"
                elif llm_compatibility is not None and current_compatibility == llm_compatibility:
                    correct_compatibility = "llm"
            formatted["correct_compatibility"] = correct_compatibility

        logger.debug(
            f"Final binary flags - needs_rewrite_problem: {formatted['needs_rewrite_problem']}, needs_rewrite_unit_tests: {formatted['needs_rewrite_unit_tests']}, needs_rewrite_gold_patch: {formatted['needs_rewrite_gold_patch']}"
        )

        return formatted

    def determine_compatibility(
        self, score: float, result: Any = None
    ) -> Phase2Compatibility:
        """Determine Phase 2 compatibility based on score and thresholds"""
        if score is None:
            return Phase2Compatibility.NEEDS_HUMAN_REVIEW

        thresholds = self.criteria_config.overall_assessment.compatibility_thresholds

        # First check for incompatible status (highest priority)
        if score < thresholds.needs_human_review:
            return Phase2Compatibility.INCOMPATIBLE

        # Check for needs_rewrite status if result is provided
        if result is not None and result.criterion_results:
            rewrite_thresholds = (
                self.criteria_config.overall_assessment.rewrite_thresholds
            )

            # Use centralized rewrite criteria checking logic
            rewrite_flags = self.check_rewrite_criteria(result, rewrite_thresholds)
            if rewrite_flags["any_rewrite_needed"]:
                return Phase2Compatibility.NEEDS_REWRITE

        # Standard compatibility logic
        if score >= thresholds.compatible:
            return Phase2Compatibility.COMPATIBLE
        elif score >= thresholds.needs_human_review:
            return Phase2Compatibility.NEEDS_HUMAN_REVIEW
        else:
            return Phase2Compatibility.INCOMPATIBLE
