"""
Harbor Result Formatter
Adapts the Step 3 result formatter for Harbor Tasks pass/fail evaluation
"""
from datetime import datetime, timezone
from typing import Any, Dict

from shared.utils.log import setup_logger

logger = setup_logger(__name__)


class HarborResultFormatter:
    """Formats Harbor evaluation results for CSV output with input_ prefix"""

    def __init__(self, criteria_config: Any):
        """
        Initialize the Harbor result formatter

        Args:
            criteria_config: Configuration object containing criteria
        """
        self.criteria_config = criteria_config
        logger.info("HarborResultFormatter initialized successfully")

    def format_evaluation_result(self, result: Any, analysis: Any = None) -> Dict[str, Any]:
        """
        Format evaluation result for Harbor CSV output

        Args:
            result: Evaluation result containing criterion_results
            analysis: Optional analysis object (not used for Harbor)

        Returns:
            Dictionary formatted for CSV output with input_ prefix
        """
        formatted = {}

        # Start with ALL original input columns (prefixed with "input_")
        if result.source:
            for key, value in result.source.items():
                # Prefix original input columns to avoid collisions with evaluation fields
                formatted[f"input_{key}"] = value

        # Add evaluation metadata
        formatted.update({
            "record_id": result.record_id,
            "evaluation_status": "success" if result.criterion_results else "failed",
            "model_used": result.model_used,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "evaluation_duration_seconds": result.evaluation_duration_seconds,
            "total_tokens_used": result.total_tokens_used,
            "cost_estimate": result.cost_estimate,
        })

        # Dynamically generate criterion result and justification fields
        # For Harbor, we use pass/fail instead of scores
        for criterion_name in self.criteria_config.criteria.keys():
            # Create result and justification columns
            formatted[f"{criterion_name}_result"] = None
            formatted[f"{criterion_name}_justification"] = ""

        # Populate criterion results
        for criterion_name, criterion_result in result.criterion_results.items():
            result_col = f"{criterion_name}_result"
            justification_col = f"{criterion_name}_justification"

            # For pass/fail criteria, use the result field
            # For root_cause_category and root_cause_summary, use result directly
            if criterion_name in ["root_cause_category", "root_cause_summary"]:
                # These are text/category results, not pass/fail
                formatted[result_col] = criterion_result.result if hasattr(criterion_result, 'result') else None
            else:
                # Convert score to pass/fail
                if criterion_result.score is not None:
                    # If score is a string "pass" or "fail", use it directly
                    if isinstance(criterion_result.score, str):
                        formatted[result_col] = criterion_result.score.lower()
                    # Otherwise, treat as boolean (True = pass, False = fail)
                    elif isinstance(criterion_result.score, bool):
                        formatted[result_col] = "pass" if criterion_result.score else "fail"
                    else:
                        # Numeric score: treat > 0 as pass
                        formatted[result_col] = "pass" if criterion_result.score > 0 else "fail"
                else:
                    formatted[result_col] = None

            formatted[justification_col] = criterion_result.justification or ""

        # Generate manual_review_notes if any check fails
        failed_checks = []
        for criterion_name, criterion_result in result.criterion_results.items():
            if criterion_name in ["root_cause_category", "root_cause_summary"]:
                continue  # Skip root cause fields
            result_value = formatted.get(f"{criterion_name}_result")
            if result_value == "fail":
                failed_checks.append(criterion_name)

        if failed_checks:
            formatted["manual_review_notes"] = f"Failed checks: {', '.join(failed_checks)}"
        else:
            formatted["manual_review_notes"] = ""

        # Add timestamp
        formatted["analyzed_at"] = datetime.now(timezone.utc).isoformat()

        return formatted

