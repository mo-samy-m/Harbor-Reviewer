"""
Template Renderer Module
-----------------------
Purpose:
- Handles template rendering and formatting
- Manages context variable fallbacks
- Separates template concerns from configuration concerns
- Provides clean interface for prompt generation
"""

from typing import Any, Dict

from shared.utils.log import setup_logger

logger = setup_logger(__name__)


class TemplateRenderer:
    """Handles template rendering with proper fallback logic"""

    def __init__(self):
        """Initialize the template renderer"""
        logger.info("TemplateRenderer initialized successfully")

    def format_context(self, context_format: str, **kwargs) -> str:
        """
        Format the context using provided keyword arguments with intelligent fallbacks.

        Args:
            context_format: The template string with placeholders
            **kwargs: Context variables for template rendering

        Returns:
            Formatted context string
        """
        try:
            return context_format.format(**kwargs)
        except KeyError as e:
            # Handle specific missing context variables gracefully by using default values or fallbacks
            missing_var = str(e).strip("'\"")

            # Check for specific fallbacks first
            # Handle gold_patch/code_content fallback (SWE-Bench standard -> legacy)
            if missing_var == "gold_patch":
                for fallback in ["code_content", "golden_patch"]:
                    if fallback in kwargs and kwargs[fallback] and str(kwargs[fallback]).lower() not in ["nan", "none", ""]:
                        logger.warning(f"Missing 'gold_patch', using '{fallback}' as fallback")
                        kwargs[missing_var] = kwargs[fallback]
                        return self.format_context(context_format, **kwargs)
            # Handle golden_tests/test_content fallback (SWE-Bench standard -> legacy)
            elif missing_var == "golden_tests":
                for fallback in ["test_content", "test_patch", "gold_tests"]:
                    if fallback in kwargs and kwargs[fallback] and str(kwargs[fallback]).lower() not in ["nan", "none", ""]:
                        logger.warning(f"Missing 'golden_tests', using '{fallback}' as fallback")
                        kwargs[missing_var] = kwargs[fallback]
                        return self.format_context(context_format, **kwargs)
            # Handle repo_name fallback
            elif missing_var == "repo_name":
                if "repo" in kwargs and kwargs["repo"] and str(kwargs["repo"]).lower() not in ["nan", "none", ""]:
                    logger.warning("Missing 'repo_name', using 'repo' as fallback")
                    kwargs[missing_var] = kwargs["repo"]
                    return self.format_context(context_format, **kwargs)
            # Handle problem_statement fallback (issue_body -> pr_body)
            elif missing_var == "problem_statement":
                if (
                    "issue_body" in kwargs
                    and kwargs["issue_body"]
                    and str(kwargs["issue_body"]).lower() not in ["nan", "none", ""]
                ):
                    logger.warning(
                        "Missing 'problem_statement', using 'issue_body' as fallback"
                    )
                    kwargs[missing_var] = kwargs["issue_body"]
                    return self.format_context(context_format, **kwargs)
                elif (
                    "pr_body" in kwargs
                    and kwargs["pr_body"]
                    and str(kwargs["pr_body"]).lower() not in ["nan", "none", ""]
                ):
                    logger.warning(
                        "Missing 'problem_statement', using 'pr_body' as fallback"
                    )
                    kwargs[missing_var] = kwargs["pr_body"]
                    return self.format_context(context_format, **kwargs)
                else:
                    logger.warning(
                        "Missing 'problem_statement', no fallback available, using empty string"
                    )
                    kwargs[missing_var] = ""
                    return self.format_context(context_format, **kwargs)
            # Handle problem_title fallback (issue_title -> pr_title)
            elif missing_var == "problem_title":
                if (
                    "issue_title" in kwargs
                    and kwargs["issue_title"]
                    and str(kwargs["issue_title"]).lower() not in ["nan", "none", ""]
                ):
                    logger.warning(
                        "Missing 'problem_title', using 'issue_title' as fallback"
                    )
                    kwargs[missing_var] = kwargs["issue_title"]
                    return self.format_context(context_format, **kwargs)
                elif (
                    "pr_title" in kwargs
                    and kwargs["pr_title"]
                    and str(kwargs["pr_title"]).lower() not in ["nan", "none", ""]
                ):
                    logger.warning(
                        "Missing 'problem_title', using 'pr_title' as fallback"
                    )
                    kwargs[missing_var] = kwargs["pr_title"]
                    return self.format_context(context_format, **kwargs)
                else:
                    logger.warning(
                        "Missing 'problem_title', no fallback available, using empty string"
                    )
                    kwargs[missing_var] = ""
                    return self.format_context(context_format, **kwargs)
            # Handle problem_labels fallback (issue_labels -> pr_labels)
            elif missing_var == "problem_labels":
                if (
                    "issue_labels" in kwargs
                    and kwargs["issue_labels"]
                    and str(kwargs["issue_labels"]).lower() not in ["nan", "none", ""]
                ):
                    logger.warning(
                        "Missing 'problem_labels', using 'issue_labels' as fallback"
                    )
                    kwargs[missing_var] = kwargs["issue_labels"]
                    return self.format_context(context_format, **kwargs)
                elif (
                    "pr_labels" in kwargs
                    and kwargs["pr_labels"]
                    and str(kwargs["pr_labels"]).lower() not in ["nan", "none", ""]
                ):
                    logger.warning(
                        "Missing 'problem_labels', using 'pr_labels' as fallback"
                    )
                    kwargs[missing_var] = kwargs["pr_labels"]
                    return self.format_context(context_format, **kwargs)
                else:
                    logger.warning(
                        "Missing 'problem_labels', no fallback available, using empty string"
                    )
                    kwargs[missing_var] = ""
                    return self.format_context(context_format, **kwargs)
            # Handle evaluation_results_summary fallback (for manual review notes)
            elif missing_var == "evaluation_results_summary":
                logger.warning("Missing 'evaluation_results_summary', using default value")
                kwargs[missing_var] = "No failures found."
                return self.format_context(context_format, **kwargs)
            # Apply default values for other specific variables
            elif (
                missing_var.startswith("repo_")
                or missing_var == "issue_labels"
                or missing_var == "issue_types"
                or missing_var == "issue_status"
                or missing_var == "pr_labels"
                or missing_var == "pr_has_linked_issues"
            ):
                logger.warning(
                    f"Missing context variable '{missing_var}', using default value"
                )
                kwargs[missing_var] = ""
                # Retry formatting with default value - this will handle multiple missing variables
                return self.format_context(context_format, **kwargs)
            else:
                # For other variables, still raise the error
                raise ValueError(f"Missing required context variable: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting context: {e}")

    def render_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        Render a template with context data.

        Args:
            template: Template string with placeholders
            context: Context variables for rendering

        Returns:
            Rendered template string
        """
        return self.format_context(template, **context)
