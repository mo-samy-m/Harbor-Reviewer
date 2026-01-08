"""
Harbor Response Parser
Adapts Step 3 response parser for Harbor pass/fail evaluation
"""
import json
import re
from typing import Optional

from shared.utils.log import setup_logger

from .models import CriterionResult
from .response_parser import ResponseParser

logger = setup_logger(__name__)


class HarborResponseParser(ResponseParser):
    """Response parser adapted for Harbor pass/fail evaluation"""

    def __init__(self, config_dir: str = "configs"):
        """Initialize Harbor response parser"""
        super().__init__(config_dir)
        # Add pass/fail patterns
        self.result_patterns = [
            r'result["\s]*:["\s]*["\']?(pass|fail)["\']?',
            r'"result"["\s]*:["\s]*["\']?(pass|fail)["\']?',
            r'outcome["\s]*:["\s]*["\']?(pass|fail)["\']?',
            r'verdict["\s]*:["\s]*["\']?(pass|fail)["\']?',
        ]

    def parse_response(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> CriterionResult:
        """
        Parse Harbor response (pass/fail or category)

        Args:
            raw_response: String response from the LLM
            criterion_name: Name of the criterion
            request_id: Optional request identifier

        Returns:
            CriterionResult with result (pass/fail/category) and justification
        """
        if not raw_response or not raw_response.strip():
            return CriterionResult(
                criterion_name=criterion_name,
                justification="No response received",
                raw_response=raw_response,
                parsing_errors=["Empty or missing response"],
            )

        # Try JSON parsing first
        json_result = self._try_json_parsing_harbor(
            raw_response, criterion_name, request_id
        )
        if json_result and json_result.is_valid():
            return json_result

        # Try regex extraction for pass/fail
        regex_result = self._try_regex_parsing_harbor(
            raw_response, criterion_name, request_id
        )
        if regex_result and regex_result.is_valid():
            return regex_result

        # Fallback to parent class parsing (for numeric scores if needed)
        return super().parse_response(raw_response, criterion_name, request_id)

    def _try_json_parsing_harbor(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> Optional[CriterionResult]:
        """Try parsing JSON response for Harbor format"""
        try:
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                # For root_cause_summary, we need a more robust pattern that handles multi-line strings
                # Find the opening brace, then try to parse until matching closing brace
                brace_start = raw_response.find('{')
                if brace_start == -1:
                    return None
                
                # Try to find the matching closing brace
                brace_count = 0
                brace_end = -1
                for i in range(brace_start, len(raw_response)):
                    if raw_response[i] == '{':
                        brace_count += 1
                    elif raw_response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i + 1
                            break
                
                if brace_end > brace_start:
                    json_str = raw_response[brace_start:brace_end]
                else:
                    return None

            # Parse JSON - handle potential issues with escaped characters
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues for multi-line summaries
                # For root_cause_summary, the summary might contain newlines or special chars
                if criterion_name == "root_cause_summary":
                    # Try a simpler approach - extract result field directly with regex
                    result_match = re.search(r'"result"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str, re.DOTALL)
                    if result_match:
                        result_text = result_match.group(1)
                        # Unescape JSON string
                        result_text = result_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                        # Create minimal valid JSON
                        data = {"result": result_text, "justification": ""}
                    else:
                        logger.debug(f"Could not extract result field for {criterion_name}: {e}")
                        return None
                else:
                    logger.debug(f"JSON parsing failed for {criterion_name}: {e}")
                    return None

            # Extract result (pass/fail or category or summary text)
            result = None
            if "result" in data:
                result_value = data["result"]
                if isinstance(result_value, str):
                    # For root_cause_summary, preserve original case (it's a multi-sentence summary)
                    # For other criteria, lowercase for consistency
                    if criterion_name == "root_cause_summary":
                        result = result_value.strip()
                    else:
                        result = result_value.lower().strip()
                elif isinstance(result_value, bool):
                    result = "pass" if result_value else "fail"
                elif isinstance(result_value, (int, float)):
                    result = "pass" if result_value > 0 else "fail"

            # Extract justification
            justification = data.get("justification", "").strip()
            if not justification:
                justification = data.get("reasoning", "").strip()
            if not justification:
                justification = data.get("explanation", "").strip()

            # For root_cause_category, result is the category name
            if criterion_name == "root_cause_category":
                if result and result not in [
                    "prompt-test mismatch",
                    "incomplete prompt",
                    "incorrect or over-scoped tests",
                    "environment/dependency issue",
                    "anti-cheat weakness",
                    "fair model failure",
                    "other",
                ]:
                    # Normalize category names
                    result_lower = result.lower()
                    if "mismatch" in result_lower or "prompt" in result_lower and "test" in result_lower:
                        result = "Prompt-test mismatch"
                    elif "incomplete" in result_lower or "missing" in result_lower:
                        result = "Incomplete prompt"
                    elif "test" in result_lower and ("incorrect" in result_lower or "over" in result_lower or "scope" in result_lower):
                        result = "Incorrect or over-scoped tests"
                    elif "environment" in result_lower or "dependency" in result_lower:
                        result = "Environment/dependency issue"
                    elif "anti" in result_lower or "cheat" in result_lower or "weakness" in result_lower:
                        result = "Anti-cheat weakness"
                    elif "fair" in result_lower or "model" in result_lower and "failure" in result_lower:
                        result = "Fair model failure"
                    else:
                        result = "Other"

            # For root_cause_summary, result is the summary text
            if criterion_name == "root_cause_summary":
                if not result and justification:
                    result = justification
                elif not result:
                    result = data.get("summary", "").strip()
                # For summary, if justification is empty, use result as justification too
                if not justification and result:
                    justification = result

            # For root_cause_summary, we only need result (summary text)
            # For other criteria, we need both result and justification
            if criterion_name == "root_cause_summary":
                if result and result.strip():
                    # Convert pass/fail to score for compatibility with existing is_valid
                    score = 1  # Set score to pass validation, result field contains actual value
                    return CriterionResult(
                        criterion_name=criterion_name,
                        result=result,
                        score=score,
                        justification=justification or result,  # Use result as justification if missing
                        raw_response=raw_response,
                    )
            elif result and justification:
                # Convert pass/fail to score for compatibility with existing is_valid
                score = None
                if result.lower() == "pass":
                    score = 1
                elif result.lower() == "fail":
                    score = 0
                # For root_cause_category, keep result as-is and set score to 1 to pass validation
                if criterion_name == "root_cause_category":
                    score = 1  # Set score to pass validation, result field contains actual value
                
                return CriterionResult(
                    criterion_name=criterion_name,
                    result=result,
                    score=score,
                    justification=justification,
                    raw_response=raw_response,
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"JSON parsing failed for {criterion_name}: {e}")
            return None

        return None

    def _try_regex_parsing_harbor(
        self, raw_response: str, criterion_name: str, request_id: Optional[str] = None
    ) -> Optional[CriterionResult]:
        """Try regex extraction for pass/fail results"""
        # Try to extract result
        result = None
        for pattern in self.result_patterns:
            match = re.search(pattern, raw_response, re.IGNORECASE)
            if match:
                result = match.group(1).lower()
                break

        # Extract justification
        justification = ""
        for pattern in self.justification_patterns:
            match = re.search(pattern, raw_response, re.IGNORECASE | re.DOTALL)
            if match:
                justification = match.group(1).strip()
                break

        # For root_cause_summary, we only need result (summary text)
        if criterion_name == "root_cause_summary":
            if result and result.strip():
                score = 1  # Set score to pass validation
                return CriterionResult(
                    criterion_name=criterion_name,
                    result=result,
                    score=score,
                    justification=justification or result,  # Use result as justification if missing
                    raw_response=raw_response,
                )
        elif result and justification:
            # Convert pass/fail to score for compatibility
            score = None
            if result.lower() == "pass":
                score = 1
            elif result.lower() == "fail":
                score = 0
            if criterion_name == "root_cause_category":
                score = 1  # Set score to pass validation
            
            return CriterionResult(
                criterion_name=criterion_name,
                result=result,
                score=score,
                justification=justification,
                raw_response=raw_response,
            )

        return None

