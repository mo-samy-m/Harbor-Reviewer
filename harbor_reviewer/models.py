"""
Data models for the LLM Evaluator system.

This module contains all dataclasses and data structures used for step 3.

Separated from business logic for better maintainability and testing.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class EvaluationStrategy(Enum):
    """Evaluation strategies for the evaluation engine"""

    FULL = "full"
    SELECTIVE = "selective"
    ADAPTIVE = "adaptive"


class Phase2Compatibility(Enum):
    """Enum for Phase 2 compatibility status"""

    COMPATIBLE = "compatible"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    NEEDS_REWRITE = "needs_rewrite"
    INCOMPATIBLE = "incompatible"


@dataclass
class LLMRequest:
    """Represents a single LLM request"""

    prompt: List[Dict[str, str]]  # system and user messages
    model_name: str
    request_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 1


@dataclass
class LLMResponse:
    """Represents a single LLM response"""

    raw_response: str
    model_used: str
    latency_ms: float
    parsed_response: Optional[Dict[str, Any]] = field(default_factory=dict)
    retry_count: int = 0
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    request_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Token usage metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    # Response metadata
    response_size_chars: Optional[int] = None
    finish_reason: Optional[str] = None


@dataclass
class LLMBatch:
    """Represents a batch of LLM requests with responses"""

    requests: List[LLMRequest]
    responses: Optional[List[LLMResponse]] = field(default_factory=list)
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_complete(self) -> bool:
        """Check if the batch has been processed and has responses"""
        # Empty batch is always considered complete
        if len(self.requests) == 0:
            return True
        # For non-empty batches, check that responses exist and match request count
        return self.responses is not None and len(self.responses) == len(self.requests)

    def get_request_response_pairs(self) -> List[tuple[LLMRequest, LLMResponse]]:
        """Get paired request-response tuples"""
        if not self.is_complete():
            raise ValueError("Batch is not complete - responses not available")
        return list(zip(self.requests, self.responses))


@dataclass
class EvaluationContext:
    """Context for a single evaluation"""

    record_id: str
    record_data: Dict[str, Any]
    criteria_to_evaluate: List[str]
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.FULL
    early_exit_rules: Dict[str, Any] = field(default_factory=dict)
    max_criteria_per_record: Optional[int] = None


@dataclass
class EvaluationProgress:
    """Tracks progress of evaluation"""

    total_criteria: int
    completed_criteria: int
    failed_criteria: int
    current_criterion: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    estimated_completion: Optional[datetime] = None


@dataclass
class CriterionResult:
    """Structured result for a single evaluation criterion"""

    criterion_name: str
    justification: str
    score: Optional[int] = None
    result: Optional[str] = None  # For pass/fail or category results (Harbor)
    confidence: Optional[float] = None
    raw_response: Optional[str] = None
    parsing_errors: List[str] = field(default_factory=list)
    # Disagreement analysis opinions (if used)
    opinion_1_score: Optional[int] = None
    result: Optional[str] = None  # For pass/fail or category results (Harbor)  # Human opinion
    opinion_1_justification: Optional[str] = None
    opinion_2_score: Optional[int] = None
    result: Optional[str] = None  # For pass/fail or category results (Harbor)  # LLM opinion
    opinion_2_justification: Optional[str] = None

    def __post_init__(self):
        """Validate score is within valid range (0 or more) or None for failures"""
        if self.score is not None:
            if not isinstance(self.score, int):
                raise ValueError(
                    f"Score must be an integer or None, got {type(self.score)}"
                )
            if not self.score >= 0:
                raise ValueError(f"Score must be 0 or more, got {self.score}")

    def is_valid(self) -> bool:
        """Check if the result is valid"""
        has_score_or_result = self.score is not None or (self.result is not None and self.result.strip())
        return (
            has_score_or_result
            and len(self.justification.strip()) > 0
            and len(self.parsing_errors) == 0
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single record"""

    record_id: str
    criterion_results: Dict[str, CriterionResult]
    model_used: Optional[str] = None
    parsing_errors: List[str] = field(default_factory=list)

    # Individual record metadata for traceability
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Dict[str, Any] = field(default_factory=dict)  # repo, pr_id, issue_id info
    criteria_evaluated: List[str] = field(default_factory=list)
    evaluation_duration_seconds: Optional[float] = None
    early_exit_triggered: bool = False
    early_exit_reason: Optional[str] = None
    llm_requests_made: int = 0
    llm_responses_errors: int = 0
    total_tokens_used: Optional[int] = None
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_response_size_chars: Optional[int] = None
    cost_estimate: Optional[float] = None
    # Disagreement analysis compatibility (if used)
    human_phase2_compatibility: Optional[str] = None
    llm_phase2_compatibility: Optional[str] = None

    def __post_init__(self):
        # Set criteria_evaluated based on criterion_results if not provided
        if not self.criteria_evaluated and self.criterion_results:
            self.criteria_evaluated = list(self.criterion_results.keys())


@dataclass
class AggregationResult:
    """Result of score aggregation"""

    overall_score: float
    weighted_score: float
    unweighted_score: float
    score_breakdown: Dict[str, float]
    aggregation_method: str
    criteria_used: List[str]
    criteria_missing: List[str]
    confidence_score: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class IndividualRecordManifest:
    """Individual record metadata for traceability as per documentation"""

    record_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Dict[str, Any] = field(default_factory=dict)  # repo, pr_id, issue_id
    model_used: Optional[str] = None
    criteria_evaluated: List[str] = field(default_factory=list)


@dataclass
class EvaluationOutput:
    """Output from evaluation containing both result and manifest"""

    result: EvaluationResult
    manifest: IndividualRecordManifest


@dataclass
class EvaluationRun:
    """Represents a complete evaluation run"""

    run_id: str
    start_time: datetime
    end_time: datetime  # Required - always set when run completes or fails
    input_files: List[str] = field(default_factory=list)
    output_file: Optional[str] = None
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    criteria_evaluated: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
