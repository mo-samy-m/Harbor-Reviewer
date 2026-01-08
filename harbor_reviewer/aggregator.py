import logging
from dataclasses import dataclass, field
from typing import Dict, List

from .input_loader import InputLoader
from .models import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Aggregate the average scores for each criterion"""

    # Dynamic: normalized average score per-category based on the config file
    eval_category_avgs: Dict[str, float] = field(default_factory=dict)

    # Overall scores (on normalized 0-1 scale)
    eval_overall_avg_score_unweighted: float = 0.0
    eval_overall_avg_score_weighted: float = 0.0


class ScoreAggregator:
    def __init__(self, config_dir: str = "configs"):
        self.input_loader = InputLoader(config_dir)
        self.criteria_config = self.input_loader.criteria_config

        self.aggregation_method = (
            self.criteria_config.overall_assessment.weighting_method
        )
        allowed_methods = {
            "weighted_average",
            "unweighted_average",
        }  # Add more methods if needed
        if self.aggregation_method not in allowed_methods:
            raise ValueError(f"Invalid aggregation method: {self.aggregation_method}")

        self.categories_mapping = self._category_mapping()  # criterion -> category
        self.categories = set(self.categories_mapping.values())

        self.weights = self._extract_weights()
        self.score_ranges = self._extract_score_ranges()

    # ---------- config helpers ----------

    def _category_mapping(self) -> Dict[str, str]:
        """Extract category for each criterion from the configuration"""
        score_categories = {}
        for key, value in self.criteria_config.criteria.items():
            # Handle both Criterion objects and dictionaries
            if hasattr(value, "category"):
                category = value.category
            elif isinstance(value, dict):
                category = value.get("category", "uncategorized")
            else:
                category = "uncategorized"
            score_categories[key] = category

        bad_values = {None, "", "uncategorized"}
        if any(v in bad_values for v in score_categories.values()):
            logger.warning("Some criteria have undefined or empty categories.")
        return score_categories

    def _extract_weights(self) -> Dict[str, float]:
        weights = {}
        for key, value in self.criteria_config.criteria.items():
            # Handle both Criterion objects and dictionaries
            if hasattr(value, "weight"):
                weight = value.weight
            elif isinstance(value, dict):
                weight = value.get("weight", 0.0)
            else:
                weight = 0.0
            weights[key] = weight
        return weights

    def _extract_score_ranges(self) -> Dict[str, Dict[str, int]]:
        """Extract min and max scores for each criterion from the rubric"""
        score_ranges: Dict[str, Dict[str, int]] = {}
        for criterion, data in self.criteria_config.criteria.items():
            # Handle both Criterion objects and dictionaries
            if hasattr(data, "rubric"):
                rubric = data.rubric
            elif isinstance(data, dict):
                rubric = data.get("rubric", [])
            else:
                rubric = []

            if not rubric:
                logger.error(f"No rubric defined for criterion: {criterion}")

            # Handle both RubricItem objects and dictionaries
            if hasattr(rubric[0], "score"):
                min_score = min(item.score for item in rubric)
                max_score = max(item.score for item in rubric)
            else:
                min_score = min(item["score"] for item in rubric)
                max_score = max(item["score"] for item in rubric)

            score_ranges[criterion] = {"min": min_score, "max": max_score}
        return score_ranges

    # ---------- math helpers ----------

    def _normalize_score(self, criterion: str, score: int) -> float:
        """Normalize a score to a 0-1 range based on the criterion's rubric"""
        # Note: Score input is expected to be an integer within the rubric's range
        sr = self.score_ranges.get(criterion)
        if not sr:
            logger.error(f"Score range not found for criterion: {criterion}")
        lo, hi = sr["min"], sr["max"]
        if hi == lo:
            logger.warning(f"Max == min for criterion '{criterion}'. Returning 0.")
            return 0.0
        if score < lo:
            logger.error(
                f"Score {score} out of range (< min) for criterion '{criterion}' ({lo}-{hi})"
            )
        elif score > hi:
            logger.error(
                f"Score {score} out of range (> max) for criterion '{criterion}' ({lo}-{hi})"
            )
        normalized = (score - lo) / (hi - lo)
        return normalized

    def _avg(self, xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    # ---------- aggregation ----------

    def analyze_evaluation(
        self, evaluation_result: EvaluationResult, aggregation_method: str = None
    ) -> AggregationResult:
        """Get the EvaluationResult for a single record and aggregate it"""

        aggregation_method = aggregation_method or self.aggregation_method

        per_cat_scores: Dict[str, List[float]] = {c: [] for c in self.categories}
        all_normalized_scores: List[float] = []
        total_weighted_norm = 0.0
        total_weight = 0.0

        # Ingest every criterion
        for crit_name, crit_result in evaluation_result.criterion_results.items():
            raw_score = crit_result.score
            if raw_score is None:
                logger.warning(f"Skipping criterion '{crit_name}' with None score")
                continue

            category = self.categories_mapping.get(crit_name)
            if not category or category not in self.categories:
                logger.warning(
                    f"Criterion '{crit_name}' has undefined or unknown category '{category}'. Skipping."
                )
                continue

            # Normalize the score and get its weight
            norm_score = self._normalize_score(crit_name, raw_score)
            weight = self.weights.get(crit_name, 0.0)

            # Append to lists for calculation
            per_cat_scores[category].append(norm_score)
            all_normalized_scores.append(norm_score)
            total_weighted_norm += norm_score * weight
            total_weight += weight

        # 1. Per-category averages (normalized and unweighted)
        category_avgs = {
            cat: self._avg(vals) for cat, vals in per_cat_scores.items() if vals
        }

        # 2. Overall unweighted average (FIXED)
        # Calculated as a simple average of all normalized scores
        eval_overall_unweighted = self._avg(all_normalized_scores)

        # 3. Overall weighted average (conditionally calculated)
        eval_overall_weighted = 0.0
        if aggregation_method == "weighted_average":
            eval_overall_weighted = (
                (total_weighted_norm / total_weight) if total_weight > 0 else 0.0
            )
        elif aggregation_method == "unweighted_average":
            # If not weighted, it should be the same as the unweighted score
            eval_overall_weighted = eval_overall_unweighted
        else:
            # Note: Implement here other aggregation methods if needed
            raise NotImplementedError(
                f"Aggregation method '{aggregation_method}' not implemented"
            )

        return AggregationResult(
            eval_category_avgs=category_avgs,
            eval_overall_avg_score_unweighted=eval_overall_unweighted,
            eval_overall_avg_score_weighted=eval_overall_weighted,
        )
