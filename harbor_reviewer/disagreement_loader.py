"""
Disagreement Loader Module
--------------------------
Purpose:
- Loads disagreement analysis CSV from step 6 compare mode
- Provides multi-level lookup for matching records
- Extracts opinions (scores and justifications) for each criterion
- Handles column name variations and criterion name mapping
"""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.utils.log import setup_logger

logger = setup_logger(__name__)


class DisagreementLoader:
    """Loads and provides lookup for disagreement analysis data"""

    def __init__(self, csv_path: str):
        """
        Initialize the disagreement loader

        Args:
            csv_path: Path to the disagreement analysis CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Disagreement analysis CSV not found: {csv_path}")

        # Load CSV data
        logger.info(f"Loading disagreement analysis data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} records from disagreement analysis CSV")

        # Create multi-level lookup dictionaries
        self._build_lookups()

    def _build_lookups(self):
        """Build multi-level lookup dictionaries for record matching"""
        # Primary lookup: record_id -> row index
        self.record_id_lookup: Dict[str, int] = {}
        # Secondary lookup: (repo_name, pr_number) -> row index
        self.repo_pr_lookup: Dict[Tuple[str, str], int] = {}
        # Tertiary lookup: instance_id -> row index
        self.instance_id_lookup: Dict[str, int] = {}
        # Fallback lookup: repo_name -> list of row indices
        self.repo_name_lookup: Dict[str, List[int]] = {}

        for idx, row in self.df.iterrows():
            # Primary: record_id
            record_id = self._safe_get_value(row, "record_id")
            if record_id and pd.notna(record_id):
                record_id_str = str(record_id).strip()
                if record_id_str:
                    self.record_id_lookup[record_id_str] = idx

            # Secondary: (repo_name, pr_number)
            repo_name = self._safe_get_value(row, "repo_name")
            pr_id = self._safe_get_value(row, "pr_id")
            if repo_name and pr_id and pd.notna(repo_name) and pd.notna(pr_id):
                repo_str = str(repo_name).strip()
                pr_str = str(pr_id).strip()
                if repo_str and pr_str:
                    self.repo_pr_lookup[(repo_str, pr_str)] = idx
            
            # Also create repo_name-only lookup for fallback matching
            if repo_name and pd.notna(repo_name):
                repo_str = str(repo_name).strip()
                if repo_str and repo_str not in self.repo_name_lookup:
                    self.repo_name_lookup[repo_str] = []
                if repo_str:
                    self.repo_name_lookup[repo_str].append(idx)

            # Tertiary: instance_id
            instance_id = self._safe_get_value(row, "instance_id")
            if instance_id and pd.notna(instance_id):
                instance_id_str = str(instance_id).strip()
                if instance_id_str:
                    self.instance_id_lookup[instance_id_str] = idx

        logger.info(
            f"Built lookups: {len(self.record_id_lookup)} record_ids, "
            f"{len(self.repo_pr_lookup)} repo+pr pairs, "
            f"{len(self.instance_id_lookup)} instance_ids"
        )

    def _safe_get_value(self, row: pd.Series, column: str) -> Optional[Any]:
        """Safely get value from row, handling missing columns"""
        if column not in row.index:
            return None
        value = row[column]
        if pd.isna(value):
            return None
        return value

    def _normalize_criterion_name(self, criterion_name: str) -> str:
        """
        Normalize criterion name to match CSV column naming convention

        Args:
            criterion_name: Step 3 criterion name (e.g., "problem_clarity")

        Returns:
            Normalized base name for CSV columns (e.g., "problem_clarity")
        """
        # Convert to lowercase and normalize separators
        normalized = criterion_name.lower().replace(" ", "_").replace("-", "_")
        # Remove multiple underscores
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized.strip("_")

    def _find_row(self, record_id: Optional[str], repo_name: Optional[str],
                  pr_number: Optional[str], instance_id: Optional[str]) -> Optional[pd.Series]:
        """
        Find a row using flexible matching priority

        Args:
            record_id: Record ID to match
            repo_name: Repository name
            pr_number: PR number
            instance_id: Instance ID

        Returns:
            Matching row as pandas Series, or None if not found
        """
        # Priority 1: record_id
        if record_id:
            record_id_str = str(record_id).strip()
            if record_id_str and record_id_str in self.record_id_lookup:
                idx = self.record_id_lookup[record_id_str]
                logger.debug(f"Matched by record_id: {record_id_str}")
                return self.df.iloc[idx]

        # Priority 2: instance_id (moved up since it's more reliable than repo+pr)
        if instance_id:
            instance_id_str = str(instance_id).strip()
            if instance_id_str and instance_id_str in self.instance_id_lookup:
                idx = self.instance_id_lookup[instance_id_str]
                logger.debug(f"Matched by instance_id lookup: {instance_id_str}")
                return self.df.iloc[idx]

        # Priority 3: (repo_name, pr_number) - only if pr_number is not NaN
        if repo_name and pr_number and not (isinstance(pr_number, float) and pd.isna(pr_number)):
            repo_str = str(repo_name).strip()
            pr_str = str(pr_number).strip()
            if repo_str and pr_str:
                key = (repo_str, pr_str)
                if key in self.repo_pr_lookup:
                    idx = self.repo_pr_lookup[key]
                    logger.debug(f"Matched by repo+pr: {repo_str}, {pr_str}")
                    return self.df.iloc[idx]

        # Priority 4: Try matching by instance_id pattern in CSV (if CSV has instance_id column)
        if instance_id and "instance_id" in self.df.columns:
            instance_id_str = str(instance_id).strip()
            # Try to find by instance_id column directly (case-insensitive, exact match)
            # Filter out NaN values first
            non_null_mask = self.df["instance_id"].notna()
            if non_null_mask.any():
                matching_rows = self.df[
                    non_null_mask & 
                    (self.df["instance_id"].astype(str).str.strip().str.lower() == instance_id_str.lower())
                ]
                if len(matching_rows) > 0:
                    logger.debug(f"Matched by instance_id column search: {instance_id_str}")
                    return matching_rows.iloc[0]

        # Priority 5: Fallback to repo_name alone (if pr_number is missing)
        if repo_name:
            repo_str = str(repo_name).strip()
            if repo_str in self.repo_name_lookup:
                matching_indices = self.repo_name_lookup[repo_str]
                if len(matching_indices) == 1:
                    # Only one match, use it
                    logger.debug(f"Matched by repo_name alone (single match): {repo_str}")
                    return self.df.iloc[matching_indices[0]]
                elif len(matching_indices) > 1:
                    # Multiple matches - try to match by instance_id pattern if available
                    if instance_id and "instance_id" in self.df.columns:
                        instance_id_str = str(instance_id).strip()
                        # Try to find a match where instance_id contains the repo name pattern
                        for idx in matching_indices:
                            row_instance_id = self._safe_get_value(self.df.iloc[idx], "instance_id")
                            if row_instance_id:
                                row_instance_id_str = str(row_instance_id).strip()
                                # Check if instance_ids are similar (e.g., both contain repo name)
                                if instance_id_str.lower() in row_instance_id_str.lower() or row_instance_id_str.lower() in instance_id_str.lower():
                                    logger.debug(f"Matched by repo_name + instance_id pattern: {repo_str}, {instance_id_str}")
                                    return self.df.iloc[idx]
                    # If no instance_id match, use first match (best effort)
                    logger.warning(f"Multiple matches for repo_name {repo_str}, using first match")
                    return self.df.iloc[matching_indices[0]]

        return None

    def get_opinions(
        self,
        record_id: Optional[str] = None,
        repo_name: Optional[str] = None,
        pr_number: Optional[str] = None,
        instance_id: Optional[str] = None,
        criterion_name: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get opinions (scores and justifications) for a specific record and criterion

        Args:
            record_id: Record ID to match (priority 1)
            repo_name: Repository name (priority 2)
            pr_number: PR number (priority 2)
            instance_id: Instance ID (priority 3)
            criterion_name: Step 3 criterion name (e.g., "problem_clarity")

        Returns:
            Dictionary with opinion_1_score, opinion_1_justification,
            opinion_2_score, opinion_2_justification, or None if not found
        """
        if not criterion_name:
            logger.warning("No criterion_name provided to get_opinions")
            return None

        # Find matching row
        row = self._find_row(record_id, repo_name, pr_number, instance_id)
        if row is None:
            logger.debug(
                f"No matching row found for record_id={record_id}, "
                f"repo_name={repo_name}, pr_number={pr_number}, instance_id={instance_id}"
            )
            return None

        # Normalize criterion name for CSV column lookup
        base_name = self._normalize_criterion_name(criterion_name)

        # Try to find columns with variations
        # Pattern: human_{base_name}, llm_{base_name}, human_{base_name}_justification, llm_{base_name}_justification
        human_score_col = f"human_{base_name}"
        llm_score_col = f"llm_{base_name}"
        human_justification_col = f"human_{base_name}_justification"
        llm_justification_col = f"llm_{base_name}_justification"

        # Check if columns exist (try variations)
        human_score = None
        llm_score = None
        human_justification = None
        llm_justification = None

        # Mapping for criteria with different CSV column names
        criterion_to_csv_mapping = {
            "problem_complexity_expertise": "problem_complexity__expertise",
            "gold_patch_complexity_algo": "gold_patch_complexity__algo",
            "gold_patch_complexity_time": "gold_patch_complexity__time_to_resolve",
            "validity_problem_match": "validity__problem_match",
            "validity_gold_patch_alignment": "validity__gold_patch_alignment",
            "unit_test_validity_false_negatives": "validity__unit_test_false_negatives",
            "unit_test_validity_false_positives": "validity__unit_test_false_positives",
            "validity_problem_test_alignment": "validity__problem_&_tests_alignment_(rewrite_signal)",
        }
        
        # Use mapped name if available
        csv_base_name = criterion_to_csv_mapping.get(criterion_name, base_name)
        if csv_base_name != base_name:
            human_score_col = f"human_{csv_base_name}"
            llm_score_col = f"llm_{csv_base_name}"
            human_justification_col = f"human_{csv_base_name}_justification"
            llm_justification_col = f"llm_{csv_base_name}_justification"

        # Try exact match first
        if human_score_col in row.index:
            human_score = self._safe_get_value(row, human_score_col)
        if llm_score_col in row.index:
            llm_score = self._safe_get_value(row, llm_score_col)
        if human_justification_col in row.index:
            human_justification = self._safe_get_value(row, human_justification_col)
        if llm_justification_col in row.index:
            llm_justification = self._safe_get_value(row, llm_justification_col)

        # If not found, try case-insensitive search
        if human_score is None or llm_score is None:
            for col in row.index:
                col_lower = col.lower()
                if human_score is None and col_lower == human_score_col.lower():
                    human_score = self._safe_get_value(row, col)
                if llm_score is None and col_lower == llm_score_col.lower():
                    llm_score = self._safe_get_value(row, col)
                if human_justification is None and col_lower == human_justification_col.lower():
                    human_justification = self._safe_get_value(row, col)
                if llm_justification is None and col_lower == llm_justification_col.lower():
                    llm_justification = self._safe_get_value(row, col)
        
        # If still not found, try fuzzy matching (remove underscores, special chars)
        if human_score is None or llm_score is None:
            def normalize_for_match(s):
                """Normalize string for fuzzy matching"""
                s = s.lower().replace('_', '').replace('-', '').replace(' ', '').replace('&', '').replace('(', '').replace(')', '')
                return s
            
            target_base = normalize_for_match(base_name)
            for col in row.index:
                if col.startswith('human_') and not col.endswith('_label') and not col.endswith('_justification'):
                    col_base = normalize_for_match(col.replace('human_', ''))
                    if target_base == col_base and human_score is None:
                        human_score = self._safe_get_value(row, col)
                        logger.debug(f"Fuzzy matched {criterion_name} to {col}")
                if col.startswith('llm_') and not col.endswith('_label') and not col.endswith('_justification'):
                    col_base = normalize_for_match(col.replace('llm_', ''))
                    if target_base == col_base and llm_score is None:
                        llm_score = self._safe_get_value(row, col)
                        logger.debug(f"Fuzzy matched {criterion_name} to {col}")

        # Convert scores to appropriate types (handle float/int)
        if human_score is not None:
            try:
                human_score = int(float(human_score)) if pd.notna(human_score) else None
            except (ValueError, TypeError):
                human_score = None

        if llm_score is not None:
            try:
                llm_score = int(float(llm_score)) if pd.notna(llm_score) else None
            except (ValueError, TypeError):
                llm_score = None

        # Get justifications (default to empty string if missing)
        human_justification = (
            str(human_justification).strip()
            if human_justification is not None and pd.notna(human_justification)
            else ""
        )
        llm_justification = (
            str(llm_justification).strip()
            if llm_justification is not None and pd.notna(llm_justification)
            else ""
        )

        # Log warning if both scores are missing (but still return dict structure)
        if human_score is None and llm_score is None:
            # Check if columns exist but are empty
            cols_exist = human_score_col in row.index or llm_score_col in row.index
            if cols_exist:
                logger.warning(
                    f"Columns exist but scores are None for criterion {criterion_name} "
                    f"(base_name: {base_name}, columns: {human_score_col}, {llm_score_col})"
                )
            else:
                logger.warning(
                    f"Columns not found for criterion {criterion_name} (base_name: {base_name}, "
                    f"looking for: {human_score_col}, {llm_score_col}). "
                    f"Available human/llm columns: {[c for c in row.index if 'human_' in c.lower() or 'llm_' in c.lower()][:10]}"
                )

        # Always return dict structure if row was found (even if values are None)
        # This ensures columns are populated in the output
        return {
            "opinion_1_score": human_score,
            "opinion_1_justification": human_justification,
            "opinion_2_score": llm_score,
            "opinion_2_justification": llm_justification,
        }
    
    def get_compatibility(
        self,
        record_id: Optional[str] = None,
        repo_name: Optional[str] = None,
        pr_number: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get phase2 compatibility values for a specific record

        Args:
            record_id: Record ID to match (priority 1)
            repo_name: Repository name (priority 2)
            pr_number: PR number (priority 2)
            instance_id: Instance ID (priority 3)

        Returns:
            Dictionary with human_phase2_compatibility and llm_phase2_compatibility, or None if not found
        """
        # Find matching row
        row = self._find_row(record_id, repo_name, pr_number, instance_id)
        if row is None:
            logger.debug(
                f"No matching row found for compatibility lookup: record_id={record_id}, "
                f"repo_name={repo_name}, pr_number={pr_number}, instance_id={instance_id}"
            )
            return None

        # Get compatibility values
        human_compatibility = self._safe_get_value(row, "human_phase2_compatibility")
        llm_compatibility = self._safe_get_value(row, "llm_phase2_compatibility")
        
        # Try case-insensitive search if not found
        if human_compatibility is None or llm_compatibility is None:
            for col in row.index:
                col_lower = col.lower()
                if human_compatibility is None and col_lower == "human_phase2_compatibility":
                    human_compatibility = self._safe_get_value(row, col)
                if llm_compatibility is None and col_lower == "llm_phase2_compatibility":
                    llm_compatibility = self._safe_get_value(row, col)

        # Convert to string if not None
        human_compatibility = str(human_compatibility).strip() if human_compatibility is not None and pd.notna(human_compatibility) else None
        llm_compatibility = str(llm_compatibility).strip() if llm_compatibility is not None and pd.notna(llm_compatibility) else None

        if human_compatibility is None and llm_compatibility is None:
            logger.debug("No compatibility data found in disagreement analysis CSV")
            return None

        return {
            "human_phase2_compatibility": human_compatibility,
            "llm_phase2_compatibility": llm_compatibility,
        }

