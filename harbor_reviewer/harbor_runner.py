"""
Harbor Tasks Evaluation Runner
Main entry point for Harbor Tasks automated review
"""
import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path for shared module imports
# When running as module: __file__ is harbor_reviewer/harbor_runner.py
# Project root is parent of harbor_reviewer directory
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Also add current working directory as fallback
_cwd = Path.cwd().resolve()
if str(_cwd) not in sys.path and _cwd != _project_root:
    sys.path.insert(0, str(_cwd))

from shared.utils.log import setup_logger

from .harbor_input_loader import HarborInputLoader
from .harbor_result_formatter import HarborResultFormatter
from .evaluation_engine import EvaluationEngine
from .input_loader import InputLoader
from .models import EvaluationRun

logger = setup_logger(__name__)


class HarborEvaluationRunner:
    """Main runner for Harbor Tasks evaluation pipeline"""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the Harbor evaluation runner

        Args:
            config_dir: Path to the configuration directory
        """
        self.config_dir = config_dir
        self.input_loader = InputLoader(config_dir)
        self.criteria_config = self.input_loader.load_criteria_config()
        self.result_formatter = HarborResultFormatter(self.criteria_config)

        logger.info("Harbor Evaluation Runner initialized successfully")

    async def run_evaluation(
        self,
        tasks_harbor_path: str,
        agent_outputs_path: str,
        output_path: Optional[str] = None,
        criteria: Optional[List[str]] = None,
        max_parallel: Optional[int] = None,
        run_id: Optional[str] = None,
        max_results: Optional[int] = None,
        filter_csv_path: Optional[str] = None,
    ) -> EvaluationRun:
        """
        Run a complete Harbor evaluation pipeline

        Args:
            tasks_harbor_path: Path to tasks-harbor folder
            agent_outputs_path: Path to agent outputs folder
            output_path: Path to save results (optional)
            criteria: List of criteria to evaluate (None for all)
            max_parallel: Maximum parallel evaluations
            run_id: Optional run identifier
            max_results: Maximum number of results to process

        Returns:
            EvaluationRun with complete run information
        """
        # Generate run ID
        if run_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            uuid_suffix = str(uuid.uuid4())[:8]
            run_id = f"harbor_eval_{timestamp}_{uuid_suffix}"

        # Create evaluation run
        start_time = datetime.now(timezone.utc)
        evaluation_run = EvaluationRun(
            run_id=run_id,
            start_time=start_time,
            end_time=start_time,
        )

        logger.info(f"Starting Harbor evaluation run: {run_id}")

        try:
            # Load Harbor task data
            harbor_loader = HarborInputLoader(
                tasks_harbor_path, agent_outputs_path, filter_csv_path=filter_csv_path
            )
            records = harbor_loader.load_all_tasks()

            # Limit records if specified
            if max_results and len(records) > max_results:
                logger.info(
                    f"Limiting input to first {max_results} records (out of {len(records)} total)"
                )
                records = records[:max_results]

            evaluation_run.records_processed = len(records)

            if not records:
                logger.warning("No records to evaluate")
                return evaluation_run

            # Determine criteria to evaluate
            if criteria is None:
                criteria = list(self.criteria_config.criteria.keys())
            else:
                # Validate that all requested criteria exist
                available_criteria = set(self.criteria_config.criteria.keys())
                requested_criteria = set(criteria)
                invalid_criteria = requested_criteria - available_criteria
                if invalid_criteria:
                    raise ValueError(
                        f"Invalid criteria: {invalid_criteria}. Available criteria: {sorted(available_criteria)}"
                    )
            evaluation_run.criteria_evaluated = criteria

            logger.info(
                f"Evaluating {len(records)} tasks with {len(criteria)} criteria"
            )

            # Run evaluation
            evaluation_engine = EvaluationEngine(
                self.config_dir, input_loader=self.input_loader
            )
            try:
                evaluation_results, record_manifests = (
                    await evaluation_engine.evaluate_batch(
                        records=records,
                        criteria=criteria,
                        strategy="full",
                        max_parallel=max_parallel,
                    )
                )

                # Track model used
                if evaluation_results:
                    evaluation_run.model_used = evaluation_results[0].model_used

            finally:
                await evaluation_engine.close()

            # Format results
            aggregated_results = []
            for result in evaluation_results:
                try:
                    formatted_result = self.result_formatter.format_evaluation_result(
                        result
                    )
                    aggregated_results.append(formatted_result)
                except Exception as e:
                    logger.error(f"Error formatting result for {result.record_id}: {e}")
                    error_result = {
                        "record_id": result.record_id,
                        "error": str(e),
                        "evaluation_status": "failed",
                    }
                    aggregated_results.append(error_result)

            # Save results
            if output_path:
                await self._save_results(
                    aggregated_results, output_path, evaluation_run
                )
                evaluation_run.output_file = output_path

            # Update run statistics
            evaluation_run.records_successful = sum(
                1
                for result in evaluation_results
                if result.criterion_results
                and any(r.is_valid() for r in result.criterion_results.values())
            )
            evaluation_run.records_failed = (
                evaluation_run.records_processed - evaluation_run.records_successful
            )

            logger.info(
                f"Evaluation completed: {evaluation_run.records_successful}/{evaluation_run.records_processed} successful"
            )

        except Exception as e:
            logger.error(f"Evaluation run failed: {e}")
            raise
        finally:
            evaluation_run.end_time = datetime.now(timezone.utc)

        return evaluation_run

    async def _save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        evaluation_run: EvaluationRun,
    ):
        """Save evaluation results to CSV file"""
        output_path = Path(output_path)

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Harbor Tasks Evaluation Runner")

    parser.add_argument(
        "--tasks-harbor",
        required=True,
        help="Path to tasks-harbor folder",
    )
    parser.add_argument(
        "--agent-outputs",
        required=True,
        help="Path to agent outputs folder",
    )
    parser.add_argument(
        "--output", "-o", help="Output CSV file path"
    )
    parser.add_argument(
        "--criteria", "-c",
        help="Criteria to evaluate (comma-separated or space-separated, e.g., 'root_cause_summary' or 'root_cause_summary,root_cause_category')"
    )
    parser.add_argument(
        "--max-parallel", "-p", type=int, help="Maximum parallel evaluations"
    )
    parser.add_argument(
        "--config-dir", default="configs", help="Configuration directory"
    )
    parser.add_argument("--run-id", help="Custom run identifier")
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of tasks to evaluate",
    )
    parser.add_argument(
        "--filter-csv",
        help="Path to CSV file with task names to filter (uses 'Task Name' column)",
    )

    args = parser.parse_args()

    # Set output path if not provided
    if not args.output:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = f"harbor_evaluation_results_{timestamp}.csv"

    async def run():
        runner = HarborEvaluationRunner(args.config_dir)

        try:
            # Parse criteria argument - handle both comma-separated and space-separated
            criteria_list = None
            if args.criteria:
                # Split by comma first, then by space (handles both formats)
                criteria_list = []
                for part in args.criteria.split(","):
                    criteria_list.extend([c.strip() for c in part.split() if c.strip()])
                if not criteria_list:
                    criteria_list = None

            evaluation_run = await runner.run_evaluation(
                tasks_harbor_path=args.tasks_harbor,
                agent_outputs_path=args.agent_outputs,
                output_path=args.output,
                criteria=criteria_list,
                max_parallel=args.max_parallel,
                run_id=args.run_id,
                max_results=args.max_results,
                filter_csv_path=args.filter_csv,
            )

            # Log report
            logger.info(
                f"Evaluation run completed: {evaluation_run.records_successful}/{evaluation_run.records_processed} successful"
            )
            logger.info(f"Results saved to: {evaluation_run.output_file}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            sys.exit(1)

    asyncio.run(run())


if __name__ == "__main__":
    main()

