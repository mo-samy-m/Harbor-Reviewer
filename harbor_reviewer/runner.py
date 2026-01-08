"""

Runner Module
------------
Purpose:
- Main flow controller for the evaluation pipeline
- Loads config, data, criteria → runs evaluation → aggregates → saves output
- Provides CLI interface and batch processing capabilities
- Handles data loading from parquet files and result persistence
- Manages the complete evaluation workflow
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from shared.storage_handler import StorageHandler
from shared.utils.data_processor import DataProcessor
from shared.utils.file_utils import get_input_files
from shared.utils.log import setup_logger

from .aggregator import ScoreAggregator
from .evaluation_engine import (
    EvaluationEngine,
)
from .input_loader import InputLoader
from .models import (
    EvaluationResult,
    EvaluationRun,
)
from .result_formatter import ResultFormatter

logger = setup_logger(__name__)


class EvaluationRunner:
    """Main runner for the evaluation pipeline"""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the evaluation runner

        Args:
            config_dir: Path to the configuration directory
        """
        self.config_dir = config_dir
        # Create InputLoader with specific file paths from config directory
        prompts_path = os.path.join(config_dir, "prompts_templates.yaml")
        criteria_path = os.path.join(config_dir, "criteria.yaml")
        llm_config_path = os.path.join(config_dir, "llm_caller.yaml")

        self.input_loader = InputLoader(
            prompts_templates_config_file_path=prompts_path,
            criteria_config_file_path=criteria_path,
            llm_evaluator_config_file_path=llm_config_path,
        )
        self.storage_handler = StorageHandler()

        # Load configuration
        self.evaluator_config = self.input_loader.load_evaluator_config()
        self.criteria_config = self.input_loader.load_criteria_config()

        # Initialize helper classes
        self.data_processor = DataProcessor(self.criteria_config)
        # ResultFormatter will be initialized in run_evaluation when disagreement_loader is available
        self.result_formatter = None

        logger.info("Evaluation Runner initialized successfully")

    async def run_evaluation(
        self,
        input_data: Union[str, List[Dict[str, Any]], pd.DataFrame],
        output_path: Optional[str] = None,
        criteria: Optional[List[str]] = None,
        strategy: str = "full",
        max_parallel: Optional[int] = None,
        run_id: Optional[str] = None,
        max_results: Optional[int] = None,
        file_format: Optional[str] = None,
        disagreement_analysis_path: Optional[str] = None,
    ) -> EvaluationRun:
        """
        Run a complete evaluation pipeline

        Args:
            input_data: Input data (file path, list of dicts, or DataFrame)
            output_path: Path to save results (optional)
            criteria: List of criteria to evaluate (None for all)
            strategy: Evaluation strategy (full, selective, adaptive)
            max_parallel: Maximum parallel evaluations
            run_id: Optional run identifier
            disagreement_analysis_path: Path to disagreement analysis CSV (optional)

        Returns:
            EvaluationRun with complete run information
        """
        # Load disagreement analysis data if provided
        disagreement_loader = None
        if disagreement_analysis_path:
            from .disagreement_loader import DisagreementLoader
            disagreement_loader = DisagreementLoader(disagreement_analysis_path)
            logger.info(f"Loaded disagreement analysis data from: {disagreement_analysis_path}")
        
        # Initialize ResultFormatter with disagreement_loader if available
        self.result_formatter = ResultFormatter(self.criteria_config, disagreement_loader=disagreement_loader)

        # Generate run ID with human-readable timestamp + UUID suffix for uniqueness
        if run_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            uuid_suffix = str(uuid.uuid4())[:8]  # First 8 chars of UUID
            run_id = f"eval_run_{timestamp}_{uuid_suffix}"

        # Create evaluation run
        start_time = datetime.now(timezone.utc)
        evaluation_run = EvaluationRun(
            run_id=run_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated in finally block
        )

        logger.info(f"Starting evaluation run: {run_id}")

        try:
            # Load input data
            records = await self._load_input_data(input_data)

            # Limit input records if max_results is specified
            if max_results and len(records) > max_results:
                logger.info(
                    f"Limiting input to first {max_results} records (out of {len(records)} total)"
                )
                records = records[:max_results]

            evaluation_run.records_processed = len(records)
            evaluation_run.input_files = self._get_input_files(input_data)

            if not records:
                logger.warning("No records to evaluate")
                return evaluation_run

            # Validate disagreement data if provided
            if disagreement_loader:
                missing_records = []
                for record in records:
                    record_id = record.get("record_id") or record.get("input_record_id")
                    repo_name = record.get("repo_name") or record.get("input_repo_name") or record.get("repo")
                    pr_number = record.get("pr_number") or record.get("input_pr_number") or record.get("pr_id")
                    instance_id = record.get("instance_id") or record.get("input_instance_id")
                    
                    # Try to find a match for at least one criterion
                    test_criterion = criteria[0] if criteria else list(self.criteria_config.criteria.keys())[0]
                    opinions = disagreement_loader.get_opinions(
                        record_id=record_id,
                        repo_name=repo_name,
                        pr_number=pr_number,
                        instance_id=instance_id,
                        criterion_name=test_criterion,
                    )
                    if opinions is None:
                        # Format pr_number for display (handle NaN)
                        pr_display = pr_number if pr_number and not (isinstance(pr_number, float) and pd.isna(pr_number)) else "NaN"
                        missing_records.append({
                            "record_id": record_id,
                            "repo_name": repo_name,
                            "pr_number": pr_display,
                            "instance_id": instance_id,
                        })
                
                if missing_records:
                    warning_msg = (
                        f"Found {len(missing_records)} records without matching disagreement data. "
                        f"These records will be evaluated without disagreement context. "
                        f"First few missing records: {missing_records[:3]}"
                    )
                    logger.warning(warning_msg)
                    # Continue evaluation - records without disagreement data will just not have opinions appended

            # Determine criteria to evaluate
            if criteria is None:
                criteria = list(self.criteria_config.criteria.keys())
            evaluation_run.criteria_evaluated = criteria

            logger.info(
                f"Evaluating {len(records)} records with {len(criteria)} criteria"
            )

            # Run evaluation
            evaluation_engine = EvaluationEngine(
                self.config_dir, input_loader=self.input_loader, disagreement_loader=disagreement_loader
            )
            try:
                evaluation_results, record_manifests = (
                    await evaluation_engine.evaluate_batch(
                        records=records,
                        criteria=criteria,
                        strategy=strategy,
                        max_parallel=max_parallel,
                    )
                )

                # Track model used
                if evaluation_results:
                    evaluation_run.model_used = evaluation_results[0].model_used

                # Note: Individual record manifests and prompts metadata are now handled separately
                # as they were removed from the EvaluationRun dataclass for better separation of concerns

            finally:
                await evaluation_engine.close()

            # Aggregate and analyze results
            aggregated_results = await self._aggregate_results(evaluation_results)

            # Save results
            if output_path:
                await self._save_results(
                    aggregated_results, output_path, evaluation_run, file_format
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

    async def _load_input_data(
        self, input_data: Union[str, Dict[str, str], List[Dict[str, Any]], pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Load input data from various sources"""
        if isinstance(input_data, str):
            # Single file - load directly
            return await self._load_from_file(input_data)
        elif isinstance(input_data, dict) and any(
            key.endswith("_file") for key in input_data.keys()
        ):
            # SWE-Bench multi-file input with separate file arguments
            logger.info(
                "Detected SWE-Bench multi-file input, performing field-based mapping"
            )
            return await self.data_processor.load_and_merge_swe_bench_files(input_data)
        elif isinstance(input_data, list):
            # Already a list of records
            return input_data
        elif isinstance(input_data, pd.DataFrame):
            # Convert DataFrame to list of dicts
            return input_data.to_dict("records")
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")

    async def _load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from file (parquet, csv, json)"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        if file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return df.to_dict("records")

    def _get_input_files(
        self, input_data: Union[str, List[str], List[Dict[str, Any]], pd.DataFrame]
    ) -> List[str]:
        """Get list of input file paths"""
        return get_input_files(input_data)

    async def _aggregate_results(
        self, evaluation_results: List[Any]
    ) -> List[Dict[str, Any]]:
        """Aggregate evaluation results with scoring analysis"""
        aggregator = ScoreAggregator(self.config_dir)
        aggregated_results = []

        for result in evaluation_results:
            try:
                # Get aggregation method from configuration
                aggregation_method = (
                    self.criteria_config.overall_assessment.weighting_method
                )

                # Perform scoring analysis
                analysis = aggregator.analyze_evaluation(result, aggregation_method)

                # Format result for output
                formatted_result = self.result_formatter.format_evaluation_result(
                    result, analysis
                )
                aggregated_results.append(formatted_result)

            except Exception as e:
                logger.error(f"Error aggregating result for {result.record_id}: {e}")
                # Add error result
                error_result = {
                    "record_id": result.record_id,
                    "error": str(e),
                    "evaluation_status": "failed",
                }
                aggregated_results.append(error_result)

        return aggregated_results

    async def _save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        evaluation_run: EvaluationRun,
        file_format: str = None,
    ):
        """Save evaluation results to file"""
        output_path = Path(output_path)

        # Handle different output formats
        if output_path.suffix.lower() == ".json":
            # For JSON, we need to include run metadata, so handle separately
            output_data = {
                "run_metadata": {
                    "run_id": evaluation_run.run_id,
                    "start_time": evaluation_run.start_time.isoformat(),
                    "end_time": evaluation_run.end_time.isoformat(),
                    "records_processed": evaluation_run.records_processed,
                    "records_successful": evaluation_run.records_successful,
                    "records_failed": evaluation_run.records_failed,
                    "criteria_evaluated": evaluation_run.criteria_evaluated,
                    "model_used": evaluation_run.model_used,
                },
                "results": results,
            }
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_path}")
        else:
            # Use StorageHandler for all other formats (parquet, csv, etc.)
            df = pd.DataFrame(results)
            saved_path = self.storage_handler.save_dataframe(
                df, output_path.name, str(output_path.parent), file_format=file_format
            )
            logger.info(f"Results saved to: {saved_path}")

    def generate_run_report(self, evaluation_run: EvaluationRun) -> Dict[str, Any]:
        """Generate a comprehensive run report"""
        duration = None
        if evaluation_run.end_time:
            duration = (
                evaluation_run.end_time - evaluation_run.start_time
            ).total_seconds()

        # Calculate duration in minutes for documentation format
        duration_minutes = None
        if duration:
            duration_minutes = f"{int(duration // 60)}m"

        report = {
            "run_id": evaluation_run.run_id,
            "input_files": evaluation_run.input_files,
            "model_used": evaluation_run.model_used,
            "num_records": evaluation_run.records_processed,
            "start_time": evaluation_run.start_time.isoformat(),
            "duration": duration_minutes,
            "output_file": evaluation_run.output_file,
        }

        return report

    def _calculate_run_statistics(
        self, evaluation_results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Calculate run statistics from evaluation results"""
        if not evaluation_results:
            return {}

        total_records = len(evaluation_results)
        successful_evaluations = sum(
            1
            for r in evaluation_results
            if r.criterion_results
            and any(cr.is_valid() for cr in r.criterion_results.values())
        )
        failed_evaluations = total_records - successful_evaluations
        early_exits = sum(1 for r in evaluation_results if r.early_exit_triggered)
        total_criteria = sum(len(r.criterion_results) for r in evaluation_results)
        total_duration_seconds = sum(
            r.evaluation_duration_seconds or 0 for r in evaluation_results
        )
        total_tokens = sum(r.total_tokens_used or 0 for r in evaluation_results)
        total_prompt_tokens = sum(
            r.total_prompt_tokens or 0 for r in evaluation_results
        )
        total_completion_tokens = sum(
            r.total_completion_tokens or 0 for r in evaluation_results
        )
        total_response_size_chars = sum(
            r.total_response_size_chars or 0 for r in evaluation_results
        )
        total_cost_estimate = sum(r.cost_estimate or 0 for r in evaluation_results)

        return {
            "total_records_processed": total_records,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "early_exits": early_exits,
            "total_criteria_evaluated": total_criteria,
            "average_criteria_per_record": (
                total_criteria / total_records if total_records > 0 else 0
            ),
            "total_evaluation_time_seconds": total_duration_seconds,
            "average_evaluation_time_seconds": (
                total_duration_seconds / total_records if total_records > 0 else 0
            ),
            "total_tokens_used": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_response_size_chars": total_response_size_chars,
            "total_cost_estimate_usd": round(total_cost_estimate, 6),
            "average_tokens_per_record": (
                total_tokens / total_records if total_records > 0 else 0
            ),
            "average_response_size_chars": (
                total_response_size_chars / total_records if total_records > 0 else 0
            ),
            "average_cost_per_record_usd": (
                round(total_cost_estimate / total_records, 6)
                if total_records > 0
                else 0
            ),
        }


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="SWE-Bench Evaluation Runner")

    # File input options - either single file or separate SWE-Bench files
    parser.add_argument(
        "--input", help="Single input file path (for non-SWE-Bench data)"
    )
    parser.add_argument("--prs-file", help="PRs data file path")
    parser.add_argument("--repos-file", help="Repositories data file path")
    parser.add_argument("--issues-file", help="Issues data file path")

    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--criteria", "-c", nargs="+", help="Criteria to evaluate")
    parser.add_argument(
        "--strategy",
        "-s",
        default="selective",
        choices=["full", "selective", "adaptive"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--max-parallel", "-p", type=int, help="Maximum parallel evaluations"
    )
    parser.add_argument(
        "--config-dir", default="configs", help="Configuration directory"
    )
    parser.add_argument("--run-id", help="Custom run identifier")
    parser.add_argument(
        "--format",
        choices=["parquet", "json", "csv"],
        default="parquet",
        help="Output format",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum number of individual results to save (useful for testing or limiting output size)",
    )
    parser.add_argument(
        "--disagreement-analysis",
        help="Path to disagreement analysis CSV from step 6 compare mode",
    )

    args = parser.parse_args()

    # Validate input arguments
    if (
        not args.input
        and not args.prs_file
        and not args.repos_file
        and not args.issues_file
    ):
        parser.error(
            "At least one input method must be specified (--input, --prs-file, --repos-file, or --issues-file)"
        )

    if args.input and (args.prs_file or args.repos_file or args.issues_file):
        parser.error("Cannot use --input with other file arguments")

    # Set output path if not provided
    if not args.output:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = f"evaluation_results_{timestamp}.{args.format}"

    async def run():
        runner = EvaluationRunner(args.config_dir)

        try:
            # Determine input data based on arguments
            if args.input:
                # Single file input
                input_data = args.input
            else:
                # SWE-Bench multi-file input
                input_data = {
                    "prs_file": args.prs_file,
                    "repos_file": args.repos_file,
                    "issues_file": args.issues_file,
                }
                # Remove None values
                input_data = {k: v for k, v in input_data.items() if v is not None}

                # Ensure at least one file is provided
                if not input_data:
                    parser.error("At least one SWE-Bench file must be specified")

            evaluation_run = await runner.run_evaluation(
                input_data=input_data,
                output_path=args.output,
                criteria=args.criteria,
                strategy=args.strategy,
                max_parallel=args.max_parallel,
                run_id=args.run_id,
                max_results=args.max_results,
                file_format=args.format,
                disagreement_analysis_path=args.disagreement_analysis,
            )

            # Generate and log report
            report = runner.generate_run_report(evaluation_run)
            logger.info(f"Evaluation run completed: {json.dumps(report, indent=2)}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            sys.exit(1)

    asyncio.run(run())


if __name__ == "__main__":
    main()
