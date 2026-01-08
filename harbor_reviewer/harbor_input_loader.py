"""
Harbor Tasks Input Loader
Loads task data from tasks-harbor folder and agent outputs
"""
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.utils.log import setup_logger

logger = setup_logger(__name__)


class HarborInputLoader:
    """Loads Harbor task data and agent outputs for evaluation"""

    def __init__(
        self,
        tasks_harbor_path: str,
        agent_outputs_path: str,
        filter_csv_path: Optional[str] = None,
    ):
        """
        Initialize the Harbor input loader

        Args:
            tasks_harbor_path: Path to tasks-harbor folder
            agent_outputs_path: Path to agent outputs folder (e.g., 11-12-2025-HarborTasks_batch_116_outputs)
            filter_csv_path: Optional path to CSV file with task names to filter (uses "Task Name" column)
        """
        self.tasks_harbor_path = Path(tasks_harbor_path)
        self.agent_outputs_path = Path(agent_outputs_path)

        if not self.tasks_harbor_path.exists():
            raise ValueError(f"Tasks harbor path does not exist: {tasks_harbor_path}")
        if not self.agent_outputs_path.exists():
            raise ValueError(f"Agent outputs path does not exist: {agent_outputs_path}")

        # Load filter list if provided
        self.allowed_task_names = None
        if filter_csv_path:
            self.allowed_task_names = self._load_filter_list(filter_csv_path)
            logger.info(f"Loaded {len(self.allowed_task_names)} task names from filter CSV")

        logger.info(f"Initialized HarborInputLoader with tasks: {tasks_harbor_path}")

    def load_task_data(self, task_name: str) -> Dict[str, Any]:
        """
        Load all data for a single task

        Args:
            task_name: Name of the task folder

        Returns:
            Dictionary with task data
        """
        task_path = self.tasks_harbor_path / task_name

        if not task_path.exists():
            raise ValueError(f"Task folder does not exist: {task_path}")

        data = {
            "task_name": task_name,
            "task_path": str(task_path),
        }

        # Load instruction.md
        instruction_path = task_path / "instruction.md"
        if instruction_path.exists():
            with open(instruction_path, "r", encoding="utf-8") as f:
                data["instruction"] = f.read()
        else:
            data["instruction"] = ""
            logger.warning(f"No instruction.md found for {task_name}")

        # Load tests
        tests_path = task_path / "tests"
        if tests_path.exists():
            test_files = []
            for test_file in tests_path.glob("*.py"):
                with open(test_file, "r", encoding="utf-8") as f:
                    test_files.append(f.read())
            data["tests"] = "\n\n".join(test_files)
        else:
            data["tests"] = ""
            logger.warning(f"No tests found for {task_name}")

        # Load Dockerfile
        dockerfile_path = task_path / "environment" / "Dockerfile"
        if dockerfile_path.exists():
            with open(dockerfile_path, "r", encoding="utf-8") as f:
                data["dockerfile"] = f.read()
        else:
            data["dockerfile"] = ""
            logger.warning(f"No Dockerfile found for {task_name}")

        # Load solution
        solution_path = task_path / "solution" / "solve.sh"
        if solution_path.exists():
            with open(solution_path, "r", encoding="utf-8") as f:
                data["solution"] = f.read()
        else:
            data["solution"] = ""
            logger.warning(f"No solution found for {task_name}")

        # Load oracle logs
        oracle_path = task_path / "verifier-oracle"
        if oracle_path.exists():
            stdout_path = oracle_path / "test-stdout.txt"
            stderr_path = oracle_path / "test-stderr.txt"
            if stdout_path.exists():
                with open(stdout_path, "r", encoding="utf-8") as f:
                    data["oracle_stdout"] = f.read()
            else:
                data["oracle_stdout"] = ""
            if stderr_path.exists():
                with open(stderr_path, "r", encoding="utf-8") as f:
                    data["oracle_stderr"] = f.read()
            else:
                data["oracle_stderr"] = ""
        else:
            data["oracle_stdout"] = ""
            data["oracle_stderr"] = ""
            logger.warning(f"No oracle logs found for {task_name}")

        return data

    def load_agent_outputs(self, task_name: str) -> Dict[str, Any]:
        """
        Load agent outputs for a task

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with agent output data
        """
        # Find the agent outputs folder structure
        # Usually: agent_outputs_path/11-12-2025-HarborTasks/task_name/
        agent_data = {
            "agent_stdout": "",
            "agent_stderr": "",
            "trajectory_summary": "",
        }

        # Try to find the task folder in agent outputs
        # The structure might vary, so we search recursively
        task_found = False
        for root, dirs, files in os.walk(self.agent_outputs_path):
            if Path(root).name == task_name:
                task_found = True
                task_output_path = Path(root)

                # Load verifier logs
                verifier_path = task_output_path / "verifier"
                if verifier_path.exists():
                    stdout_path = verifier_path / "test-stdout.txt"
                    stderr_path = verifier_path / "test-stderr.txt"
                    if stdout_path.exists():
                        with open(stdout_path, "r", encoding="utf-8") as f:
                            agent_data["agent_stdout"] = f.read()
                    if stderr_path.exists():
                        with open(stderr_path, "r", encoding="utf-8") as f:
                            agent_data["agent_stderr"] = f.read()

                # Load trajectory
                trajectory_path = task_output_path / "agent" / "trajectory.json"
                if trajectory_path.exists():
                    with open(trajectory_path, "r", encoding="utf-8") as f:
                        trajectory = json.load(f)
                        # Create a summary of the trajectory
                        agent_data["trajectory_summary"] = self._summarize_trajectory(
                            trajectory
                        )
                break

        if not task_found:
            logger.warning(f"No agent outputs found for {task_name}")

        return agent_data

    def _summarize_trajectory(self, trajectory: Dict[str, Any]) -> str:
        """
        Create a summary of the agent trajectory

        Args:
            trajectory: Trajectory JSON data

        Returns:
            Summary string
        """
        steps = trajectory.get("steps", [])
        if not steps:
            return "No steps in trajectory"

        summary_parts = []
        summary_parts.append(f"Total steps: {len(steps)}")

        # Get first and last steps
        if steps:
            first_step = steps[0]
            last_step = steps[-1]

            # Extract key information
            if "message" in first_step:
                first_msg = first_step["message"][:200] + "..." if len(first_step["message"]) > 200 else first_step["message"]
                summary_parts.append(f"First step: {first_msg}")

            if "observation" in last_step:
                obs = last_step.get("observation", {})
                if "results" in obs and obs["results"]:
                    last_result = str(obs["results"][0].get("content", ""))[:200]
                    summary_parts.append(f"Last result: {last_result}")

        return "\n".join(summary_parts)

    def _load_filter_list(self, csv_path: str) -> set:
        """
        Load task names from CSV file

        Args:
            csv_path: Path to CSV file with "Task Name" column

        Returns:
            Set of task names to include
        """
        task_names = set()
        csv_path = Path(csv_path)

        if not csv_path.exists():
            logger.warning(f"Filter CSV not found: {csv_path}")
            return task_names

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_name = row.get("Task Name", "").strip()
                    if task_name:
                        task_names.add(task_name)
        except Exception as e:
            logger.error(f"Error loading filter CSV: {e}")
            return task_names

        return task_names

    def load_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Load all tasks with their data and agent outputs

        Returns:
            List of task data dictionaries
        """
        tasks = []

        # Get all task folders
        if not self.tasks_harbor_path.exists():
            logger.error(f"Tasks harbor path does not exist: {self.tasks_harbor_path}")
            return tasks

        for task_folder in sorted(self.tasks_harbor_path.iterdir()):
            if not task_folder.is_dir():
                continue

            task_name = task_folder.name

            # Filter by allowed task names if filter is provided
            if self.allowed_task_names is not None:
                if task_name not in self.allowed_task_names:
                    logger.debug(f"Skipping task {task_name} (not in filter list)")
                    continue

            logger.info(f"Loading task: {task_name}")

            try:
                # Load task data
                task_data = self.load_task_data(task_name)

                # Load agent outputs
                agent_data = self.load_agent_outputs(task_name)

                # Merge data
                task_data.update(agent_data)

                # Add record_id for tracking
                task_data["record_id"] = task_name

                tasks.append(task_data)

            except Exception as e:
                logger.error(f"Error loading task {task_name}: {e}")
                continue

        logger.info(f"Loaded {len(tasks)} tasks")
        return tasks

