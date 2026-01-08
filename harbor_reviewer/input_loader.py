import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import yaml

from shared.utils.log import setup_logger

# Set up logger
logger = setup_logger(__name__)


@dataclass
class RubricItem:
    """Represents a single rubric item with score, label, description, and examples."""

    score: int
    label: str
    description: str
    examples: List[str]

    def __post_init__(self):
        """Validate rubric item data."""
        if self.score is None:
            raise ValueError("Score must be provided")
        if not isinstance(self.score, int):
            raise ValueError(f"Score must be an integer, got {type(self.score)}")
        if not isinstance(self.label, str):
            raise ValueError(f"Label must be a string, got {type(self.label)}")
        if not self.label:
            raise ValueError("Label cannot be empty")
        if not isinstance(self.description, str):
            raise ValueError(
                f"Description must be a string, got {type(self.description)}"
            )
        if not self.description:
            raise ValueError("Description cannot be empty")
        if not isinstance(self.examples, list):
            raise ValueError("Examples must be a list")

    def get_rubric_item_score(self) -> int:
        """Get the rubric item score."""
        return self.score

    def get_rubric_item_label(self) -> str:
        """Get the rubric item label."""
        return self.label

    def get_rubric_item_description(self) -> str:
        """Get the rubric item description."""
        return self.description

    def get_rubric_item_examples(self) -> List[str]:
        """Get the rubric item examples."""
        return self.examples


@dataclass
class Criterion:
    """Represents a single evaluation criterion."""

    description: str
    weight: float
    is_deciding_factor: bool
    rubric: List[RubricItem]
    category: str
    deciding_threshold: Optional[int] = None
    is_rewrite_signal: Optional[bool] = None

    def __post_init__(self):
        """Validate criterion data."""
        if not isinstance(self.description, str):
            raise ValueError(
                f"Description must be a string, got {type(self.description)}"
            )
        if not self.description:
            raise ValueError("Description cannot be empty")
        if not isinstance(self.category, str):
            raise ValueError(f"Category must be a string, got {type(self.category)}")
        if not self.category:
            raise ValueError("Category cannot be empty")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")
        if not isinstance(self.is_deciding_factor, bool):
            raise ValueError(
                f"Is deciding factor must be a boolean, got {type(self.is_deciding_factor)}"
            )
        if not self.rubric:
            raise ValueError("Rubric cannot be empty")
        # Rubrics Validation
        # Starting from 0 or 1
        # Must be continuous
        scores = [item.score for item in self.rubric]
        scores.sort()
        if not scores:
            raise ValueError("Rubric cannot be empty")
        if scores[0] not in (0, 1):
            raise ValueError("Scores must start from 0 or 1")
        for i in range(1, len(scores)):
            if scores[i] != scores[i - 1] + 1:
                raise ValueError("Scores must be continuous")
        if len(scores) != len(set(scores)):
            raise ValueError("Scores must be unique")
        if self.is_deciding_factor and not self.deciding_threshold:
            raise ValueError("Deciding threshold is required for deciding factors")

    def get_criterion_description(self) -> str:
        """Get the criterion description."""
        return self.description

    def get_criterion_rubric(self) -> List[RubricItem]:
        """Get the criterion rubric."""
        return self.rubric


@dataclass
class Metadata:
    """Represents the metadata for the evaluation criteria."""

    version: str
    description: str
    total_criteria: int

    def __post_init__(self):
        """Validate metadata data."""
        if not self.version:
            raise ValueError("Version cannot be empty")
        if not self.description:
            raise ValueError("Description cannot be empty")
        if not isinstance(self.total_criteria, int):
            raise ValueError(
                f"Total criteria must be an integer, got {type(self.total_criteria)}"
            )
        if self.total_criteria <= 0:
            raise ValueError("Total criteria must be greater than 0")


@dataclass
class RewriteThresholds:
    """Represents thresholds for determining if content needs rewriting."""

    problem_clarity: int
    unit_test_validity_fp: int
    unit_test_validity_fn: int
    validity_gold_patch_alignment: int
    validity_problem_test_alignment: int

    def __post_init__(self):
        """Validate thresholds."""
        if not isinstance(self.problem_clarity, int):
            raise ValueError(
                f"Problem clarity threshold must be an integer, got {type(self.problem_clarity)}"
            )
        if not isinstance(self.unit_test_validity_fp, int):
            raise ValueError(
                f"Unit test validity fp threshold must be an integer, got {type(self.unit_test_validity_fp)}"
            )
        if not isinstance(self.unit_test_validity_fn, int):
            raise ValueError(
                f"Unit test validity fn threshold must be an integer, got {type(self.unit_test_validity_fn)}"
            )
        if not isinstance(self.validity_gold_patch_alignment, int):
            raise ValueError(
                f"Validity gold patch alignment threshold must be an integer, got {type(self.validity_gold_patch_alignment)}"
            )
        if not isinstance(self.validity_problem_test_alignment, int):
            raise ValueError(
                f"Validity problem test alignment threshold must be an integer, got {type(self.validity_problem_test_alignment)}"
            )


@dataclass
class CompatibilityThresholds:
    """Represents compatibility thresholds for overall assessment."""

    compatible: float
    needs_human_review: float
    incompatible: float

    def __post_init__(self):
        """Validate thresholds."""
        if not isinstance(self.compatible, float):
            raise ValueError(
                f"Compatible threshold must be a float, got {type(self.compatible)}"
            )
        if not isinstance(self.needs_human_review, float):
            raise ValueError(
                f"Needs human review threshold must be a float, got {type(self.needs_human_review)}"
            )
        if not isinstance(self.incompatible, float):
            raise ValueError(
                f"Incompatible threshold must be a float, got {type(self.incompatible)}"
            )


@dataclass
class OverallAssessment:
    """Represents overall assessment configuration."""

    description: str
    weighting_method: str
    compatibility_thresholds: CompatibilityThresholds
    rewrite_thresholds: RewriteThresholds

    def __post_init__(self):
        """Validate overall assessment."""
        if not self.description:
            raise ValueError("Description cannot be empty")
        if not isinstance(self.weighting_method, str):
            raise ValueError(
                f"Weighting method must be a string, got {type(self.weighting_method)}"
            )
        # if self.weighting_method not in ["weighted_average"]:
        #     raise ValueError("Weighting method must be 'weighted_average'")


@dataclass
class CriteriaConfig:
    """Main configuration class that holds all criteria data."""

    metadata: Metadata
    criteria: Dict[str, Criterion]
    overall_assessment: OverallAssessment

    def __post_init__(self):
        """Validate the complete configuration."""
        # Validate total criteria count matches actual criteria
        if len(self.criteria) != self.metadata.total_criteria:
            raise ValueError(
                f"Total criteria count ({self.metadata.total_criteria}) "
                f"doesn't match actual criteria count ({len(self.criteria)})"
            )

        # Validate weights sum to approximately 1.0 (allowing for small floating point errors)
        total_weight = sum(criterion.weight for criterion in self.criteria.values())
        if abs(total_weight - 1.0) > 1e-10:
            raise ValueError(f"Criteria weights must sum to 1.0, got {total_weight}")

    def list_criteria(self) -> List[str]:
        """List all criteria names and overall assessment."""
        criteria_list = list(self.criteria.keys())
        return criteria_list

    def get_criterion(self, criterion_name: str) -> Union[Criterion, OverallAssessment]:
        """Get a specific criterion or overall assessment by name."""
        if criterion_name == "overall_assessment":
            return self.overall_assessment
        if criterion_name not in self.criteria:
            raise ValueError(
                f"Criterion '{criterion_name}' not found. Available criteria: {list(self.criteria.keys())} and overall_assessment"
            )
        return self.criteria[criterion_name]


# Data classes for prompts templates
@dataclass
class PromptTemplate:
    """Represents a single prompt template with all its components."""

    system_instruction: str
    task_description: str
    context_format: str
    response_format: str
    disagreement_context_format: Optional[str] = None

    def __post_init__(self):
        """Validate prompt template data."""
        if not isinstance(self.system_instruction, str):
            raise ValueError(
                f"System instruction must be a string, got {type(self.system_instruction)}"
            )
        if not self.system_instruction.strip():
            raise ValueError("System instruction cannot be empty")

        if not isinstance(self.task_description, str):
            raise ValueError(
                f"Task description must be a string, got {type(self.task_description)}"
            )
        if not self.task_description.strip():
            raise ValueError("Task description cannot be empty")

        if not isinstance(self.context_format, str):
            raise ValueError(
                f"Context format must be a string, got {type(self.context_format)}"
            )
        if not self.context_format.strip():
            raise ValueError("Context format cannot be empty")

        if not isinstance(self.response_format, str):
            raise ValueError(
                f"Response format must be a string, got {type(self.response_format)}"
            )
        if not self.response_format.strip():
            raise ValueError("Response format cannot be empty")

    def format_context(self, **kwargs) -> str:
        """Format the context using provided keyword arguments."""
        try:
            return self.context_format.format(**kwargs)
        except KeyError as e:
            # Handle missing context variables gracefully by using default values
            missing_var = str(e).strip("'\"")

            # Apply default values for specific variables that are used in templates
            if (
                missing_var.startswith("repo_")
                or missing_var == "issue_status"
                or missing_var == "pr_has_linked_issues"
            ):
                logger.warning(
                    f"Missing context variable '{missing_var}', using default value"
                )
                kwargs[missing_var] = ""
                # Retry formatting with default value - this will handle multiple missing variables
                return self.format_context(**kwargs)
            else:
                # For other variables, raise the error
                raise ValueError(f"Missing required context variable: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting context: {e}")

    def get_system_instruction(self) -> str:
        """Get the system instruction."""
        return self.system_instruction

    def get_task_description(self) -> str:
        """Get the task description."""
        return self.task_description

    def get_response_format(self) -> str:
        """Get the response format."""
        return self.response_format


@dataclass
class PromptsTemplatesConfig:
    """Main configuration class that holds all prompt templates."""

    templates: Dict[str, PromptTemplate]

    def __post_init__(self):
        """Validate the complete configuration."""
        if not self.templates:
            raise ValueError("Templates dictionary cannot be empty")

        # Validate template names are valid
        for template_name in self.templates.keys():
            if not isinstance(template_name, str):
                raise ValueError(
                    f"Template name must be a string, got {type(template_name)}"
                )
            if not template_name.strip():
                raise ValueError("Template name cannot be empty")

    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific template by name."""
        if template_name not in self.templates:
            raise ValueError(
                f"Template '{template_name}' not found. Available templates: {list(self.templates.keys())}"
            )
        return self.templates[template_name]

    def list_templates(self) -> List[str]:
        """Get a list of all available template names."""
        return list(self.templates.keys())


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""

    provider: str
    api_key_env: str
    max_retries: int
    retry_delay: float
    timeout: int
    temperature: float
    max_tokens: int
    base_url: str = None  # Optional for local models
    litellm_model_name: str = None
    pricing: Optional[Dict[str, float]] = None  # Pricing per 1K tokens (input, output)

    def __post_init__(self):
        """Validate model configuration."""
        if not isinstance(self.provider, str):
            raise ValueError(f"Provider must be a string, got {type(self.provider)}")
        if not self.provider:
            raise ValueError("Provider cannot be empty")

        if not isinstance(self.api_key_env, str):
            raise ValueError(
                f"API key env must be a string, got {type(self.api_key_env)}"
            )

        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError(
                f"Max retries must be a non-negative integer, got {self.max_retries}"
            )

        if not isinstance(self.retry_delay, (int, float)) or self.retry_delay < 0:
            raise ValueError(
                f"Retry delay must be a non-negative number, got {self.retry_delay}"
            )

        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ValueError(f"Timeout must be a positive integer, got {self.timeout}")

        if not isinstance(self.temperature, (int, float)) or not (
            0.0 <= self.temperature <= 2.0
        ):
            raise ValueError(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError(
                f"Max tokens must be a positive integer, got {self.max_tokens}"
            )

        # For local providers, base_url is required
        if self.provider == "local" and not self.base_url:
            raise ValueError("base_url is required for local providers")

    def to_litellm_config(self) -> Dict[str, Any]:
        """Convert the model configuration to a dictionary."""
        return {
            "num_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def get_litellm_model_name(self) -> str:
        """Get the litellm model name."""
        if not self.litellm_model_name:
            logger.warning(
                f"Litellm model name is not set for model {self.provider}/{self.model_name}"
            )
            return f"{self.provider}/{self.model_name}"
        return self.litellm_model_name


@dataclass
class LlmConfig:
    """Configuration for LLM models and settings."""

    primary_model: str
    fallback_models: list[str]
    models: Dict[str, ModelConfig]

    def __post_init__(self):
        """Validate LLM configuration."""
        if not isinstance(self.primary_model, str):
            raise ValueError(
                f"Primary model must be a string, got {type(self.primary_model)}"
            )
        if not self.primary_model:
            raise ValueError("Primary model cannot be empty")

        if not isinstance(self.fallback_models, list):
            raise ValueError(
                f"Fallback models must be a list, got {type(self.fallback_models)}"
            )

        if not isinstance(self.models, dict):
            raise ValueError(f"Models must be a dictionary, got {type(self.models)}")

        # Validate that primary model exists in models
        if self.primary_model not in self.models:
            raise ValueError(
                f"Primary model '{self.primary_model}' not found in models configuration"
            )

        # Validate that all fallback models exist in models
        for fallback_model in self.fallback_models:
            if fallback_model not in self.models:
                raise ValueError(
                    f"Fallback model '{fallback_model}' not found in models configuration"
                )


@dataclass
class ProcessingConfig:
    """Configuration for processing and evaluation logic."""

    max_parallel_workers: int
    save_progress_interval: int
    max_valid_prs_per_repo: int
    early_exit_threshold: float

    def __post_init__(self):
        """Validate processing configuration."""
        if (
            not isinstance(self.max_parallel_workers, int)
            or self.max_parallel_workers <= 0
        ):
            raise ValueError(
                f"Max parallel workers must be a positive integer, got {self.max_parallel_workers}"
            )

        if (
            not isinstance(self.save_progress_interval, int)
            or self.save_progress_interval <= 0
        ):
            raise ValueError(
                f"Save progress interval must be a positive integer, got {self.save_progress_interval}"
            )

        if (
            not isinstance(self.max_valid_prs_per_repo, int)
            or self.max_valid_prs_per_repo <= 0
        ):
            raise ValueError(
                f"Max valid PRs per repo must be a positive integer, got {self.max_valid_prs_per_repo}"
            )

        if (
            not isinstance(self.early_exit_threshold, (int, float))
            or self.early_exit_threshold < 0
        ):
            raise ValueError(
                f"Early exit threshold must be a non-negative number, got {self.early_exit_threshold}"
            )


@dataclass
class StorageConfig:
    """Configuration for data storage and backup."""

    input_format: str
    output_format: str
    database_url_env: str = None
    backup_enabled: bool = True
    backup_interval: int = 100

    def __post_init__(self):
        """Validate storage configuration."""
        if not isinstance(self.input_format, str):
            raise ValueError(
                f"Input format must be a string, got {type(self.input_format)}"
            )
        if self.input_format not in ["csv", "parquet"]:
            raise ValueError(
                f"Input format must be 'csv' or 'parquet', got {self.input_format}"
            )

        if not isinstance(self.output_format, str):
            raise ValueError(
                f"Output format must be a string, got {type(self.output_format)}"
            )
        if self.output_format not in ["csv", "parquet"]:
            raise ValueError(
                f"Output format must be 'csv' or 'parquet', got {self.output_format}"
            )

        if self.database_url_env is not None and not isinstance(
            self.database_url_env, str
        ):
            raise ValueError(
                f"Database URL env must be a string or None, got {type(self.database_url_env)}"
            )

        if not isinstance(self.backup_enabled, bool):
            raise ValueError(
                f"Backup enabled must be a boolean, got {type(self.backup_enabled)}"
            )

        if not isinstance(self.backup_interval, int) or self.backup_interval <= 0:
            raise ValueError(
                f"Backup interval must be a positive integer, got {self.backup_interval}"
            )


@dataclass
class InputFilesConfig:
    """Configuration for input data files."""

    data_file: str

    def __post_init__(self):
        """Validate input files configuration."""
        if not isinstance(self.data_file, str):
            raise ValueError(f"Data file must be a string, got {type(self.data_file)}")
        if not self.data_file:
            raise ValueError("Data file cannot be empty")


@dataclass
class LlmEvaluatorConfig:
    """Main configuration class for the LLM evaluator."""

    llm: LlmConfig
    processing: ProcessingConfig
    storage: StorageConfig
    input_files: InputFilesConfig
    output_path: str
    prompts_dir: str
    cost_estimation: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate the complete configuration."""
        if not isinstance(self.output_path, str):
            raise ValueError(
                f"Output path must be a string, got {type(self.output_path)}"
            )
        if not self.output_path:
            raise ValueError("Output path cannot be empty")

        if not isinstance(self.prompts_dir, str):
            raise ValueError(
                f"Prompts directory must be a string, got {type(self.prompts_dir)}"
            )
        if not self.prompts_dir:
            raise ValueError("Prompts directory cannot be empty")

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get the model configuration."""
        return self.llm.models[model_name]

    def get_litellm_model_name(self, model_name: str) -> str:
        """Get the litellm model name."""
        return self.llm.models[model_name].get_litellm_model_name()

    def get_api_key(self, model_name: str) -> str:
        """Get the API key."""
        import os

        api_key_env = self.llm.models[model_name].api_key_env
        if not api_key_env:
            return ""
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        return api_key

    def get_litellm_config(self, model_name: str) -> Dict[str, Any]:
        """Get the litellm configuration."""
        return self.llm.models[model_name].to_litellm_config()


class InputLoader:
    def __init__(
        self,
        config_dir: Optional[str] = None,
        prompts_templates_config_file_path: Optional[str] = None,
        criteria_config_file_path: Optional[str] = None,
        llm_evaluator_config_file_path: Optional[str] = None,
    ):
        """
        Initialize the input loader.

        Args:
            config_dir: Directory containing configuration files. If provided, individual
                       file paths are ignored and files are loaded from this directory.
            prompts_templates_config_file_path: [DEPRECATED] Path to prompts templates config file
            criteria_config_file_path: [DEPRECATED] Path to criteria config file
            llm_evaluator_config_file_path: [DEPRECATED] Path to LLM evaluator config file
        """
        # Use config directory approach if provided (new preferred method)
        if config_dir is not None:
            self.config_dir = config_dir
            self.prompts_templates_config = self.load_prompts_config(
                os.path.join(config_dir, "prompts_templates.yaml")
            )
            self.criteria_config = self.load_criteria_config(
                os.path.join(config_dir, "criteria.yaml")
            )
            self.llm_evaluator_config = self.load_llm_evaluator_config(
                os.path.join(config_dir, "llm_caller.yaml")
            )
        else:
            # Maintain backward compatibility with individual file paths
            self.config_dir = "configs"  # Default directory
            if prompts_templates_config_file_path is not None:
                self.prompts_templates_config = self.load_prompts_config(
                    prompts_templates_config_file_path
                )
            else:
                self.prompts_templates_config = self.load_prompts_config()
            if criteria_config_file_path is not None:
                self.criteria_config = self.load_criteria_config(
                    criteria_config_file_path
                )
            else:
                self.criteria_config = self.load_criteria_config()

            if llm_evaluator_config_file_path is not None:
                self.llm_evaluator_config = self.load_llm_evaluator_config(
                    llm_evaluator_config_file_path
                )
            else:
                self.llm_evaluator_config = self.load_llm_evaluator_config()

        # Check if there is any criteria is not in the templates
        self.check_criteria_in_templates()

    def check_criteria_in_templates(self) -> None:
        """Check if there is any criteria is not in the templates"""
        all_criteria = self.criteria_config.list_criteria()
        all_templates = self.prompts_templates_config.list_templates()
        for criterion in all_criteria:
            if criterion not in all_templates:
                raise ValueError(f"Criterion '{criterion}' not found in the templates")

    def load_yaml_file(self, yaml_path: str) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file.

        :param yaml_path: Path to the YAML configuration file.
        :return: Dictionary containing the parsed YAML content.
        :raises FileNotFoundError: If the YAML file doesn't exist.
        :raises yaml.YAMLError: If the YAML file is malformed.
        """
        # Ensure YAML config exists
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

        # Load YAML content
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f)
            return cfg
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")

    def load_criteria_config(
        self, criteria_yaml_path: str = "configs/criteria.yaml"
    ) -> CriteriaConfig:
        """Load and validate criteria configuration."""
        cfg = self.load_yaml_file(criteria_yaml_path)

        # Parse metadata
        metadata = Metadata(**cfg["metadata"])

        # Parse criteria - convert the nested structure
        criteria = {}
        for criterion_name, criterion_data in cfg["criteria"].items():
            # Parse rubric items first
            rubric_items = []
            for rubric_data in criterion_data["rubric"]:
                rubric_items.append(RubricItem(**rubric_data))

            # Create criterion with parsed rubric
            criterion_data["rubric"] = rubric_items
            criteria[criterion_name] = Criterion(**criterion_data)

        # Parse overall assessment
        overall_data = cfg["overall_assessment"]

        # Parse compatibility thresholds
        compatibility_thresholds = CompatibilityThresholds(
            **overall_data["compatibility_thresholds"]
        )

        # Parse rewrite thresholds
        rewrite_thresholds = RewriteThresholds(**overall_data["rewrite_thresholds"])

        # Set both in overall_data
        overall_data["compatibility_thresholds"] = compatibility_thresholds
        overall_data["rewrite_thresholds"] = rewrite_thresholds
        overall_assessment = OverallAssessment(**overall_data)

        return CriteriaConfig(
            metadata=metadata, criteria=criteria, overall_assessment=overall_assessment
        )

    def load_prompts_config(
        self, prompts_yaml_path: str = "configs/prompts_templates.yaml"
    ) -> PromptsTemplatesConfig:
        """Load and validate prompts configuration."""
        cfg = self.load_yaml_file(prompts_yaml_path)

        # Get templates section
        templates_data = cfg.get("templates")

        # Parse each template
        templates = {}
        for template_name, template_data in templates_data.items():
            # Create PromptTemplate instance
            template = PromptTemplate(
                system_instruction=template_data["system_instruction"],
                task_description=template_data["task_description"],
                context_format=template_data["context_format"],
                response_format=template_data["response_format"],
                disagreement_context_format=template_data.get("disagreement_context_format"),
            )

            templates[template_name] = template

        return PromptsTemplatesConfig(templates=templates)

    def load_llm_evaluator_config(
        self, llm_evaluator_yaml_path: str = "configs/llm_evaluator.yaml"
    ) -> LlmEvaluatorConfig:
        """Load and validate LLM evaluator configuration."""
        cfg = self.load_yaml_file(llm_evaluator_yaml_path)

        # Parse LLM configuration
        llm_data = cfg.get("llm", {})
        if not llm_data:
            raise ValueError("'llm' section must be specified in YAML config")

        # Parse models
        models = {}
        for model_name, model_data in llm_data.get("models", {}).items():
            # Filter out fields that ModelConfig doesn't accept
            filtered_model_data = {
                k: v for k, v in model_data.items() if k not in ["rate_limit_rpm"]
            }
            models[model_name] = ModelConfig(**filtered_model_data)

        llm_config = LlmConfig(
            primary_model=llm_data["primary_model"],
            fallback_models=llm_data.get("fallback_models", []),
            models=models,
        )

        # Parse processing configuration
        processing_data = cfg.get("processing", {})
        # Add default value for early_exit_threshold if not present
        if "early_exit_threshold" not in processing_data:
            processing_data["early_exit_threshold"] = 0.0
        processing_config = ProcessingConfig(**processing_data)

        # Parse storage configuration
        storage_data = cfg.get("storage", {})
        storage_config = StorageConfig(**storage_data)

        # Parse input files configuration
        input_files_data = cfg.get("input_files", {})
        input_files_config = InputFilesConfig(**input_files_data)

        # Get top-level configurations
        output_path = cfg.get("output_path")
        if not output_path:
            raise ValueError("'output_path' must be specified in YAML config")

        prompts_dir = cfg.get("prompts_dir")
        if not prompts_dir:
            raise ValueError("'prompts_dir' must be specified in YAML config")

        # Try to load cost estimation configuration from llm_caller.yaml if it exists
        cost_estimation_config = None
        try:
            llm_caller_path = os.path.join(
                os.path.dirname(llm_evaluator_yaml_path), "llm_caller.yaml"
            )
            if os.path.exists(llm_caller_path):
                llm_caller_cfg = self.load_yaml_file(llm_caller_path)
                cost_estimation_config = llm_caller_cfg.get("cost_estimation")
                if cost_estimation_config:
                    logger.info(
                        "Loaded cost estimation configuration from llm_caller.yaml"
                    )
        except Exception as e:
            logger.warning(f"Failed to load cost estimation config: {e}")

        return LlmEvaluatorConfig(
            llm=llm_config,
            processing=processing_config,
            storage=storage_config,
            input_files=input_files_config,
            output_path=output_path,
            prompts_dir=prompts_dir,
            cost_estimation=cost_estimation_config,
        )

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get the model configuration."""
        return self.llm_evaluator_config.get_model_config(model_name)

    def get_litellm_model_name(self, model_name: str) -> str:
        """Get the litellm model name."""
        return self.llm_evaluator_config.get_litellm_model_name(model_name)

    def get_api_key(self, model_name: str) -> str:
        """Get the API key."""
        return self.llm_evaluator_config.get_api_key(model_name)

    def get_litellm_config(self, model_name: str) -> Dict[str, Any]:
        """Get the litellm configuration."""
        return self.llm_evaluator_config.get_litellm_config(model_name)

    def load_evaluator_config(self) -> LlmEvaluatorConfig:
        """Load the evaluator configuration."""
        return self.llm_evaluator_config
