import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from jinja2 import Template

# Add project root to path for shared module imports
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_cwd = Path.cwd().resolve()
if str(_cwd) not in sys.path and _cwd != _project_root:
    sys.path.insert(0, str(_cwd))

from .input_loader import (
    InputLoader,
    PromptTemplate,
    RubricItem,
)
from .template_renderer import TemplateRenderer
from shared.constants.prompt_template import USER_PROMPT_TEMPLATE
from shared.utils.log import setup_logger

# Setup logger
logger = setup_logger(__name__)


# Message class
@dataclass
class Message:
    """Represents a single message in a conversation with an LLM."""

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert Message to the dict format expected by LLM APIs."""
        return {"role": self.role, "content": self.content}


# Prompt class
@dataclass
class Prompt:
    """Represents a complete prompt with system and user messages."""

    system_message: Message
    user_message: Message

    def add_system(self, content: str) -> "Prompt":
        """Add a system instruction."""
        self.system_message = Message(role="system", content=content)
        return self

    def add_user(self, content: str) -> "Prompt":
        """Add a user prompt."""
        self.user_message = Message(role="user", content=content)
        return self

    def get_system_message(self) -> Message:
        """Get the system message."""
        return self.system_message

    def get_user_message(self) -> Message:
        """Get the user message."""
        return self.user_message

    def __getitem__(self, key: str) -> str:
        """Allow dict-style access to messages."""
        if key == "system":
            return self.system_message.content
        elif key == "user":
            return self.user_message.content
        raise KeyError(f"Invalid message type: {key}. Must be 'system' or 'user'")

    def to_llm_format(self) -> List[Dict[str, str]]:
        """
        Return the list of messages as dicts for LLM consumption.

        Returns:
            List of message dictionaries in LLM API format.
        """
        return [self.system_message.to_dict(), self.user_message.to_dict()]


class PromptBuilder:
    """Builds prompts from templates using configuration."""

    def __init__(
        self,
        config_dir: str = "configs",
    ):
        """
        Initialize the prompt builder.

        Args:
            config_dir: Directory containing configuration files.
        """
        logger.info("Loading prompts templates configuration")
        self.input_loader = InputLoader(config_dir=config_dir)
        self.template_renderer = TemplateRenderer()
        self.prompts_templates_config = self.input_loader.prompts_templates_config
        self.criteria_config = self.input_loader.criteria_config

    def render_prompt_template(
        self,
        *,
        task_description: str,
        criterion_description: str,
        context: str,
        response_format: str,
        rubric: List[RubricItem],
        disagreement_context: Optional[str] = None,
        template: str = USER_PROMPT_TEMPLATE,
    ) -> str:
        """
        Render the user message using a Jinja2 template.

        Args:
            task_description: Description of the task to perform.
            context: Formatted context information (already formatted from config).
            response_format: Expected response format block.
            rubric: List of rubric items.
            disagreement_context: Optional disagreement context to append.
            template: Jinja2 template to use for rendering.
        """
        t = Template(template)
        return t.render(
            task_description=task_description,
            criterion_description=criterion_description,
            context=context,
            response_format=response_format,
            rubric=rubric,
            disagreement_context=disagreement_context,
        )

    def get_templates_in_criteria(self) -> List[str]:
        """Get all templates that are in the criteria"""
        # Get all templates and criteria
        all_templates = self.prompts_templates_config.list_templates()
        all_criteria = self.criteria_config.list_criteria()
        return [template for template in all_templates if template in all_criteria]

    def format_context(
        self,
        criterion_name: str,
        prompt_template: PromptTemplate,
        context: Dict[str, Any],
    ) -> str:
        """Format the context for a prompt template using TemplateRenderer."""
        try:
            return self.template_renderer.format_context(
                prompt_template.context_format, **context
            )
        except KeyError as e:
            logger.error(
                f"Missing required context variable in criterion {criterion_name}: {e}"
            )
            raise ValueError(f"Missing required context variable: {e}")
        except Exception as e:
            logger.error(f"Error formatting context in criterion {criterion_name}: {e}")
            raise ValueError(f"Error formatting context: {e}")

    def build_prompts(self, **kwargs) -> List[Prompt]:
        """
        Build all prompts from templates.

        Args:
            **kwargs: Context variables for template rendering.

        Returns:
            List of Prompt objects.
        """
        prompts: List[Prompt] = []
        only_templates_in_criteria = self.get_templates_in_criteria()

        for template_name in only_templates_in_criteria:
            # Template info from prompts_templates_config
            prompt_template = self.prompts_templates_config.get_template(template_name)
            system_instruction = prompt_template.get_system_instruction()
            task_description = prompt_template.get_task_description()
            context_formatted = self.format_context(
                template_name, prompt_template, kwargs
            )
            response_format = prompt_template.get_response_format()

            # Template info from criteria_config
            criterion = self.criteria_config.get_criterion(template_name)
            criterion_description = getattr(
                criterion, "get_criterion_description", lambda: None
            )()
            criterion_rubric = getattr(
                criterion, "get_criterion_rubric", lambda: None
            )()

            # Build user message content using the constant template
            user_content = self.render_prompt_template(
                criterion_description=criterion_description,
                task_description=task_description,
                context=context_formatted,
                response_format=response_format,
                rubric=criterion_rubric,
            )

            # Create Prompt with system and user messages
            prompt = Prompt(
                system_message=Message(role="system", content=system_instruction),
                user_message=Message(role="user", content=user_content),
            )
            prompts.append(prompt)

        logger.info("Built %d prompts", len(prompts))
        return prompts

    def build_prompt(self, criterion_name: str, context: Dict[str, Any], disagreement_opinions: Optional[Dict[str, Any]] = None) -> Prompt:
        """
        Build a single prompt from a template.

        Args:
            criterion_name: Name of the criterion template to use.
            context: Dictionary of context variables for template rendering.
                     Supports both prefixed (input_*) and non-prefixed field names.
            disagreement_opinions: Optional dictionary with opinion_1_score, opinion_1_justification,
                                  opinion_2_score, opinion_2_justification to append to prompt.

        Returns:
            List of message dictionaries in LLM API format.

        Raises:
            ValueError: If criterion_name is not found in templates.
        """
        logger.info(f"Building prompt for criterion {criterion_name}")

        # Normalize context: strip "input_" prefix if present to support both formats
        normalized_context = {}
        prefixed_keys_found = []
        for key, value in context.items():
            if key.startswith("input_"):
                # Strip "input_" prefix for template rendering
                normalized_key = key[6:]  # Remove "input_" (6 characters)
                prefixed_keys_found.append((key, normalized_key))
                # Only use normalized key if non-prefixed version doesn't already exist
                if normalized_key not in context:
                    normalized_context[normalized_key] = value
                    logger.debug(
                        f"Stripped 'input_' prefix: '{key}' -> '{normalized_key}' for criterion {criterion_name}"
                    )
                else:
                    # Prefer non-prefixed version if both exist
                    normalized_context[normalized_key] = context[normalized_key]
                    logger.debug(
                        f"Found both '{key}' and '{normalized_key}', using non-prefixed version for criterion {criterion_name}"
                    )
            else:
                # Keep non-prefixed keys as-is
                normalized_context[key] = value
        
        if prefixed_keys_found:
            logger.info(
                f"Normalized {len(prefixed_keys_found)} prefixed field(s) for criterion {criterion_name}: "
                f"{', '.join([f'{old}->{new}' for old, new in prefixed_keys_found])}"
            )

        # Template info from prompt templates config
        prompt_template = self.prompts_templates_config.get_template(criterion_name)
        system_instruction = prompt_template.get_system_instruction()
        task_description = prompt_template.get_task_description()
        context_formatted = self.format_context(
            criterion_name, prompt_template, normalized_context
        )
        response_format = prompt_template.get_response_format()

        # Template info from criteria config
        criterion = self.criteria_config.get_criterion(criterion_name)
        criterion_description = getattr(
            criterion, "get_criterion_description", lambda: None
        )()
        criterion_rubric = getattr(criterion, "get_criterion_rubric", lambda: None)()

        # Format disagreement context if available
        disagreement_context = None
        if disagreement_opinions and prompt_template.disagreement_context_format:
            try:
                opinion_1_score = disagreement_opinions.get("opinion_1_score", "N/A")
                opinion_1_justification = disagreement_opinions.get("opinion_1_justification", "")
                opinion_2_score = disagreement_opinions.get("opinion_2_score", "N/A")
                opinion_2_justification = disagreement_opinions.get("opinion_2_justification", "")
                
                disagreement_context = prompt_template.disagreement_context_format.format(
                    opinion_1_score=opinion_1_score,
                    opinion_1_justification=opinion_1_justification,
                    opinion_2_score=opinion_2_score,
                    opinion_2_justification=opinion_2_justification,
                )
                logger.debug(f"Formatted disagreement context for criterion {criterion_name}")
            except Exception as e:
                logger.warning(f"Failed to format disagreement context for {criterion_name}: {e}")

        # Build user message content using the constant template
        user_content = self.render_prompt_template(
            criterion_description=criterion_description,
            task_description=task_description,
            context=context_formatted,
            response_format=response_format,
            rubric=criterion_rubric,
            disagreement_context=disagreement_context,
        )

        # Create Prompt with system and user messages
        prompt = Prompt(
            system_message=Message(role="system", content=system_instruction),
            user_message=Message(role="user", content=user_content),
        )
        logger.info(f"Built prompt for criterion {criterion_name}")

        return prompt
