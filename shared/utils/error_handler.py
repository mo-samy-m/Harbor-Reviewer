"""
Error handling module for the SR Code SWE system.
Provides centralized error management using configuration-based error messages
for all system components including repository fetching, PR/issue extraction,
LLM evaluation, and human annotation.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import yaml
from .log import setup_logger


@dataclass
class ErrorInfo:
    """Structured error information"""

    code: str
    message: str
    severity: str
    retryable: bool
    user_friendly: str
    timestamp: datetime = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class ErrorHandler:
    """Centralized error handler for the SR Code SWE system"""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the error handler

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self.logger = setup_logger(__name__)
        self.error_config = self._load_error_config()

    def _load_error_config(self) -> Dict[str, Any]:
        """Load error configuration from errors.yaml"""
        error_file = os.path.join(self.config_dir, "errors.yaml")

        if not os.path.exists(error_file):
            # Return default error config if file doesn't exist
            return self._get_default_error_config()

        try:
            with open(error_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:  # Empty file
                    self.logger.warning(f"Empty error config file: {error_file}")
                    return self._get_default_error_config()
                
                config = yaml.safe_load(content)
                if config is None:  # YAML parsed as None
                    self.logger.warning(f"Error config file parsed as None: {error_file}")
                    return self._get_default_error_config()
                
                return config
        except Exception as e:
            # Fallback to default config if loading fails
            self.logger.warning(f"Failed to load error config from {error_file}: {e}")
            return self._get_default_error_config()

    def _get_default_error_config(self) -> Dict[str, Any]:
        """Get default error configuration"""
        return {
            "errors": {
                # Model-related errors
                "model": {
                    "not_found": {
                        "code": "MODEL_NOT_FOUND",
                        "message": "Model configuration not found: {model_name}",
                        "severity": "error",
                        "retryable": False,
                        "user_friendly": "The requested model '{model_name}' is not available in the current configuration.",
                    },
                },
                # API-related errors
                "api": {
                    "timeout": {
                        "code": "API_TIMEOUT",
                        "message": "Request timed out after {timeout}s",
                        "severity": "warning",
                        "retryable": True,
                        "user_friendly": "The request took too long to complete. Please try again.",
                    },
                    "service_unavailable": {
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "Service temporarily unavailable for model: {model_name}",
                        "severity": "warning",
                        "retryable": True,
                        "user_friendly": "The service is temporarily unavailable. Please try again later.",
                    },
                },
                # Processing errors
                "processing": {
                    "unexpected_error": {
                        "code": "UNEXPECTED_ERROR",
                        "message": "Unexpected error occurred for model: {model_name}",
                        "severity": "error",
                        "retryable": False,
                        "user_friendly": "An unexpected error occurred. Please try again or contact support.",
                    },
                },
                # System errors (fallback)
                "system": {
                    "config_not_found": {
                        "code": "CONFIG_NOT_FOUND",
                        "message": "Configuration not found: {config_name}",
                        "severity": "error",
                        "retryable": False,
                        "user_friendly": "The requested configuration '{config_name}' is not available.",
                    },
                    "generic_error": {
                        "code": "GENERIC_ERROR",
                        "message": "An error occurred: {details}",
                        "severity": "error",
                        "retryable": False,
                        "user_friendly": "An unexpected error occurred.",
                    },
                },
            },
            "error_severity_levels": ["info", "warning", "error", "critical"],
        }

    def get_error_info(self, error_path: str, **kwargs) -> ErrorInfo:
        """
        Get error information for a specific error type

        Args:
            error_path: Dot-separated path to error (e.g., 'api.timeout', 'config.not_found')
            **kwargs: Parameters to format the error message

        Returns:
            ErrorInfo object with formatted error details
        """
        # Navigate to the error configuration
        error_config = self._get_error_config_by_path(error_path)

        if not error_config:
            # Return generic error if specific error not found
            return ErrorInfo(
                code="UNKNOWN_ERROR",
                message=f"Unknown error: {error_path}",
                severity="error",
                retryable=False,
                user_friendly="An unexpected error occurred.",
                details={"error_path": error_path, **kwargs},
            )

        # Get required fields with defaults for missing keys
        code = error_config.get("code", "UNKNOWN_ERROR")
        message = error_config.get("message", f"Error: {error_path}")
        severity = error_config.get("severity", "error")
        retryable = error_config.get("retryable", False)
        user_friendly = error_config.get("user_friendly", "An unexpected error occurred.")

        # Format the message with provided parameters
        try:
            formatted_message = message.format(**kwargs)
        except (KeyError, ValueError):
            # If formatting fails, use the raw message
            formatted_message = message

        try:
            formatted_user_friendly = user_friendly.format(**kwargs)
        except (KeyError, ValueError):
            # If formatting fails, use the raw user_friendly message
            formatted_user_friendly = user_friendly

        return ErrorInfo(
            code=code,
            message=formatted_message,
            severity=severity,
            retryable=retryable,
            user_friendly=formatted_user_friendly,
            details=kwargs,
        )

    def _get_error_config_by_path(self, error_path: str) -> Optional[Dict[str, Any]]:
        """Get error configuration by dot-separated path"""
        try:
            config = self.error_config["errors"]
            for key in error_path.split("."):
                config = config[key]
            return config
        except (KeyError, TypeError):
            return None

    def create_error_response(self, error_path: str, **kwargs) -> Dict[str, Any]:
        """
        Create a standardized error response

        Args:
            error_path: Dot-separated path to error
            **kwargs: Parameters to format the error message

        Returns:
            Dictionary with error response
        """
        error_info = self.get_error_info(error_path, **kwargs)

        return {
            "error": True,
            "code": error_info.code,
            "message": error_info.message,
            "user_friendly": error_info.user_friendly,
            "severity": error_info.severity,
            "retryable": error_info.retryable,
            "timestamp": error_info.timestamp.isoformat(),
            "details": error_info.details,
        }

    def is_retryable_error(self, error_path: str, **kwargs) -> bool:
        """Check if an error is retryable"""
        error_info = self.get_error_info(error_path, **kwargs)
        return error_info.retryable

    def get_error_severity(self, error_path: str, **kwargs) -> str:
        """Get error severity level"""
        error_info = self.get_error_info(error_path, **kwargs)
        return error_info.severity


# Global error handler instance
_error_handler = None


def get_error_handler(config_dir: str = "configs") -> ErrorHandler:
    """Get or create the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(config_dir)
    return _error_handler


def create_error(error_path: str, **kwargs) -> ErrorInfo:
    """Convenience function to create an error"""
    handler = get_error_handler()
    return handler.get_error_info(error_path, **kwargs)


def is_retryable(error_path: str, **kwargs) -> bool:
    """Convenience function to check if error is retryable"""
    handler = get_error_handler()
    return handler.is_retryable_error(error_path, **kwargs)
