"""Domain-specific exceptions for the validation subsystem."""

from __future__ import annotations


class DataValidationError(Exception):
    """Base exception for validation pipeline failures."""


class ValidationConfigurationError(DataValidationError):
    """Raised when validation configuration is missing or invalid."""


class ValidationInputError(DataValidationError):
    """Raised when validation input data cannot be loaded safely."""


class ValidationArtifactError(DataValidationError):
    """Raised when validation artifacts cannot be written or read."""
