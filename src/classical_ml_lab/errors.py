"""Domain-specific errors with stable public error codes."""


class LabError(Exception):
    """Base class for expected, user-actionable failures."""

    code = "lab_error"


class InputValidationError(LabError):
    """Raised before training when CLI input or dataset validation fails."""

    code = "input_validation_error"


class MetricValidationError(LabError):
    """Raised when a metric cannot be computed safely."""

    code = "metric_validation_error"
