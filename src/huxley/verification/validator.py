"""
Output verification and validation.

Provides a framework for validating agent outputs against
configurable rules and constraints. This is critical for
scientific applications where output correctness is essential.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from huxley.core.exceptions import OutputVerificationError
from huxley.core.logging import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """Severity level for validation failures."""

    ERROR = "error"  # Must fix, blocks output
    WARNING = "warning"  # Should fix, but can proceed
    INFO = "info"  # Informational only


@dataclass
class ValidationFailure:
    """A single validation failure."""

    rule_name: str
    message: str
    severity: Severity
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule": self.rule_name,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of running validation on an output."""

    valid: bool
    failures: list[ValidationFailure] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationFailure]:
        """Get only error-level failures."""
        return [f for f in self.failures if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationFailure]:
        """Get only warning-level failures."""
        return [f for f in self.failures if f.severity == Severity.WARNING]

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge two validation results."""
        return ValidationResult(
            valid=self.valid and other.valid,
            failures=self.failures + other.failures,
            metadata={**self.metadata, **other.metadata},
        )


class ValidationRule(ABC):
    """
    Abstract base class for validation rules.

    Implement custom rules by subclassing and implementing
    the validate method.
    """

    def __init__(
        self,
        name: str,
        severity: Severity = Severity.ERROR,
    ) -> None:
        self.name = name
        self.severity = severity

    @abstractmethod
    def validate(self, output: Any, context: dict[str, Any]) -> ValidationResult:
        """
        Validate an output.

        Args:
            output: The output to validate
            context: Additional context (e.g., input, execution info)

        Returns:
            ValidationResult indicating pass/fail
        """
        ...


class LambdaRule(ValidationRule):
    """
    Validation rule defined by a lambda/function.

    Convenient for simple validations.
    """

    def __init__(
        self,
        name: str,
        check: Callable[[Any], bool],
        message: str = "Validation failed",
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(name, severity)
        self._check = check
        self._message = message

    def validate(self, output: Any, context: dict[str, Any]) -> ValidationResult:
        try:
            if self._check(output):
                return ValidationResult(valid=True)
            else:
                return ValidationResult(
                    valid=False,
                    failures=[
                        ValidationFailure(
                            rule_name=self.name,
                            message=self._message,
                            severity=self.severity,
                        )
                    ],
                )
        except Exception as e:
            return ValidationResult(
                valid=False,
                failures=[
                    ValidationFailure(
                        rule_name=self.name,
                        message=f"Rule raised exception: {e}",
                        severity=self.severity,
                        details={"exception": str(e)},
                    )
                ],
            )


class NotEmptyRule(ValidationRule):
    """Validates that output is not empty."""

    def __init__(self, severity: Severity = Severity.ERROR) -> None:
        super().__init__("not_empty", severity)

    def validate(self, output: Any, context: dict[str, Any]) -> ValidationResult:
        if output is None:
            return ValidationResult(
                valid=False,
                failures=[
                    ValidationFailure(
                        rule_name=self.name,
                        message="Output is None",
                        severity=self.severity,
                    )
                ],
            )

        if isinstance(output, str) and not output.strip():
            return ValidationResult(
                valid=False,
                failures=[
                    ValidationFailure(
                        rule_name=self.name,
                        message="Output is empty string",
                        severity=self.severity,
                    )
                ],
            )

        if isinstance(output, (list, dict)) and len(output) == 0:
            return ValidationResult(
                valid=False,
                failures=[
                    ValidationFailure(
                        rule_name=self.name,
                        message="Output is empty collection",
                        severity=self.severity,
                    )
                ],
            )

        return ValidationResult(valid=True)


class RegexRule(ValidationRule):
    """Validates that string output matches a regex pattern."""

    def __init__(
        self,
        name: str,
        pattern: str,
        message: str | None = None,
        severity: Severity = Severity.ERROR,
        must_match: bool = True,
    ) -> None:
        super().__init__(name, severity)
        self._pattern = re.compile(pattern)
        self._message = message or f"Output must {'match' if must_match else 'not match'} pattern"
        self._must_match = must_match

    def validate(self, output: Any, context: dict[str, Any]) -> ValidationResult:
        if not isinstance(output, str):
            return ValidationResult(
                valid=False,
                failures=[
                    ValidationFailure(
                        rule_name=self.name,
                        message="Output is not a string",
                        severity=self.severity,
                    )
                ],
            )

        matches = bool(self._pattern.search(output))
        valid = matches if self._must_match else not matches

        if valid:
            return ValidationResult(valid=True)

        return ValidationResult(
            valid=False,
            failures=[
                ValidationFailure(
                    rule_name=self.name,
                    message=self._message,
                    severity=self.severity,
                    details={"pattern": self._pattern.pattern},
                )
            ],
        )


class JSONSchemaRule(ValidationRule):
    """Validates that output conforms to a JSON schema."""

    def __init__(
        self,
        name: str,
        schema: dict[str, Any],
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(name, severity)
        self._schema = schema

    def validate(self, output: Any, context: dict[str, Any]) -> ValidationResult:
        try:
            import jsonschema

            jsonschema.validate(output, self._schema)
            return ValidationResult(valid=True)

        except ImportError:
            logger.warning("jsonschema not installed, skipping schema validation")
            return ValidationResult(valid=True)

        except jsonschema.ValidationError as e:
            return ValidationResult(
                valid=False,
                failures=[
                    ValidationFailure(
                        rule_name=self.name,
                        message=str(e.message),
                        severity=self.severity,
                        details={"path": list(e.path)},
                    )
                ],
            )


class LengthRule(ValidationRule):
    """Validates string/collection length is within bounds."""

    def __init__(
        self,
        name: str = "length",
        min_length: int | None = None,
        max_length: int | None = None,
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(name, severity)
        self._min = min_length
        self._max = max_length

    def validate(self, output: Any, context: dict[str, Any]) -> ValidationResult:
        try:
            length = len(output)
        except TypeError:
            return ValidationResult(
                valid=False,
                failures=[
                    ValidationFailure(
                        rule_name=self.name,
                        message="Output has no length",
                        severity=self.severity,
                    )
                ],
            )

        failures = []

        if self._min is not None and length < self._min:
            failures.append(
                ValidationFailure(
                    rule_name=self.name,
                    message=f"Length {length} is below minimum {self._min}",
                    severity=self.severity,
                )
            )

        if self._max is not None and length > self._max:
            failures.append(
                ValidationFailure(
                    rule_name=self.name,
                    message=f"Length {length} exceeds maximum {self._max}",
                    severity=self.severity,
                )
            )

        return ValidationResult(valid=len(failures) == 0, failures=failures)


class OutputValidator:
    """
    Orchestrates validation of agent outputs.

    Runs a configurable set of validation rules and
    aggregates results.
    """

    def __init__(
        self,
        rules: list[ValidationRule] | None = None,
        *,
        fail_fast: bool = False,
    ) -> None:
        """
        Initialize the validator.

        Args:
            rules: Validation rules to apply
            fail_fast: Stop on first error (default: run all rules)
        """
        self._rules = rules or []
        self._fail_fast = fail_fast

    def add_rule(self, rule: ValidationRule) -> "OutputValidator":
        """
        Add a validation rule.

        Args:
            rule: Rule to add

        Returns:
            Self for chaining
        """
        self._rules.append(rule)
        return self

    def validate(
        self,
        output: Any,
        *,
        context: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """
        Validate an output against all rules.

        Args:
            output: Output to validate
            context: Additional validation context
            raise_on_error: Raise exception on validation failure

        Returns:
            Aggregated ValidationResult

        Raises:
            OutputVerificationError: If raise_on_error and validation fails
        """
        context = context or {}
        result = ValidationResult(valid=True)

        for rule in self._rules:
            try:
                rule_result = rule.validate(output, context)
                result = result.merge(rule_result)

                if self._fail_fast and not rule_result.valid:
                    break

            except Exception as e:
                logger.error(
                    "validation_rule_error",
                    rule=rule.name,
                    error=str(e),
                )
                result = result.merge(
                    ValidationResult(
                        valid=False,
                        failures=[
                            ValidationFailure(
                                rule_name=rule.name,
                                message=f"Rule raised exception: {e}",
                                severity=Severity.ERROR,
                            )
                        ],
                    )
                )

        if raise_on_error and not result.valid:
            raise OutputVerificationError(
                f"Validation failed with {len(result.errors)} error(s)",
                failures=[f.to_dict() for f in result.failures],
            )

        logger.debug(
            "validation_completed",
            valid=result.valid,
            error_count=len(result.errors),
            warning_count=len(result.warnings),
        )

        return result


# Factory functions for common validators
def create_basic_validator() -> OutputValidator:
    """Create a validator with basic rules."""
    return OutputValidator(rules=[NotEmptyRule()])


def create_json_validator(schema: dict[str, Any]) -> OutputValidator:
    """Create a validator for JSON output with schema."""
    return OutputValidator(
        rules=[
            NotEmptyRule(),
            JSONSchemaRule("schema", schema),
        ]
    )
