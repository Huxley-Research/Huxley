"""Tests for verification layer."""

import pytest

from huxley.verification.validator import (
    OutputValidator,
    ValidationResult,
    ValidationRule,
    Severity,
    NotEmptyRule,
    LengthRule,
    RegexRule,
    LambdaRule,
)
from huxley.core.exceptions import OutputVerificationError


class TestValidationRules:
    def test_not_empty_rule_valid(self):
        rule = NotEmptyRule()
        result = rule.validate("hello", {})
        assert result.valid is True

    def test_not_empty_rule_none(self):
        rule = NotEmptyRule()
        result = rule.validate(None, {})
        assert result.valid is False

    def test_not_empty_rule_empty_string(self):
        rule = NotEmptyRule()
        result = rule.validate("", {})
        assert result.valid is False

    def test_length_rule_valid(self):
        rule = LengthRule(min_length=1, max_length=10)
        result = rule.validate("hello", {})
        assert result.valid is True

    def test_length_rule_too_short(self):
        rule = LengthRule(min_length=10)
        result = rule.validate("hi", {})
        assert result.valid is False

    def test_length_rule_too_long(self):
        rule = LengthRule(max_length=5)
        result = rule.validate("hello world", {})
        assert result.valid is False

    def test_regex_rule_match(self):
        rule = RegexRule("email", r"@.*\.com$")
        result = rule.validate("test@example.com", {})
        assert result.valid is True

    def test_regex_rule_no_match(self):
        rule = RegexRule("email", r"@.*\.com$")
        result = rule.validate("invalid-email", {})
        assert result.valid is False

    def test_lambda_rule_valid(self):
        rule = LambdaRule("positive", lambda x: x > 0)
        result = rule.validate(5, {})
        assert result.valid is True

    def test_lambda_rule_invalid(self):
        rule = LambdaRule("positive", lambda x: x > 0, message="Must be positive")
        result = rule.validate(-1, {})
        assert result.valid is False
        assert result.failures[0].message == "Must be positive"


class TestOutputValidator:
    def test_empty_validator(self):
        validator = OutputValidator()
        result = validator.validate("anything")
        assert result.valid is True

    def test_multiple_rules_all_pass(self):
        validator = OutputValidator(rules=[
            NotEmptyRule(),
            LengthRule(min_length=1, max_length=100),
        ])
        result = validator.validate("hello")
        assert result.valid is True
        assert len(result.failures) == 0

    def test_multiple_rules_one_fails(self):
        validator = OutputValidator(rules=[
            NotEmptyRule(),
            LengthRule(min_length=10),
        ])
        result = validator.validate("hi")
        assert result.valid is False
        assert len(result.failures) == 1

    def test_fail_fast(self):
        validator = OutputValidator(
            rules=[
                NotEmptyRule(),
                LengthRule(min_length=100),
                LengthRule(max_length=5),
            ],
            fail_fast=True,
        )
        result = validator.validate("hello")
        # Should stop after first failure (length min)
        assert result.valid is False
        assert len(result.failures) == 1

    def test_raise_on_error(self):
        validator = OutputValidator(rules=[NotEmptyRule()])
        with pytest.raises(OutputVerificationError):
            validator.validate(None, raise_on_error=True)

    def test_add_rule_chaining(self):
        validator = (
            OutputValidator()
            .add_rule(NotEmptyRule())
            .add_rule(LengthRule(min_length=1))
        )
        result = validator.validate("test")
        assert result.valid is True

    def test_severity_levels(self):
        validator = OutputValidator(rules=[
            LambdaRule("error_rule", lambda x: False, severity=Severity.ERROR),
            LambdaRule("warning_rule", lambda x: False, severity=Severity.WARNING),
        ])
        result = validator.validate("test")
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


class TestValidationResult:
    def test_merge(self):
        r1 = ValidationResult(valid=True)
        r2 = ValidationResult(valid=False, failures=[
            ValidationResult.__class__  # placeholder
        ])

        # Test that merging preserves validity
        r3 = ValidationResult(valid=True)
        r4 = ValidationResult(valid=True)
        merged = r3.merge(r4)
        assert merged.valid is True

    def test_merge_invalid(self):
        from huxley.verification.validator import ValidationFailure

        r1 = ValidationResult(valid=True)
        r2 = ValidationResult(
            valid=False,
            failures=[
                ValidationFailure("test", "Failed", Severity.ERROR)
            ],
        )
        merged = r1.merge(r2)
        assert merged.valid is False
        assert len(merged.failures) == 1
