"""Tests for pquantlib.testing.tolerance (EXACT / TIGHT / LOOSE / custom)."""

from __future__ import annotations

import math

import pytest

from pquantlib.testing import tolerance

# --- exact -------------------------------------------------------------------


def test_exact_passes_for_bit_identical_values() -> None:
    tolerance.exact(1.0, 1.0)
    tolerance.exact(-0.0, -0.0)
    tolerance.exact(math.inf, math.inf)


def test_exact_fails_on_fp_non_associativity() -> None:
    # 0.1 + 0.2 != 0.3 at bit level
    with pytest.raises(AssertionError, match="EXACT"):
        tolerance.exact(0.1 + 0.2, 0.3)


def test_exact_fails_for_positive_vs_negative_zero() -> None:
    with pytest.raises(AssertionError, match="EXACT"):
        tolerance.exact(0.0, -0.0)


def test_exact_includes_reason_when_supplied() -> None:
    with pytest.raises(AssertionError, match="custom-rationale"):
        tolerance.exact(1.0, 2.0, reason="custom-rationale")


# --- tight -------------------------------------------------------------------


def test_tight_passes_for_fp_non_associativity_within_1e_minus_12() -> None:
    tolerance.tight(0.1 + 0.2, 0.3)


def test_tight_fails_above_tight_threshold() -> None:
    with pytest.raises(AssertionError, match="TIGHT"):
        tolerance.tight(1.0, 1.0 + 1e-10)


# --- loose -------------------------------------------------------------------


def test_loose_passes_within_1e_minus_8() -> None:
    tolerance.loose(1.0, 1.0 + 1e-10)


def test_loose_fails_above_loose_threshold() -> None:
    with pytest.raises(AssertionError, match="LOOSE"):
        tolerance.loose(1.0, 1.0 + 1e-4)


# --- custom ------------------------------------------------------------------


def test_custom_passes_within_supplied_tolerance() -> None:
    tolerance.custom(1.0, 1.0 + 1e-5, abs_tol=1e-4, rel_tol=1e-4, reason="loose-by-design")


def test_custom_fails_above_supplied_tolerance() -> None:
    with pytest.raises(AssertionError, match="custom"):
        tolerance.custom(1.0, 1.0 + 1e-3, abs_tol=1e-4, rel_tol=1e-4, reason="needs-loose")


def test_custom_requires_reason_keyword_argument() -> None:
    # `reason` is keyword-only and required; calling without it is a TypeError.
    with pytest.raises(TypeError):
        tolerance.custom(  # type: ignore[call-arg]
            1.0, 1.0, abs_tol=1e-4, rel_tol=1e-4
        )
