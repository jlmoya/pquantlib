"""Cross-validate FlatHazardRate against C++.

Probe source: migration-harness/cpp/probes/cluster_l8b/probe.cpp
Reference:    migration-harness/references/cluster/l8b.json (key: "flat_hazard_rate")
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l8b")["flat_hazard_rate"]


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


def test_survival_probability_at_sample_times(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    fhr = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    # TIGHT: closed-form exp(-lambda*t) matches up to 1 ulp on libm.
    tolerance.tight(fhr.survival_probability(0.5), cpp_ref["survival_t05"])
    tolerance.tight(fhr.survival_probability(1.0), cpp_ref["survival_t1"])
    tolerance.tight(fhr.survival_probability(2.0), cpp_ref["survival_t2"])
    tolerance.tight(fhr.survival_probability(5.0), cpp_ref["survival_t5"])


def test_default_probability_at_sample_times(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    fhr = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    tolerance.tight(fhr.default_probability(0.5), cpp_ref["default_t05"])
    tolerance.tight(fhr.default_probability(5.0), cpp_ref["default_t5"])


def test_hazard_rate_is_constant(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    fhr = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    # EXACT: constant rate stored verbatim; struct.pack bit-identical.
    tolerance.exact(fhr.hazard_rate(5.0), cpp_ref["hazard_t5"])


def test_default_density_at_sample(cpp_ref: dict[str, Any], ref_date: Date) -> None:
    fhr = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    # TIGHT: p(t) = lambda * exp(-lambda*t) is also closed-form.
    tolerance.tight(fhr.default_density(5.0), cpp_ref["default_density_t5"])


def test_quote_based_constructor(ref_date: Date) -> None:
    q = SimpleQuote(0.02)
    fhr = FlatHazardRate(ref_date, q, Actual365Fixed())
    s0 = fhr.survival_probability(5.0)
    # Mutating quote invalidates upstream observers — but FlatHazardRate
    # reads the quote value live, so the next call sees the new rate.
    q.set_value(0.05)
    s1 = fhr.survival_probability(5.0)
    assert s0 != s1


def test_two_arg_default_probability(ref_date: Date) -> None:
    """default_probability(t1, t2) returns S(t1) - S(t2)."""
    fhr = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    # Should equal S(2) - S(5).
    expected = fhr.survival_probability(2.0) - fhr.survival_probability(5.0)
    tolerance.tight(fhr.default_probability(2.0, 5.0), expected)


def test_max_date_is_open_ended(ref_date: Date) -> None:
    fhr = FlatHazardRate.from_rate(ref_date, 0.02, Actual365Fixed())
    assert fhr.max_date() == Date.max_date()
