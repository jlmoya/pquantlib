"""Cross-validate InterpolatedSurvivalProbabilityCurve, InterpolatedHazardRateCurve,
InterpolatedDefaultDensityCurve against C++.

Probe source: migration-harness/cpp/probes/cluster_l8b/probe.cpp
Reference:    migration-harness/references/cluster/l8b.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.credit.interpolated_default_density_curve import (
    InterpolatedDefaultDensityCurve,
)
from pquantlib.termstructures.credit.interpolated_hazard_rate_curve import (
    InterpolatedHazardRateCurve,
)
from pquantlib.termstructures.credit.interpolated_survival_probability_curve import (
    InterpolatedSurvivalProbabilityCurve,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l8b")


@pytest.fixture(scope="module")
def ref_date() -> Date:
    return Date.from_ymd(15, Month.June, 2026)


@pytest.fixture(scope="module")
def four_pillar_dates(ref_date: Date) -> list[Date]:
    return [
        ref_date,
        ref_date + Period(1, TimeUnit.Years),
        ref_date + Period(2, TimeUnit.Years),
        ref_date + Period(5, TimeUnit.Years),
    ]


# --- InterpolatedSurvivalProbabilityCurve (LogLinear) -----------------------


def test_sp_curve_at_node_dates(
    cpp_ref: dict[str, Any], four_pillar_dates: list[Date],
) -> None:
    probs = [1.0, 0.98, 0.95, 0.85]
    curve = InterpolatedSurvivalProbabilityCurve(four_pillar_dates, probs, Actual365Fixed())
    ref = cpp_ref["interpolated_survival_probability"]
    # TIGHT: at known nodes the interpolator returns the exact node value
    # (modulo rounding through log/exp); within TIGHT.
    tolerance.tight(curve.survival_probability(four_pillar_dates[1]), ref["sp_at_node1y"])
    tolerance.tight(curve.survival_probability(four_pillar_dates[2]), ref["sp_at_node2y"])
    tolerance.tight(curve.survival_probability(four_pillar_dates[3]), ref["sp_at_node5y"])


def test_sp_curve_at_midpoint(
    cpp_ref: dict[str, Any], ref_date: Date, four_pillar_dates: list[Date],
) -> None:
    probs = [1.0, 0.98, 0.95, 0.85]
    curve = InterpolatedSurvivalProbabilityCurve(four_pillar_dates, probs, Actual365Fixed())
    # Mid-point between node 0 and node 1 in time.
    dc = Actual365Fixed()
    t_mid = dc.year_fraction(ref_date, four_pillar_dates[1]) / 2.0
    ref = cpp_ref["interpolated_survival_probability"]
    tolerance.tight(curve.survival_probability(t_mid), ref["sp_mid_0_1"])


def test_sp_curve_inspectors(four_pillar_dates: list[Date]) -> None:
    probs = [1.0, 0.98, 0.95, 0.85]
    curve = InterpolatedSurvivalProbabilityCurve(four_pillar_dates, probs, Actual365Fixed())
    assert curve.dates() == four_pillar_dates
    assert curve.survival_probabilities() == probs
    assert curve.max_date() == four_pillar_dates[-1]
    assert len(curve.times()) == 4
    assert curve.times()[0] == 0.0


def test_sp_curve_first_must_be_one(four_pillar_dates: list[Date]) -> None:
    probs = [0.99, 0.98, 0.95, 0.85]
    with pytest.raises(LibraryException, match="first probability"):
        InterpolatedSurvivalProbabilityCurve(four_pillar_dates, probs, Actual365Fixed())


def test_sp_curve_monotonic(four_pillar_dates: list[Date]) -> None:
    probs = [1.0, 0.95, 0.98, 0.85]  # 0.98 > 0.95 → negative hazard
    with pytest.raises(LibraryException, match="negative hazard rate"):
        InterpolatedSurvivalProbabilityCurve(four_pillar_dates, probs, Actual365Fixed())


# --- InterpolatedHazardRateCurve (BackwardFlat) -----------------------------


def test_hr_curve_hazard_at_sample_times(
    cpp_ref: dict[str, Any], four_pillar_dates: list[Date],
) -> None:
    hazards = [0.01, 0.02, 0.025, 0.03]
    curve = InterpolatedHazardRateCurve(four_pillar_dates, hazards, Actual365Fixed())
    ref = cpp_ref["interpolated_hazard_rate"]
    # BackwardFlat: hazard between (t_{i-1}, t_i] is hazards[i].
    tolerance.tight(curve.hazard_rate(0.5), ref["hazard_t05"])
    tolerance.tight(curve.hazard_rate(1.5), ref["hazard_t15"])
    tolerance.tight(curve.hazard_rate(3.0), ref["hazard_t3"])


def test_hr_curve_survival_via_primitive(
    cpp_ref: dict[str, Any], four_pillar_dates: list[Date],
) -> None:
    hazards = [0.01, 0.02, 0.025, 0.03]
    curve = InterpolatedHazardRateCurve(four_pillar_dates, hazards, Actual365Fixed())
    ref = cpp_ref["interpolated_hazard_rate"]
    # TIGHT: S(t) = exp(- primitive(t)), and the primitive is a closed-form
    # sum of dx * hazards[i], so reproducible bit-for-bit.
    tolerance.tight(curve.survival_probability(1.0), ref["sp_t1"])
    tolerance.tight(curve.survival_probability(2.0), ref["sp_t2"])
    tolerance.tight(curve.survival_probability(5.0), ref["sp_t5"])


def test_hr_curve_extrapolates_flat(four_pillar_dates: list[Date]) -> None:
    hazards = [0.01, 0.02, 0.025, 0.03]
    curve = InterpolatedHazardRateCurve(four_pillar_dates, hazards, Actual365Fixed())
    # Past last knot: h = 0.03 (flat extrapolation). Must opt into
    # extrapolation explicitly via the ``extrapolate`` flag.
    h_past = curve.hazard_rate(10.0, extrapolate=True)
    assert h_past == 0.03


def test_hr_curve_rejects_negative_hazard(four_pillar_dates: list[Date]) -> None:
    hazards = [0.01, -0.01, 0.025, 0.03]
    with pytest.raises(LibraryException, match="negative hazard rate"):
        InterpolatedHazardRateCurve(four_pillar_dates, hazards, Actual365Fixed())


def test_hr_curve_inspectors(four_pillar_dates: list[Date]) -> None:
    hazards = [0.01, 0.02, 0.025, 0.03]
    curve = InterpolatedHazardRateCurve(four_pillar_dates, hazards, Actual365Fixed())
    assert curve.hazard_rates() == hazards
    assert curve.dates() == four_pillar_dates
    assert len(curve.times()) == 4


# --- InterpolatedDefaultDensityCurve (Linear) -------------------------------


def test_dd_curve_density_at_node_and_interior(
    cpp_ref: dict[str, Any], four_pillar_dates: list[Date],
) -> None:
    densities = [0.01, 0.012, 0.015, 0.02]
    curve = InterpolatedDefaultDensityCurve(four_pillar_dates, densities, Actual365Fixed())
    ref = cpp_ref["interpolated_default_density"]
    tolerance.tight(curve.default_density(1.0), ref["density_at_1y"])
    tolerance.tight(curve.default_density(0.5), ref["density_at_05"])


def test_dd_curve_survival_via_primitive(
    cpp_ref: dict[str, Any], four_pillar_dates: list[Date],
) -> None:
    densities = [0.01, 0.012, 0.015, 0.02]
    curve = InterpolatedDefaultDensityCurve(four_pillar_dates, densities, Actual365Fixed())
    ref = cpp_ref["interpolated_default_density"]
    # TIGHT: S(t) = 1 - integral; Linear primitive is closed-form.
    tolerance.tight(curve.survival_probability(1.0), ref["sp_t1"])
    tolerance.tight(curve.survival_probability(2.0), ref["sp_t2"])


def test_dd_curve_rejects_negative_density(four_pillar_dates: list[Date]) -> None:
    densities = [0.01, -0.012, 0.015, 0.02]
    with pytest.raises(LibraryException, match="negative default density"):
        InterpolatedDefaultDensityCurve(four_pillar_dates, densities, Actual365Fixed())


def test_dd_curve_inspectors(four_pillar_dates: list[Date]) -> None:
    densities = [0.01, 0.012, 0.015, 0.02]
    curve = InterpolatedDefaultDensityCurve(four_pillar_dates, densities, Actual365Fixed())
    assert curve.default_densities() == densities
    assert curve.dates() == four_pillar_dates
