"""Cross-validate the affine survival term structures against C++ QuantLib v1.43.

Covers ``OneFactorAffineSurvivalStructure`` (onefactoraffinesurvival.hpp) and
``InterpolatedAffineHazardRateCurve`` + the ``AffineHazardRate`` bootstrap trait
(interpolatedaffinehazardratecurve.hpp).

Probe source: migration-harness/cpp/probes/v143_experimental_creditts/probe.cpp
              (``blockC`` at :318-376, ``blockD`` at :381-500)
Reference:    migration-harness/references/v143/experimental/creditts.json

Both curves are driven by the same Feller-constrained CIR model the probe
builds, so the discount-bond surface is pinned first: if ``cir_*`` matches, a
later mismatch is the credit wrapper's, not the short-rate model's.

Everything is TIGHT. Neither curve runs a solver — the interpolated curve is
constructed from given hazard rates rather than bootstrapped — so there is no
solver path to amplify floating-point noise.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.credit.interpolated_affine_hazard_rate_curve import (
    AffineHazardRate,
    InterpolatedAffineHazardRateCurve,
)
from pquantlib.experimental.credit.one_factor_affine_survival import (
    OneFactorAffineSurvivalStructure,
)
from pquantlib.models.shortrate.onefactor.cox_ingersoll_ross import CoxIngersollRoss
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

#: probe.cpp:140.
EVAL_DATE = Date.from_ymd(15, Month.January, 2024)
#: probe.cpp:317 — the shared time grid, including t = 0.
T_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0)
#: probe.cpp:394 — the input hazard rates of the interpolated curve.
INPUT_HAZARD_RATES: tuple[float, ...] = (0.008, 0.008, 0.012, 0.016, 0.021)


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    """The C++ v1.43 reference emitted by the creditts probe."""
    return reference_reader.load("v143/experimental/creditts")


def make_cir() -> CoxIngersollRoss:
    """probe.cpp:319-322 — ``CoxIngersollRoss(0.02, 0.04, 0.5, 0.1, true)``."""
    return CoxIngersollRoss(0.02, 0.04, 0.5, 0.1, with_feller_constraint=True)


def make_survival_structure() -> OneFactorAffineSurvivalStructure:
    """probe.cpp:332 — fixed reference date, NullCalendar, Actual365Fixed."""
    return OneFactorAffineSurvivalStructure(
        make_cir(),
        reference_date=EVAL_DATE,
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
    )


def make_affine_curve() -> InterpolatedAffineHazardRateCurve:
    """probe.cpp:386-396 — BackwardFlat over five pillars out to 10y."""
    dates = [
        EVAL_DATE,
        EVAL_DATE + Period(1, TimeUnit.Years),
        EVAL_DATE + Period(3, TimeUnit.Years),
        EVAL_DATE + Period(5, TimeUnit.Years),
        EVAL_DATE + Period(10, TimeUnit.Years),
    ]
    return InterpolatedAffineHazardRateCurve(
        dates,
        list(INPUT_HAZARD_RATES),
        Actual365Fixed(),
        make_cir(),
        NullCalendar(),
    )


def _assert_array(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        tolerance.tight(a, e)


# -----------------------------------------------------------------------------
# The CIR model underneath — pinned first so a later mismatch localises.
# -----------------------------------------------------------------------------


def test_cir_initial_state_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cir = make_cir()
    dyn = cir.dynamics()
    tolerance.exact(dyn.process.x0(), cpp_ref["cir_r0"])
    tolerance.exact(
        dyn.short_rate(0.0, dyn.process.x0()), cpp_ref["cir_short_rate_at_x0"]
    )


def test_cir_discount_bond_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    cir = make_cir()
    _assert_array(
        [cir.discount_bond_scalar(0.0, t, 0.02) for t in T_GRID],
        cpp_ref["cir_discount_bond_r002"],
    )


# -----------------------------------------------------------------------------
# OneFactorAffineSurvivalStructure
# -----------------------------------------------------------------------------


def test_survival_structure_dates_match_cpp(cpp_ref: dict[str, Any]) -> None:
    ts = make_survival_structure()
    assert ts.reference_date().serial_number() == cpp_ref["ofas_reference_date"]
    assert ts.max_date().serial_number() == cpp_ref["ofas_max_date"]


def test_survival_structure_probabilities_match_cpp(cpp_ref: dict[str, Any]) -> None:
    ts = make_survival_structure()
    _assert_array(
        [ts.survival_probability(t, True) for t in T_GRID],
        cpp_ref["ofas_survival_probability"],
    )
    _assert_array(
        [ts.default_probability(t, True) for t in T_GRID],
        cpp_ref["ofas_default_probability"],
    )


def test_survival_structure_hazard_and_density_are_zero(
    cpp_ref: dict[str, Any],
) -> None:
    """The deterministic component is identically zero on the base class.

    # C++ parity: ``hazardRateImpl`` returns 0 (onefactoraffinesurvival.hpp:
    # 140-143), which makes ``defaultDensityImpl`` zero too. Pinned rather
    # than "fixed" — the reference says so.
    """
    ts = make_survival_structure()
    _assert_array(
        [ts.hazard_rate(t, True) for t in T_GRID], cpp_ref["ofas_hazard_rate"]
    )
    _assert_array(
        [ts.default_density(t, True) for t in T_GRID],
        cpp_ref["ofas_default_density"],
    )


def test_survival_structure_conditional_survival_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    """The full (t_fwd, t_tgt, y) grid the probe walks — probe.cpp:349-367."""
    ts = make_survival_structure()
    actual = [
        ts.conditional_survival_probability(tf, tt, yv, True)
        for tf, tt, yv in zip(
            cpp_ref["ofas_csp_tfwd"],
            cpp_ref["ofas_csp_ttgt"],
            cpp_ref["ofas_csp_yval"],
            strict=True,
        )
    ]
    _assert_array(actual, cpp_ref["ofas_csp"])


def test_survival_structure_conditional_survival_date_overload(
    cpp_ref: dict[str, Any],
) -> None:
    """probe.cpp:370-374 — the Date overload converts and re-enters."""
    ts = make_survival_structure()
    d_fwd = EVAL_DATE + Period(1, TimeUnit.Years)
    d_tgt = EVAL_DATE + Period(5, TimeUnit.Years)
    assert d_fwd.serial_number() == cpp_ref["ofas_csp_date_dfwd"]
    assert d_tgt.serial_number() == cpp_ref["ofas_csp_date_dtgt"]
    tolerance.tight(
        ts.conditional_survival_probability(d_fwd, d_tgt, 0.02, True),
        cpp_ref["ofas_csp_date"],
    )


# -----------------------------------------------------------------------------
# InterpolatedAffineHazardRateCurve
# -----------------------------------------------------------------------------


def test_affine_curve_inspectors_match_cpp(cpp_ref: dict[str, Any]) -> None:
    curve = make_affine_curve()
    assert [d.serial_number() for d in curve.dates()] == cpp_ref["iahrc_out_dates"]
    assert curve.reference_date().serial_number() == cpp_ref["iahrc_reference_date"]
    assert curve.max_date().serial_number() == cpp_ref["iahrc_max_date"]
    _assert_array(curve.times(), cpp_ref["iahrc_times"])
    _assert_array(curve.data(), cpp_ref["iahrc_data"])
    _assert_array(curve.hazard_rates(), cpp_ref["iahrc_hazard_rates"])
    assert list(INPUT_HAZARD_RATES) == cpp_ref["iahrc_input_hazard_rates"]


def test_affine_curve_nodes_match_cpp(cpp_ref: dict[str, Any]) -> None:
    curve = make_affine_curve()
    nodes = curve.nodes()
    assert [d.serial_number() for d, _ in nodes] == cpp_ref["iahrc_node_dates"]
    _assert_array([v for _, v in nodes], cpp_ref["iahrc_node_values"])


def test_affine_curve_date_overloads_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """On-node and off-node dates, including past the last pillar."""
    curve = make_affine_curve()
    dates = [Date(int(s)) for s in cpp_ref["iahrc_probe_dates"]]
    _assert_array(
        [curve.time_from_reference(d) for d in dates], cpp_ref["iahrc_probe_times"]
    )
    _assert_array(
        [curve.hazard_rate(d, True) for d in dates], cpp_ref["iahrc_probe_hazard_rate"]
    )
    _assert_array(
        [curve.survival_probability(d, True) for d in dates],
        cpp_ref["iahrc_probe_survival_probability"],
    )
    _assert_array(
        [curve.default_probability(d, True) for d in dates],
        cpp_ref["iahrc_probe_default_probability"],
    )
    _assert_array(
        [curve.default_density(d, True) for d in dates],
        cpp_ref["iahrc_probe_default_density"],
    )


def test_affine_curve_time_overloads_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """Includes t = 0, which takes the early-out branch."""
    curve = make_affine_curve()
    assert list(T_GRID) == cpp_ref["iahrc_t_grid"]
    _assert_array(
        [curve.survival_probability(t, True) for t in T_GRID],
        cpp_ref["iahrc_time_survival_probability"],
    )
    _assert_array(
        [curve.hazard_rate(t, True) for t in T_GRID],
        cpp_ref["iahrc_time_hazard_rate"],
    )


def test_affine_curve_conditional_survival_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """The interpolated-curve override over the full grid — probe.cpp:456-472."""
    curve = make_affine_curve()
    actual = [
        curve.conditional_survival_probability(tf, tt, yv, True)
        for tf, tt, yv in zip(
            cpp_ref["iahrc_csp_tfwd"],
            cpp_ref["iahrc_csp_ttgt"],
            cpp_ref["iahrc_csp_yval"],
            strict=True,
        )
    ]
    _assert_array(actual, cpp_ref["iahrc_csp"])


def test_affine_curve_conditional_survival_edge_branches(
    cpp_ref: dict[str, Any],
) -> None:
    """The two special branches: equal times, and a zero forward time."""
    curve = make_affine_curve()
    tolerance.tight(
        curve.conditional_survival_probability(3.0, 3.0, 0.05, True),
        cpp_ref["iahrc_csp_equal_times"],
    )
    tolerance.tight(
        curve.conditional_survival_probability(0.0, 4.0, 0.05, True),
        cpp_ref["iahrc_csp_zero_fwd"],
    )


# -----------------------------------------------------------------------------
# AffineHazardRate bootstrap trait
# -----------------------------------------------------------------------------


def test_affine_hazard_rate_trait_constants_match_cpp(cpp_ref: dict[str, Any]) -> None:
    curve = make_affine_curve()
    assert AffineHazardRate.max_iterations() == cpp_ref["aff_traits_max_iterations"]
    tolerance.exact(
        AffineHazardRate.initial_value(curve), cpp_ref["aff_traits_initial_value"]
    )
    assert (
        AffineHazardRate.initial_date(curve).serial_number()
        == cpp_ref["aff_traits_initial_date"]
    )


def test_affine_hazard_rate_trait_guess_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    """``guess`` seeds pillar 1 with a literal and extrapolates the rest.

    # C++ parity: interpolatedaffinehazardratecurve.hpp:171-192 — the
    # ``avgHazardRate`` seed is commented out in C++, so i == 1 returns
    # ``0.0001``; later pillars read the curve, not ``data[i-1]``.
    """
    curve = make_affine_curve()
    tolerance.exact(
        AffineHazardRate.guess(1, curve, False), cpp_ref["aff_traits_guess_i1_invalid"]
    )
    tolerance.tight(
        AffineHazardRate.guess(2, curve, False), cpp_ref["aff_traits_guess_i2_invalid"]
    )
    tolerance.tight(
        AffineHazardRate.guess(4, curve, False), cpp_ref["aff_traits_guess_i4_invalid"]
    )
    tolerance.exact(
        AffineHazardRate.guess(2, curve, True), cpp_ref["aff_traits_guess_i2_valid"]
    )


def test_affine_hazard_rate_trait_bounds_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """``min_value_after`` floors at -1, not QL_EPSILON: the deterministic
    component of a ++ model is allowed to be negative.
    """
    curve = make_affine_curve()
    tolerance.exact(
        AffineHazardRate.min_value_after(2, curve, False),
        cpp_ref["aff_traits_min_after_invalid"],
    )
    tolerance.exact(
        AffineHazardRate.max_value_after(2, curve, False),
        cpp_ref["aff_traits_max_after_invalid"],
    )
    tolerance.tight(
        AffineHazardRate.min_value_after(2, curve, True),
        cpp_ref["aff_traits_min_after_valid"],
    )
    tolerance.tight(
        AffineHazardRate.max_value_after(2, curve, True),
        cpp_ref["aff_traits_max_after_valid"],
    )


def test_affine_hazard_rate_trait_update_guess_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    """``i == 1`` writes two slots; any other ``i`` writes one."""
    data = [1.0, 2.0, 3.0, 4.0]
    AffineHazardRate.update_guess(data, 9.0, 2)
    _assert_array(data, cpp_ref["aff_traits_update_guess_i2"])

    data = [1.0, 2.0, 3.0, 4.0]
    AffineHazardRate.update_guess(data, 9.0, 1)
    _assert_array(data, cpp_ref["aff_traits_update_guess_i1"])
