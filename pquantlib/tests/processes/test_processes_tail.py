"""Cross-validation for the ql/processes coverage tail @ v1.43.

Covers ``EndEulerDiscretization``, ``GeometricBrownianMotionProcess``,
``GarmanKohlagenProcess``, ``HullWhiteProcess``, ``Merton76Process``,
``MfStateProcess``, ``JointStochasticProcess`` (+ ``CachingKey``),
``HybridHestonHullWhiteProcess`` and ``HestonSLVProcess``.

Every expected value comes from
``migration-harness/references/v143/processes/tail.json``, emitted by
``migration-harness/cpp/probes/v143_processes_tail/probe.cpp`` built against
C++ QuantLib v1.43. No expected value in this module is invented.

The market data below reproduces the probe's setup exactly — in particular
the two NON-FLAT zero curves, without which the instantaneous forward rate
and the 1bp forward rate coincide and the drift formulas become
indistinguishable.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.processes.end_euler_discretization import EndEulerDiscretization
from pquantlib.processes.euler_discretization import EulerDiscretization
from pquantlib.processes.garman_kohlagen_process import GarmanKohlagenProcess
from pquantlib.processes.geometric_brownian_motion_process import (
    GeometricBrownianMotionProcess,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.processes.heston_slv_process import HestonSLVProcess
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.processes.hull_white_process import HullWhiteProcess
from pquantlib.processes.hybrid_heston_hull_white_process import (
    HybridHestonHullWhiteProcess,
)
from pquantlib.processes.joint_stochastic_process import (
    CachingKey,
    JointStochasticProcess,
)
from pquantlib.processes.merton76_process import Merton76Process
from pquantlib.processes.mf_state_process import MfStateProcess
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.processes.stochastic_process_array import (
    StochasticProcessArray,
    _spectral_sqrt,  # pyright: ignore[reportPrivateUsage]  # white-box: pins pseudoSqrt(Spectral)
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# probe.cpp:109 — ``const Date kToday(15, June, 2026);``
TODAY: Final[Date] = Date.from_ymd(15, Month.June, 2026)
DC: Final[Actual365Fixed] = Actual365Fixed()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/processes/tail")


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the evaluation date to the probe's.

    The probe sets ``Settings::instance().evaluationDate() = kToday`` at
    probe.cpp:main (``Date(15, June, 2026)``, probe.cpp:109). Every curve here
    is anchored on an explicit reference date, but the global is pinned anyway
    so no future edit can make this module wall-clock dependent.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


# --- market data (mirrors probe.cpp:159-200) ---------------------------------


def _curve_dates() -> list[Date]:
    return [
        TODAY,
        TODAY + Period(6, TimeUnit.Months),
        TODAY + Period(1, TimeUnit.Years),
        TODAY + Period(3, TimeUnit.Years),
        TODAY + Period(7, TimeUnit.Years),
        TODAY + Period(15, TimeUnit.Years),
    ]


def _domestic_curve() -> YieldTermStructure:
    return InterpolatedZeroCurve(
        _curve_dates(), [0.0180, 0.0225, 0.0265, 0.0310, 0.0345, 0.0362], DC
    )


def _foreign_curve() -> YieldTermStructure:
    return InterpolatedZeroCurve(
        _curve_dates(), [0.0090, 0.0105, 0.0135, 0.0118, 0.0142, 0.0175], DC
    )


def _black_vol(v: float) -> BlackVolTermStructure:
    return BlackConstantVol(
        reference_date=TODAY, calendar=NullCalendar(), volatility=v, day_counter=DC
    )


def _leverage_surface() -> LocalVolTermStructure:
    """probe.cpp:203-217 — a genuine (time, strike) leverage grid."""
    times = [0.25, 0.50, 1.00, 2.00]
    strikes = [80.0, 90.0, 100.0, 110.0, 125.0]
    values = np.array(
        [
            [1.35, 1.28, 1.19, 1.11],
            [1.22, 1.16, 1.09, 1.04],
            [1.05, 1.02, 1.00, 0.98],
            [0.94, 0.93, 0.92, 0.915],
            [0.86, 0.87, 0.885, 0.90],
        ],
        dtype=np.float64,
    )
    return FixedLocalVolSurface(
        reference_date=TODAY,
        times=times,
        strikes=strikes,
        local_vol_matrix=values,
        day_counter=DC,
    )


def _heston() -> HestonProcess:
    """probe.cpp — the shared Heston fixture (v0/kappa/theta/sigma/rho all distinct)."""
    return HestonProcess(
        risk_free_rate=_domestic_curve(),
        dividend_yield=_foreign_curve(),
        s0=SimpleQuote(102.5),
        v0=0.0625,
        kappa=1.35,
        theta=0.0475,
        sigma=0.62,
        rho=-0.40,
    )


def _arr(values: Sequence[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _assert_array(actual: npt.NDArray[np.float64], expected: Sequence[float]) -> None:
    assert len(actual) == len(expected)
    for i, e in enumerate(expected):
        tight(float(actual[i]), float(e))


def _assert_matrix(
    actual: npt.NDArray[np.float64], expected: Sequence[Sequence[float]]
) -> None:
    assert actual.shape == (len(expected), len(expected[0]))
    for i, row in enumerate(expected):
        for j, e in enumerate(row):
            tight(float(actual[i, j]), float(e))


# =============================================================================
# GeometricBrownianMotionProcess
# =============================================================================


def _gbm() -> GeometricBrownianMotionProcess:
    return GeometricBrownianMotionProcess(95.5, 0.073, 0.231)


def test_gbm_constructor_arguments_are_all_retained(cpp: dict[str, Any]) -> None:
    """All three arguments are distinct and none may be dropped or swapped."""
    ref = cpp["gbm"]
    p = _gbm()
    exact(p.x0(), ref["init_initial_value"])
    exact(p.mue(), ref["init_mue"])
    exact(p.sigma(), ref["init_sigma"])


def test_gbm_matches_cpp(cpp: dict[str, Any]) -> None:
    ref = cpp["gbm"]
    p = _gbm()
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    tight(p.x0(), ref["x0"])
    _assert_array(p.initial_values(), ref["initial_values"])
    tight(p.drift_1d(0.0, 95.5), ref["drift_t0_x95"])
    tight(p.drift_1d(2.0, 120.0), ref["drift_t2_x120"])
    tight(p.drift_1d(2.0, -30.0), ref["drift_t2_xneg"])
    tight(p.diffusion_1d(0.0, 95.5), ref["diffusion_t0_x95"])
    tight(p.diffusion_1d(2.0, 120.0), ref["diffusion_t2_x120"])
    _assert_array(p.drift(1.5, _arr([110.0])), ref["drift_vec"])
    _assert_matrix(p.diffusion(1.5, _arr([110.0])), ref["diffusion_mat"])
    tight(p.expectation_1d(1.5, 110.0, 0.25), ref["expectation"])
    tight(p.std_deviation_1d(1.5, 110.0, 0.25), ref["std_deviation"])
    tight(p.variance_1d(1.5, 110.0, 0.25), ref["variance"])
    tight(p.apply_1d(110.0, 3.75), ref["apply"])
    tight(p.evolve_1d(1.5, 110.0, 0.25, 0.83), ref["evolve"])
    _assert_array(p.evolve(1.5, _arr([110.0]), 0.25, _arr([0.83])), ref["evolve_vec"])
    _assert_matrix(p.covariance(1.5, _arr([110.0]), 0.25), ref["covariance_mat"])


def test_gbm_expectation_is_the_euler_step_not_the_exponential(cpp: dict[str, Any]) -> None:
    """C++ supplies an EulerDiscretization and overrides nothing else.

    So ``expectation == x0 * (1 + mue*dt)``, NOT ``x0 * exp(mue*dt)``. This is
    an approximation the C++ makes on purpose; a port that "improved" it would
    still pass a lone drift check.
    """
    ref = cpp["gbm"]
    expected_euler = 110.0 * (1.0 + 0.073 * 0.25)
    tight(expected_euler, ref["expectation"])
    assert not math.isclose(110.0 * math.exp(0.073 * 0.25), ref["expectation"], rel_tol=1e-9)


# =============================================================================
# GarmanKohlagenProcess
# =============================================================================


def _gk(force: bool = False) -> GarmanKohlagenProcess:
    return GarmanKohlagenProcess(
        x0=SimpleQuote(1.2345),
        foreign_risk_free_ts=_foreign_curve(),
        domestic_risk_free_ts=_domestic_curve(),
        black_vol_ts=_black_vol(0.1725),
        force_discretization=force,
    )


def test_garman_kohlagen_maps_foreign_to_dividend_and_domestic_to_riskfree(
    cpp: dict[str, Any],
) -> None:
    """The only thing this class does is the argument mapping; pin it directly.

    A swap of the two curves would leave every OTHER assertion in the drift
    test passing in sign but wrong in value, so the accessors are pinned on
    their own.
    """
    ref = cpp["garman_kohlagen"]
    p = _gk()
    tight(p.dividend_yield().discount(2.0), ref["dividend_yield_discount_2y"])
    tight(p.risk_free_rate().discount(2.0), ref["risk_free_rate_discount_2y"])
    # And they must be different curves, or the test above proves nothing.
    assert ref["dividend_yield_discount_2y"] != ref["risk_free_rate_discount_2y"]


def test_garman_kohlagen_matches_cpp(cpp: dict[str, Any]) -> None:
    ref = cpp["garman_kohlagen"]
    p = _gk()
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    tight(p.x0(), ref["x0"])
    tight(p.state_variable().value(), ref["state_variable_value"])
    tight(p.black_volatility().black_vol_at_time(1.0, 1.2345, True), ref["black_vol_1y_atm"])
    tight(p.local_volatility().local_vol_at_time(1.0, 1.2345, True), ref["local_vol_1y_atm"])
    tight(p.drift_1d(0.0, 1.2345), ref["drift_t0"])
    tight(p.drift_1d(1.5, 1.31), ref["drift_t1_5"])
    tight(p.diffusion_1d(1.5, 1.31), ref["diffusion_t1_5"])
    tight(p.expectation_1d(1.5, 1.31, 0.5), ref["expectation"])
    tight(p.std_deviation_1d(1.5, 1.31, 0.5), ref["std_deviation"])
    tight(p.variance_1d(1.5, 1.31, 0.5), ref["variance"])
    tight(p.apply_1d(1.31, 0.042), ref["apply"])
    tight(p.evolve_1d(1.5, 1.31, 0.5, -0.62), ref["evolve"])
    tight(p.time(TODAY + 365), ref["time_1y"])


def test_garman_kohlagen_force_discretization_is_threaded(cpp: dict[str, Any]) -> None:
    """``force_discretization=True`` must reach the base and change the answer."""
    ref = cpp["garman_kohlagen"]
    forced = _gk(force=True)
    tight(forced.variance_1d(1.5, 1.31, 0.5), ref["forced_variance"])
    tight(forced.std_deviation_1d(1.5, 1.31, 0.5), ref["forced_std_deviation"])
    tight(forced.evolve_1d(1.5, 1.31, 0.5, -0.62), ref["forced_evolve"])
    # The flag must actually matter, otherwise it could be silently dropped.
    assert ref["forced_variance"] != ref["variance"]
    assert ref["forced_evolve"] != ref["evolve"]


# =============================================================================
# HullWhiteProcess
# =============================================================================


def _hw() -> HullWhiteProcess:
    return HullWhiteProcess(_domestic_curve(), 0.07, 0.013)


def test_hull_white_process_matches_cpp(cpp: dict[str, Any]) -> None:
    ref = cpp["hull_white"]
    p = _hw()
    exact(p.a(), ref["init_a"])
    exact(p.sigma(), ref["init_sigma"])
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    tight(p.x0(), ref["x0"])
    _assert_array(p.initial_values(), ref["initial_values"])
    tight(p.alpha(0.0), ref["alpha_t0"])
    tight(p.alpha(1.0), ref["alpha_t1"])
    tight(p.alpha(5.0), ref["alpha_t5"])
    tight(p.drift_1d(0.0, 0.018), ref["drift_t0"])
    tight(p.drift_1d(1.5, 0.0245), ref["drift_t1_5"])
    tight(p.drift_1d(4.0, 0.031), ref["drift_t4"])
    tight(p.diffusion_1d(1.5, 0.0245), ref["diffusion_t1_5"])
    tight(p.expectation_1d(1.5, 0.0245, 0.5), ref["expectation"])
    tight(p.std_deviation_1d(1.5, 0.0245, 0.5), ref["std_deviation"])
    tight(p.variance_1d(1.5, 0.0245, 0.5), ref["variance"])
    tight(p.apply_1d(0.0245, 0.0031), ref["apply"])
    tight(p.evolve_1d(1.5, 0.0245, 0.5, 0.71), ref["evolve"])
    _assert_array(p.drift(1.5, _arr([0.0245])), ref["drift_vec"])
    _assert_matrix(p.diffusion(1.5, _arr([0.0245])), ref["diffusion_mat"])
    _assert_matrix(p.covariance(1.5, _arr([0.0245]), 0.5), ref["covariance_mat"])


def test_hull_white_process_rejects_negative_parameters(cpp: dict[str, Any]) -> None:
    """C++ HullWhiteProcess DOES range-check; HullWhiteForwardProcess does not.

    Which message comes back is NOT guessed: ``cpp["errors"]`` records the C++
    ``what()``. For a negative sigma the OrnsteinUhlenbeckProcess member
    initialiser fires first, so the body's "negative sigma given" QL_REQUIRE is
    unreachable and the observable message is "negative volatility given".
    """
    errors = cpp["errors"]
    assert errors["hw_negative_a"] == "negative a given"
    assert errors["hw_negative_sigma"] == "negative volatility given"
    curve = _domestic_curve()
    with pytest.raises(LibraryException, match=errors["hw_negative_a"]):
        HullWhiteProcess(curve, -0.01, 0.013)
    with pytest.raises(LibraryException, match=errors["hw_negative_sigma"]):
        HullWhiteProcess(curve, 0.07, -0.013)
    # The forward variant has no QL_REQUIREs of its own, so a negative ``a``
    # goes through — but its OrnsteinUhlenbeckProcess member initialiser still
    # rejects a negative sigma. Both facts come from cpp["errors"].
    assert errors["hwf_negative_a"] == "<no exception>"
    assert errors["hwf_negative_sigma"] == "negative volatility given"
    HullWhiteForwardProcess(curve, -0.01, 0.013)
    with pytest.raises(LibraryException, match=errors["hwf_negative_sigma"]):
        HullWhiteForwardProcess(curve, 0.07, -0.013)


def test_hull_white_process_differs_from_forward_process_only_by_the_measure(
    cpp: dict[str, Any],
) -> None:
    """drift/expectation drop the forward-measure corrections; the rest is shared."""
    ref = cpp["hull_white"]
    fwd = HullWhiteForwardProcess(_domestic_curve(), 0.07, 0.013)
    fwd.set_forward_measure_time(3.0)
    # diffusion / std_deviation / variance are measure-independent.
    tight(fwd.diffusion_1d(1.5, 0.0245), ref["diffusion_t1_5"])
    tight(fwd.variance_1d(1.5, 0.0245, 0.5), ref["variance"])
    # drift and expectation are not.
    expected_drift = ref["drift_t1_5"] - fwd.B(1.5, 3.0) * 0.013 * 0.013
    tight(fwd.drift_1d(1.5, 0.0245), expected_drift)
    expected_exp = ref["expectation"] - fwd.M_T(1.5, 2.0, 3.0)
    tight(fwd.expectation_1d(1.5, 0.0245, 0.5), expected_exp)


# =============================================================================
# Merton76Process
# =============================================================================


def _merton() -> Merton76Process:
    return Merton76Process(
        state_variable=SimpleQuote(97.25),
        dividend_ts=_foreign_curve(),
        risk_free_ts=_domestic_curve(),
        black_vol_ts=_black_vol(0.2130),
        jump_int=SimpleQuote(1.37),
        log_j_mean=SimpleQuote(-0.041),
        log_j_vol=SimpleQuote(0.0925),
    )


def test_merton76_inspectors_match_cpp(cpp: dict[str, Any]) -> None:
    """The three jump quotes are adjacent and same-typed: a permutation must fail."""
    ref = cpp["merton76"]
    p = _merton()
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    tight(p.x0(), ref["x0"])
    _assert_array(p.initial_values(), ref["initial_values"])
    tight(p.state_variable().value(), ref["state_variable"])
    tight(p.jump_intensity().value(), ref["jump_intensity"])
    tight(p.log_mean_jump().value(), ref["log_mean_jump"])
    tight(p.log_jump_volatility().value(), ref["log_jump_volatility"])
    tight(p.dividend_yield().discount(2.0), ref["dividend_yield_discount_2y"])
    tight(p.risk_free_rate().discount(2.0), ref["risk_free_rate_discount_2y"])
    tight(p.black_volatility().black_vol_at_time(1.0, 97.25, True), ref["black_vol_1y"])
    tight(p.time(TODAY + 365), ref["time_1y"])
    tight(p.time(TODAY + 1095), ref["time_3y"])


def test_merton76_dynamic_methods_all_fail() -> None:
    """merton76process.hpp:50-52 — drift / diffusion / apply are QL_FAILs."""
    p = _merton()
    with pytest.raises(LibraryException, match="does not implement drift"):
        p.drift_1d(1.0, 97.25)
    with pytest.raises(LibraryException, match="does not implement diffusion"):
        p.diffusion_1d(1.0, 97.25)
    with pytest.raises(LibraryException, match="does not implement apply"):
        p.apply_1d(97.25, 0.01)
    # ... and therefore so does everything routed through them.
    with pytest.raises(LibraryException, match="does not implement"):
        p.expectation_1d(1.0, 97.25, 0.5)
    with pytest.raises(LibraryException, match="does not implement"):
        p.evolve_1d(1.0, 97.25, 0.5, 0.3)


# =============================================================================
# MfStateProcess
# =============================================================================

_MF_TIMES: Final[list[float]] = [0.5, 1.0, 2.0, 5.0]
_MF_VOLS: Final[list[float]] = [0.0050, 0.0075, 0.0090, 0.0110, 0.0130]
_MF_SAMPLE_TIMES: Final[list[float]] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.5, 6.0]


def _mf_a() -> MfStateProcess:
    return MfStateProcess(0.031, _MF_TIMES, _MF_VOLS)


def _mf_b() -> MfStateProcess:
    return MfStateProcess(0.017, [0.75, 1.5, 3.0], [0.0062, 0.0081, 0.0104, 0.0121])


def test_mf_state_process_matches_cpp(cpp: dict[str, Any]) -> None:
    ref = cpp["mf_state"]
    a = _mf_a()
    assert a.size() == ref["size"]
    assert a.factors() == ref["factors"]
    tight(a.x0(), ref["x0"])
    _assert_array(a.initial_values(), ref["initial_values"])
    exact(a.drift_1d(1.25, 0.03), ref["drift"])
    exact(a.reversion(), 0.031)

    # The bucket lookup is upper_bound (STRICTLY greater), so t on a node
    # selects the bucket above it. Sampled inside every bucket and on every node.
    for i, t in enumerate(_MF_SAMPLE_TIMES):
        exact(a.diffusion_1d(t, 0.02), ref[f"diffusion_t{i}"])

    exact(a.expectation_1d(1.25, 0.037, 0.5), ref["expectation"])
    exact(a.expectation_1d(3.0, 0.037, 1.5), ref["expectation_other_t"])
    tight(a.variance_1d(1.25, 0.02, 0.2), ref["variance_within_bucket"])
    tight(a.variance_1d(0.75, 0.02, 0.5), ref["variance_one_boundary"])
    tight(a.variance_1d(0.25, 0.02, 5.0), ref["variance_many_buckets"])
    tight(a.variance_1d(1.0, 0.02, 1.0), ref["variance_on_node"])
    exact(a.variance_1d(1.25, 0.02, 1e-17), ref["variance_tiny_dt"])
    tight(a.std_deviation_1d(1.25, 0.02, 0.2), ref["std_deviation"])
    tight(a.evolve_1d(1.25, 0.037, 0.5, -0.43), ref["evolve"])
    tight(a.apply_1d(0.037, 0.0021), ref["apply"])


def test_mf_state_process_zero_reversion_arm(cpp: dict[str, Any]) -> None:
    """``reversionZero_`` selects an entirely different variance formula."""
    ref = cpp["mf_state"]
    z = MfStateProcess(0.0, _MF_TIMES, _MF_VOLS)
    tight(z.variance_1d(1.25, 0.02, 0.2), ref["zero_variance_within_bucket"])
    tight(z.variance_1d(0.25, 0.02, 5.0), ref["zero_variance_many_buckets"])
    tight(z.std_deviation_1d(1.25, 0.02, 0.2), ref["zero_std_deviation"])
    exact(z.diffusion_1d(1.5, 0.02), ref["zero_diffusion_t1_5"])
    # The two arms must genuinely disagree.
    assert ref["zero_variance_within_bucket"] != ref["variance_within_bucket"]


def test_mf_state_process_empty_times_arm(cpp: dict[str, Any]) -> None:
    """``times_.empty()`` short-circuits variance to a single closed form."""
    ref = cpp["mf_state"]
    e0 = MfStateProcess(0.0, [], [0.0083])
    en = MfStateProcess(0.029, [], [0.0083])
    tight(e0.variance_1d(1.25, 0.02, 0.5), ref["empty_zero_variance"])
    tight(en.variance_1d(1.25, 0.02, 0.5), ref["empty_nonzero_variance"])
    exact(e0.diffusion_1d(3.3, 0.02), ref["empty_diffusion"])
    tight(en.std_deviation_1d(1.25, 0.02, 0.5), ref["empty_nonzero_std_deviation"])


def test_mf_state_process_validates_its_grids(cpp: dict[str, Any]) -> None:
    """mfstateprocess.cpp:43-56 — three separate QL_REQUIREs.

    The messages (including their interpolated values) come from the C++
    ``what()`` recorded in ``cpp["errors"]``, so the port reproduces the text
    rather than an approximation of it.
    """
    errors = cpp["errors"]
    with pytest.raises(LibraryException) as bad_count:
        MfStateProcess(0.03, [0.5, 1.0], [0.01, 0.01])
    assert str(bad_count.value) == errors["mf_bad_vol_count"]

    with pytest.raises(LibraryException) as unsorted:
        MfStateProcess(0.03, [1.0, 0.5], [0.01, 0.01, 0.01])
    assert str(unsorted.value) == errors["mf_times_not_increasing"]

    with pytest.raises(LibraryException) as negative:
        MfStateProcess(0.03, [0.5, 1.0], [0.01, -0.01, 0.01])
    assert str(negative.value) == errors["mf_negative_vol"]

    # Empty vols: C++ compares against ``vols_.size() - 1`` in UNSIGNED
    # arithmetic, which wraps; Python's -1 fails the same requirement.
    with pytest.raises(LibraryException) as empty:
        MfStateProcess(0.03, [], [])
    assert str(empty.value) == errors["mf_empty_vols"]


def test_mf_state_process_setters_revalidate_and_notify() -> None:
    """``setTimes`` / ``setVols`` are MarkovFunctional-only but must still check."""
    p = _mf_a()
    p._set_vols([0.006, 0.008, 0.010, 0.012, 0.014])  # pyright: ignore[reportPrivateUsage]
    exact(p.diffusion_1d(0.25, 0.0), 0.006)
    p._set_times([0.25, 1.0, 2.0, 5.0])  # pyright: ignore[reportPrivateUsage]
    exact(p.diffusion_1d(0.5, 0.0), 0.008)
    with pytest.raises(LibraryException, match="number of volatilities"):
        p._set_vols([0.01])  # pyright: ignore[reportPrivateUsage]


# =============================================================================
# EndEulerDiscretization
# =============================================================================


def _check_1d_discretizations(
    ref: dict[str, Any], process: StochasticProcess1D
) -> None:
    euler = EulerDiscretization()
    end_euler = EndEulerDiscretization()
    t0, x0, dt = float(ref["t0"]), float(ref["x0"]), float(ref["dt"])
    tight(euler.drift(process, t0, x0, dt), ref["euler_drift"])
    tight(euler.diffusion(process, t0, x0, dt), ref["euler_diffusion"])
    tight(euler.variance(process, t0, x0, dt), ref["euler_variance"])
    tight(end_euler.drift(process, t0, x0, dt), ref["end_euler_drift"])
    tight(end_euler.diffusion(process, t0, x0, dt), ref["end_euler_diffusion"])
    tight(end_euler.variance(process, t0, x0, dt), ref["end_euler_variance"])


def test_end_euler_1d_on_piecewise_constant_vol(cpp: dict[str, Any]) -> None:
    """MfStateProcess: t0 and t0+dt land in different vol buckets."""
    ref = cpp["end_euler_mf"]
    _check_1d_discretizations(ref, _mf_a())
    # The whole point of the class: the diffusion must move.
    assert ref["end_euler_diffusion"] != ref["euler_diffusion"]
    assert ref["end_euler_variance"] != ref["euler_variance"]


def test_end_euler_1d_on_time_dependent_drift(cpp: dict[str, Any]) -> None:
    """HullWhiteProcess: the drift moves, the diffusion does not."""
    ref = cpp["end_euler_hw"]
    _check_1d_discretizations(ref, _hw())
    assert ref["end_euler_drift"] != ref["euler_drift"]
    exact(ref["end_euler_diffusion"], ref["euler_diffusion"])


def test_end_euler_multi_dimensional(cpp: dict[str, Any]) -> None:
    """Two correlated MfStateProcesses inside a StochasticProcessArray."""
    ref = cpp["end_euler_array"]
    corr = np.array([[1.0, -0.35], [-0.35, 1.0]], dtype=np.float64)
    spa: StochasticProcess = StochasticProcessArray([_mf_a(), _mf_b()], corr)
    euler = EulerDiscretization()
    end_euler = EndEulerDiscretization()
    t0, dt = ref["t0"], ref["dt"]
    x0 = _arr(ref["x0"])
    _assert_array(euler.drift(spa, t0, x0, dt), ref["euler_drift"])
    _assert_matrix(euler.diffusion(spa, t0, x0, dt), ref["euler_diffusion"])
    _assert_matrix(euler.covariance(spa, t0, x0, dt), ref["euler_covariance"])
    _assert_array(end_euler.drift(spa, t0, x0, dt), ref["end_euler_drift"])
    _assert_matrix(end_euler.diffusion(spa, t0, x0, dt), ref["end_euler_diffusion"])
    _assert_matrix(end_euler.covariance(spa, t0, x0, dt), ref["end_euler_covariance"])
    assert ref["end_euler_covariance"] != ref["euler_covariance"]


# =============================================================================
# StochasticProcessArray — the pseudoSqrt(Spectral) it exposes
#
# Not one of this wave's ten classes, but ``JointStochasticProcess`` and the
# ``EndEulerDiscretization`` multi-D case both lean on it, so it is
# cross-validated here rather than assumed.
# =============================================================================


def _spa(correlation: Sequence[Sequence[float]]) -> StochasticProcessArray:
    return StochasticProcessArray(
        [
            GeometricBrownianMotionProcess(100.0, 0.055, 0.21),
            GeometricBrownianMotionProcess(85.0, 0.032, 0.34),
        ],
        np.array(correlation, dtype=np.float64),
    )


_IDENTITY2: Final[list[list[float]]] = [[1.0, 0.0], [0.0, 1.0]]
_POS_CORR: Final[list[list[float]]] = [[1.0, 0.25], [0.25, 1.0]]
_NEG_CORR: Final[list[list[float]]] = [[1.0, -0.35], [-0.35, 1.0]]


def test_stochastic_process_array_spectral_sqrt_matches_cpp(cpp: dict[str, Any]) -> None:
    """The pseudo-root ITSELF, not just ``M @ M.T``, must match C++.

    ``diffusion`` / ``std_deviation`` / ``evolve`` all expose the root
    directly, so its column order and per-column sign are observable. The
    identity case is the one a naive "reverse the ascending eigensolver"
    implementation gets wrong: its eigenvalues are degenerate and C++ breaks
    the tie lexicographically on the eigenvectors, giving back the identity
    rather than a permutation of it.
    """
    ref = cpp["spa"]
    _assert_matrix(_spectral_sqrt(np.array(_IDENTITY2, dtype=np.float64)),
                   ref["pseudo_sqrt_identity2"])
    _assert_matrix(_spectral_sqrt(np.eye(3, dtype=np.float64)), ref["pseudo_sqrt_identity3"])
    _assert_matrix(_spectral_sqrt(np.array(_POS_CORR, dtype=np.float64)), ref["pseudo_sqrt_pos"])
    _assert_matrix(_spectral_sqrt(np.array(_NEG_CORR, dtype=np.float64)), ref["pseudo_sqrt_neg"])
    # C++ rankReducedSqrt agrees with pseudoSqrt on the identity too.
    assert ref["rank_reduced_sqrt_identity2"] == ref["pseudo_sqrt_identity2"]


def test_stochastic_process_array_identity_correlation_matches_cpp(
    cpp: dict[str, Any],
) -> None:
    ref = cpp["spa"]
    spa = _spa(_IDENTITY2)
    x0 = _arr([102.0, 88.0])
    dw = _arr([0.5, -0.3])
    _assert_matrix(spa.correlation(), ref["identity_correlation"])
    _assert_matrix(spa.diffusion(0.75, x0), ref["identity_diffusion"])
    _assert_matrix(spa.std_deviation(0.75, x0, 0.5), ref["identity_std_deviation"])
    _assert_array(spa.evolve(0.75, x0, 0.5, dw), ref["identity_evolve"])
    # Under identity correlation evolve must reduce to the component-wise 1-D
    # evolve; C++ confirms it does.
    tight(ref["identity_evolve"][0], ref["identity_evolve_component0"])
    tight(ref["identity_evolve"][1], ref["identity_evolve_component1"])


@pytest.mark.parametrize(
    ("correlation", "diffusion_key", "evolve_key"),
    [(_POS_CORR, "pos_diffusion", "pos_evolve"), (_NEG_CORR, "neg_diffusion", "neg_evolve")],
)
def test_stochastic_process_array_correlated_matches_cpp(
    cpp: dict[str, Any],
    correlation: Sequence[Sequence[float]],
    diffusion_key: str,
    evolve_key: str,
) -> None:
    ref = cpp["spa"]
    spa = _spa(correlation)
    x0 = _arr([102.0, 88.0])
    dw = _arr([0.5, -0.3])
    _assert_matrix(spa.diffusion(0.75, x0), ref[diffusion_key])
    _assert_array(spa.evolve(0.75, x0, 0.5, dw), ref[evolve_key])


def test_stochastic_process_array_std_deviation_and_covariance(
    cpp: dict[str, Any],
) -> None:
    """``covariance`` is sign-blind but ``std_deviation`` is not; pin both."""
    ref = cpp["spa"]
    x0 = _arr([102.0, 88.0])
    _assert_matrix(_spa(_POS_CORR).std_deviation(0.75, x0, 0.5), ref["pos_std_deviation"])
    _assert_matrix(_spa(_NEG_CORR).covariance(0.75, x0, 0.5), ref["neg_covariance"])


# =============================================================================
# HestonProcess — the dependency HybridHestonHullWhite and HestonSLV lean on
# =============================================================================


def test_heston_dependency_uses_the_instantaneous_forward_rate(
    cpp: dict[str, Any],
) -> None:
    """``HestonProcess::drift`` reads forwardRate(t, t), not forwardRate(t, t+1e-4).

    This module's non-flat curves make the two visibly different (see
    ``rf_instantaneous_fwd_t1_5`` vs ``rf_1e4_fwd_t1_5`` in the reference); on a
    flat curve they coincide exactly, which is how the divergence this test
    now guards against survived earlier waves.
    """
    ref = cpp["heston_dependency"]
    assert ref["rf_instantaneous_fwd_t1_5"] != ref["rf_1e4_fwd_t1_5"]
    assert ref["div_instantaneous_fwd_t1_5"] != ref["div_1e4_fwd_t1_5"]

    rf = _domestic_curve()
    div = _foreign_curve()
    from pquantlib.time.compounding import Compounding  # noqa: PLC0415
    from pquantlib.time.frequency import Frequency  # noqa: PLC0415

    tight(
        rf.forward_rate(1.5, 1.5, Compounding.Continuous, Frequency.Annual).rate(),
        ref["rf_instantaneous_fwd_t1_5"],
    )
    tight(
        div.forward_rate(1.5, 1.5, Compounding.Continuous, Frequency.Annual).rate(),
        ref["div_instantaneous_fwd_t1_5"],
    )


def test_heston_dependency_matches_cpp(cpp: dict[str, Any]) -> None:
    ref = cpp["heston_dependency"]
    p = _heston()
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    _assert_array(p.initial_values(), ref["initial_values"])
    _assert_array(p.drift(0.0, _arr([102.5, 0.0625])), ref["drift_t0"])
    _assert_array(p.drift(1.5, _arr([108.0, 0.0512])), ref["drift_t1_5"])
    _assert_array(p.drift(4.0, _arr([95.0, 0.0710])), ref["drift_t4"])
    _assert_array(p.drift(1.5, _arr([108.0, -0.002])), ref["drift_negative_v"])
    _assert_matrix(p.diffusion(1.5, _arr([108.0, 0.0512])), ref["diffusion_t1_5"])
    _assert_matrix(p.diffusion(1.5, _arr([108.0, 0.0])), ref["diffusion_zero_v"])
    _assert_array(p.apply(_arr([108.0, 0.0512]), _arr([0.031, -0.0044])), ref["apply"])
    tight(p.time(TODAY + 365), ref["time_1y"])


# =============================================================================
# HybridHestonHullWhiteProcess
# =============================================================================


def _hhw(
    discretization: HybridHestonHullWhiteProcess.Discretization,
) -> HybridHestonHullWhiteProcess:
    hwf = HullWhiteForwardProcess(_domestic_curve(), 0.07, 0.013)
    hwf.set_forward_measure_time(3.0)
    return HybridHestonHullWhiteProcess(_heston(), hwf, -0.35, discretization)


@pytest.mark.parametrize(
    ("ref_key", "discretization"),
    [
        ("hhw_bsm", HybridHestonHullWhiteProcess.Discretization.BSMHullWhite),
        ("hhw_euler", HybridHestonHullWhiteProcess.Discretization.Euler),
    ],
)
def test_hybrid_heston_hull_white_matches_cpp(
    cpp: dict[str, Any],
    ref_key: str,
    discretization: HybridHestonHullWhiteProcess.Discretization,
) -> None:
    ref = cpp[ref_key]
    p = _hhw(discretization)
    assert int(p.discretization()) == ref["discretization"]
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    exact(p.eta(), ref["init_corr_equity_short_rate"])
    exact(p.hull_white_process().get_forward_measure_time(), ref["forward_measure_time"])
    _assert_array(p.initial_values(), ref["initial_values"])
    _assert_array(p.drift(0.75, _arr([102.5, 0.0625, 0.0215])), ref["drift"])
    _assert_matrix(p.diffusion(0.75, _arr([102.5, 0.0625, 0.0215])), ref["diffusion"])
    _assert_array(
        p.apply(_arr([102.5, 0.0625, 0.0215]), _arr([0.031, -0.0044, 0.0007])), ref["apply"]
    )
    _assert_array(
        p.evolve(0.75, _arr([102.5, 0.0625, 0.0215]), 0.5, _arr([0.63, -1.21, 0.47])),
        ref["evolve"],
    )
    _assert_array(
        p.evolve(1.75, _arr([97.0, 0.0410, 0.0288]), 0.25, _arr([-0.44, 0.91, -0.13])),
        ref["evolve_other"],
    )
    tight(p.numeraire(0.75, _arr([102.5, 0.0625, 0.0215])), ref["numeraire_t0"])
    tight(p.numeraire(2.0, _arr([95.0, 0.05, 0.0310])), ref["numeraire_t2"])
    tight(p.numeraire(3.0, _arr([95.0, 0.05, 0.0310])), ref["numeraire_t3"])
    tight(p.time(TODAY + 365), ref["time_1y"])


def test_hybrid_heston_hull_white_discretizations_disagree(cpp: dict[str, Any]) -> None:
    """The two schemes must take different branches, else the enum is dead."""
    assert cpp["hhw_bsm"]["evolve"] != cpp["hhw_euler"]["evolve"]
    assert cpp["hhw_bsm"]["discretization"] != cpp["hhw_euler"]["discretization"]
    # Euler == 0, BSMHullWhite == 1 (hybridhestonhullwhiteprocess.hpp:44).
    assert cpp["hhw_euler"]["discretization"] == 0
    assert cpp["hhw_bsm"]["discretization"] == 1


def test_hybrid_heston_hull_white_numeraire_is_one_at_the_horizon(
    cpp: dict[str, Any],
) -> None:
    """P(T, T; r)/P(0, T) reduces to 1/endDiscount at t == T."""
    ref = cpp["hhw_bsm"]
    tight(ref["numeraire_t3"], 1.0 / _domestic_curve().discount(3.0))


def test_hybrid_heston_hull_white_rejects_an_indefinite_correlation(
    cpp: dict[str, Any],
) -> None:
    """Both rejections are pinned to the C++ ``what()`` in ``cpp["errors"]``.

    Note the second one: the ``HullWhite`` MODEL member initialiser rejects a
    zero sigma (PositiveConstraint) before the body's "positive vol of Hull
    White process is required" QL_REQUIRE can run, so that message is
    unreachable in C++ too.
    """
    errors = cpp["errors"]
    assert errors["hhw_indefinite_correlation"] == "correlation matrix is not positive definite"
    assert "invalid value" in errors["hhw_zero_hw_sigma"]
    hwf = HullWhiteForwardProcess(_domestic_curve(), 0.07, 0.013)
    hwf.set_forward_measure_time(3.0)
    with pytest.raises(LibraryException, match="not positive definite"):
        HybridHestonHullWhiteProcess(_heston(), hwf, -0.95)
    zero_vol = HullWhiteForwardProcess(_domestic_curve(), 0.07, 0.0)
    zero_vol.set_forward_measure_time(3.0)
    with pytest.raises(LibraryException, match="invalid value"):
        HybridHestonHullWhiteProcess(_heston(), zero_vol, -0.35)


def test_hybrid_heston_hull_white_has_no_discretization() -> None:
    """C++ passes none to the base, so expectation/covariance are unavailable."""
    p = _hhw(HybridHestonHullWhiteProcess.Discretization.BSMHullWhite)
    with pytest.raises(LibraryException, match="no discretization provided"):
        p.expectation(0.75, _arr([102.5, 0.0625, 0.0215]), 0.5)
    with pytest.raises(LibraryException, match="no discretization provided"):
        p.covariance(0.75, _arr([102.5, 0.0625, 0.0215]), 0.5)


# =============================================================================
# HestonSLVProcess
# =============================================================================


def _slv(mixing: float = 0.75) -> HestonSLVProcess:
    return HestonSLVProcess(_heston(), _leverage_surface(), mixing)


def test_heston_slv_inspectors_match_cpp(cpp: dict[str, Any]) -> None:
    ref = cpp["heston_slv"]
    p = _slv()
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    tight(p.v0(), ref["v0"])
    tight(p.rho(), ref["rho"])
    tight(p.kappa(), ref["kappa"])
    tight(p.theta(), ref["theta"])
    tight(p.sigma(), ref["sigma"])
    exact(p.mixing_factor(), ref["init_mixing_factor"])
    exact(HestonSLVProcess(_heston(), _leverage_surface()).mixing_factor(),
          ref["default_mixing_factor"])
    tight(p.s0().value(), ref["s0"])
    tight(p.dividend_yield().discount(2.0), ref["dividend_yield_discount_2y"])
    tight(p.risk_free_rate().discount(2.0), ref["risk_free_rate_discount_2y"])
    tight(p.leverage_fct().local_vol_at_time(0.5, 100.0, True), ref["leverage_t0_5_k100"])
    tight(p.leverage_fct().local_vol_at_time(1.25, 95.0, True), ref["leverage_t1_25_k95"])
    _assert_array(p.initial_values(), ref["initial_values"])
    _assert_array(p.apply(_arr([108.0, 0.0512]), _arr([0.031, -0.0044])), ref["apply"])
    tight(p.time(TODAY + 365), ref["time_1y"])


def test_heston_slv_drift_and_diffusion_match_cpp(cpp: dict[str, Any]) -> None:
    ref = cpp["heston_slv"]
    p = _slv()
    _assert_array(p.drift(0.5, _arr([100.0, 0.0625])), ref["drift_t0_5"])
    _assert_array(p.drift(1.25, _arr([95.0, 0.0410])), ref["drift_t1_25"])
    _assert_matrix(p.diffusion(0.5, _arr([100.0, 0.0625])), ref["diffusion_t0_5"])
    _assert_matrix(p.diffusion(1.25, _arr([95.0, 0.0410])), ref["diffusion_t1_25"])


def test_heston_slv_mixing_factor_is_threaded(cpp: dict[str, Any]) -> None:
    """mixingFactor scales the vol-of-vol only; the leveraged spot vol is unchanged."""
    ref = cpp["heston_slv"]
    default = HestonSLVProcess(_heston(), _leverage_surface())
    _assert_matrix(default.diffusion(0.5, _arr([100.0, 0.0625])), ref["default_diffusion_t0_5"])
    # Row 0 (spot) identical, row 1 (variance) scaled: proves the argument is
    # applied where C++ applies it and nowhere else.
    exact(ref["default_diffusion_t0_5"][0][0], ref["diffusion_t0_5"][0][0])
    assert ref["default_diffusion_t0_5"][1][0] != ref["diffusion_t0_5"][1][0]


def test_heston_slv_evolve_covers_all_three_branches(cpp: dict[str, Any]) -> None:
    """psi < 1.5; psi >= 1.5 with u > p; psi >= 1.5 with u <= p (variance == 0)."""
    ref = cpp["heston_slv"]
    assert ref["psi_low"] < 1.5
    assert ref["psi_high"] >= 1.5
    p = _slv()
    _assert_array(
        p.evolve(0.5, _arr([100.0, 0.25]), 0.05, _arr([0.63, -0.42])), ref["evolve_psi_low"]
    )
    _assert_array(
        p.evolve(0.5, _arr([100.0, 0.001]), 3.0, _arr([0.63, 0.85])), ref["evolve_psi_high"]
    )
    zero_branch = p.evolve(0.5, _arr([100.0, 0.001]), 3.0, _arr([0.63, -1.85]))
    _assert_array(zero_branch, ref["evolve_psi_high_zero"])
    # The u <= p arm sets the variance to EXACTLY zero.
    exact(float(zero_branch[1]), 0.0)

    default = HestonSLVProcess(_heston(), _leverage_surface())
    _assert_array(
        default.evolve(0.5, _arr([100.0, 0.25]), 0.05, _arr([0.63, -0.42])),
        ref["default_evolve_psi_low"],
    )
    _assert_array(
        default.evolve(0.5, _arr([100.0, 0.001]), 3.0, _arr([0.63, 0.85])),
        ref["default_evolve_psi_high"],
    )


def test_heston_slv_update_refreshes_the_cached_parameters() -> None:
    """``setParameters`` runs again on update, so a relinked Heston propagates."""
    heston = _heston()
    p = HestonSLVProcess(heston, _leverage_surface(), 0.75)
    tight(p.sigma(), 0.62)
    # Stands in for relinking a C++ Handle: HestonSLVProcess must re-read the
    # Heston parameters on update(), not cache them at construction.
    heston._sigma = 0.5  # pyright: ignore[reportPrivateUsage]
    p.update()
    tight(p.sigma(), 0.5)
    tight(p.diffusion(0.5, _arr([100.0, 0.0625]))[1][1], 0.75 * 0.5 * math.sqrt(0.0625)
          * math.sqrt(1.0 - 0.4 * 0.4))


# =============================================================================
# JointStochasticProcess + CachingKey
# =============================================================================


class _ProbeJointProcess(JointStochasticProcess):
    """The concrete subclass the probe defines (probe.cpp:249-286).

    QuantLib v1.43 ships no concrete ``JointStochasticProcess``, so both sides
    of the cross-validation supply the same minimal one.
    """

    def __init__(
        self,
        l: Sequence[StochasticProcess],  # noqa: E741 — mirrors the C++ parameter name
        factors: int | None,
        state_dependent: bool,
    ) -> None:
        super().__init__(l, factors)
        self._state_dependent: bool = state_dependent
        self.last_dv: npt.NDArray[np.float64] = np.zeros(0, dtype=np.float64)

    def pre_evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
    ) -> None:
        self.last_dv = np.array(dw, dtype=np.float64, copy=True)

    def post_evolve(
        self,
        t0: float,
        x0: npt.NDArray[np.float64],
        dt: float,
        dw: npt.NDArray[np.float64],
        y0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r = np.array(y0, dtype=np.float64, copy=True)
        r[0] += 0.125
        return r

    def numeraire(self, t: float, x: npt.NDArray[np.float64]) -> float:
        return math.exp(-float(x[len(x) - 1]) * t)

    def correlation_is_state_dependent(self) -> bool:
        return self._state_dependent

    def cross_model_correlation(
        self, t0: float, x0: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        m = np.zeros((self.size(), self.size()), dtype=np.float64)
        scale = math.tanh(float(x0[0]) / 200.0) * 4.0 if self._state_dependent else 1.0
        m[0, 2] = m[2, 0] = 0.35 * scale
        m[1, 2] = m[2, 1] = -0.20 * scale
        return m


def _joint_constituents() -> list[StochasticProcess]:
    """probe.cpp:288-302 — a 2-D StochasticProcessArray plus a HullWhiteProcess."""
    inner = [
        GeometricBrownianMotionProcess(100.0, 0.055, 0.21),
        GeometricBrownianMotionProcess(85.0, 0.032, 0.34),
    ]
    corr = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)
    return [
        StochasticProcessArray(inner, corr),
        HullWhiteProcess(_domestic_curve(), 0.07, 0.013),
    ]


_JOINT_X0: Final[list[float]] = [102.5, 88.0, 0.0215]
_JOINT_T0: Final[float] = 0.75
_JOINT_DT: Final[float] = 0.5


@pytest.mark.parametrize(
    ("ref_key", "factors", "state_dependent"),
    [
        ("joint_default_factors", None, False),
        ("joint_two_factors", 2, False),
        ("joint_state_dependent", None, True),
    ],
)
def test_joint_stochastic_process_matches_cpp(
    cpp: dict[str, Any], ref_key: str, factors: int | None, state_dependent: bool
) -> None:
    ref = cpp[ref_key]
    p = _ProbeJointProcess(_joint_constituents(), factors, state_dependent)
    x0 = _arr(_JOINT_X0)

    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    assert p.correlation_is_state_dependent() is ref["correlation_is_state_dependent"]
    _assert_array(p.initial_values(), ref["initial_values"])
    _assert_array(p.drift(_JOINT_T0, x0), ref["drift"])
    _assert_matrix(p.diffusion(_JOINT_T0, x0), ref["diffusion"])
    _assert_array(p.expectation(_JOINT_T0, x0, _JOINT_DT), ref["expectation"])
    _assert_matrix(p.covariance(_JOINT_T0, x0, _JOINT_DT), ref["covariance"])
    _assert_matrix(p.std_deviation(_JOINT_T0, x0, _JOINT_DT), ref["std_deviation"])
    _assert_matrix(p.cross_model_correlation(_JOINT_T0, x0), ref["cross_model_correlation"])
    _assert_array(p.apply(x0, _arr([0.031, -0.018, 0.0007])), ref["apply"])
    tight(p.numeraire(_JOINT_T0, x0), ref["numeraire"])

    # Two evolve() calls at the same (t0, dt): on the non-state-dependent
    # instances the second one takes the correlationCache_ hit path.
    first = p.evolve(_JOINT_T0, x0, _JOINT_DT, _arr(ref["dw1"]))
    _assert_array(p.last_dv, ref["pre_evolve_dv1"])
    _assert_array(first, ref["evolve1"])
    second = p.evolve(_JOINT_T0, x0, _JOINT_DT, _arr(ref["dw2"]))
    _assert_array(p.last_dv, ref["pre_evolve_dv2"])
    _assert_array(second, ref["evolve2"])


def test_joint_stochastic_process_factors_argument_is_honoured(cpp: dict[str, Any]) -> None:
    """``factors=None`` means Null<Size>() -> modelFactors; ``2`` overrides it."""
    assert cpp["joint_default_factors"]["factors"] == 3
    assert cpp["joint_two_factors"]["factors"] == 2
    with pytest.raises(LibraryException) as too_many:
        _ProbeJointProcess(_joint_constituents(), 4, False)
    assert str(too_many.value) == cpp["errors"]["joint_too_many_factors"]


def test_joint_stochastic_process_caches_only_when_state_independent() -> None:
    """The cache is keyed on (t0, dt) and cleared by update()."""
    p = _ProbeJointProcess(_joint_constituents(), None, False)
    x0 = _arr(_JOINT_X0)
    dw = _arr([0.63, -1.21, 0.47])
    assert len(p._correlation_cache) == 0  # pyright: ignore[reportPrivateUsage]
    p.evolve(_JOINT_T0, x0, _JOINT_DT, dw)
    cached_keys = list(p._correlation_cache)  # pyright: ignore[reportPrivateUsage]
    assert cached_keys == [CachingKey(_JOINT_T0, _JOINT_DT)]
    p.evolve(_JOINT_T0, x0, _JOINT_DT, dw)
    assert len(p._correlation_cache) == 1  # pyright: ignore[reportPrivateUsage]
    p.evolve(1.25, x0, _JOINT_DT, dw)
    assert len(p._correlation_cache) == 2  # pyright: ignore[reportPrivateUsage]
    p.update()
    assert len(p._correlation_cache) == 0  # pyright: ignore[reportPrivateUsage]

    state_dep = _ProbeJointProcess(_joint_constituents(), None, True)
    state_dep.evolve(_JOINT_T0, x0, _JOINT_DT, dw)
    assert len(state_dep._correlation_cache) == 0  # pyright: ignore[reportPrivateUsage]


def test_joint_stochastic_process_post_evolve_is_applied(cpp: dict[str, Any]) -> None:
    """The probe's postEvolve adds 0.125 to component 0; a dropped hook shows here."""
    ref = cpp["joint_default_factors"]
    constituents = _joint_constituents()
    p = _ProbeJointProcess(constituents, None, False)
    x0 = _arr(_JOINT_X0)
    evolved = p.evolve(_JOINT_T0, x0, _JOINT_DT, _arr(ref["dw1"]))
    # Reconstruct the pre-postEvolve value from the constituents directly.
    dv = _arr(ref["pre_evolve_dv1"])
    raw0 = constituents[0].evolve(
        _JOINT_T0, _arr(_JOINT_X0[:2]), _JOINT_DT, _arr([dv[0], dv[1]])
    )
    tight(float(evolved[0]), float(raw0[0]) + 0.125)


def test_joint_stochastic_process_time_delegates_to_the_first_constituent(
    cpp: dict[str, Any],
) -> None:
    """jointstochasticprocess.cpp:298-302 — ``l_[0]->time(date)``, no fallback."""
    ref = cpp["joint_time"]
    l: list[StochasticProcess] = [  # noqa: E741 — mirrors the C++ parameter name
        _gk(),
        HullWhiteProcess(_domestic_curve(), 0.07, 0.013),
    ]
    p = _ProbeJointProcess(l, None, False)
    assert p.size() == ref["size"]
    assert p.factors() == ref["factors"]
    tight(p.time(TODAY + 365), ref["time_1y"])
    tight(p.time(TODAY + 1095), ref["time_3y"])

    # With a head that has no daycounter the delegation raises, exactly as C++
    # does (cpp["joint_default_factors"]["time_raises"] records that).
    assert cpp["joint_default_factors"]["time_raises"] is True
    headless = _ProbeJointProcess(_joint_constituents(), None, False)
    with pytest.raises(LibraryException, match="date/time conversion not supported"):
        headless.time(TODAY + 365)


def test_joint_stochastic_process_constituents_are_returned_in_order() -> None:
    constituents = _joint_constituents()
    p = _ProbeJointProcess(constituents, None, False)
    assert p.constituents() == constituents


def test_caching_key_ordering_matches_the_cpp_comparator() -> None:
    """jointstochasticprocess.hpp:88-91 — by t0 first, then dt."""
    assert CachingKey(1.0, 0.5) < CachingKey(2.0, 0.1)
    assert CachingKey(1.0, 0.5) < CachingKey(1.0, 0.9)
    assert not CachingKey(1.0, 0.5) < CachingKey(1.0, 0.5)
    assert not CachingKey(2.0, 0.1) < CachingKey(1.0, 0.5)
    # Hashability is what makes it usable as a Python dict key.
    assert {CachingKey(1.0, 0.5): 1}[CachingKey(1.0, 0.5)] == 1
    assert CachingKey(1.0, 0.5) == CachingKey(1.0, 0.5)
    assert CachingKey(1.0, 0.5) != CachingKey(1.0, 0.6)


# =============================================================================
# Cross-cutting: the classes are exported under their C++ names
# =============================================================================


def test_new_classes_are_exported_from_the_package() -> None:
    """The coverage gate matches on the C++ symbol name; keep them reachable."""
    import pquantlib.processes as procs  # noqa: PLC0415

    for name in (
        "EndEulerDiscretization",
        "GarmanKohlagenProcess",
        "GeometricBrownianMotionProcess",
        "HestonSLVProcess",
        "HullWhiteProcess",
        "HybridHestonHullWhiteProcess",
        "JointStochasticProcess",
        "CachingKey",
        "Merton76Process",
        "MfStateProcess",
    ):
        assert hasattr(procs, name), name
        assert name in procs.__all__, name

