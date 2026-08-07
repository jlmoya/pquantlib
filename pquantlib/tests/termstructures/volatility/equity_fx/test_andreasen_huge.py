"""Andreasen-Huge volatility interpolation + adapters, cross-validated vs v1.43.

References: ``migration-harness/references/v143/ts/andreasenhuge.json``,
emitted verbatim by ``migration-harness/cpp/probes/v143_ts_andreasenhuge/probe.cpp``.

The surface is a ``LazyObject`` whose ``performCalculations`` runs a per-expiry
least-squares calibration, so the reference is emitted in two layers and tested
at two tiers:

**Frozen layer** (``noopt`` / ``noopt_adapters``) — a stub optimiser stamps a
fixed, spelled-out sigma vector onto the ``Problem`` instead of searching, so
nothing downstream depends on an optimiser. This is the layer that must be
TIGHT: the mesher, the first/second-derivative operators, the tridiagonal
splitting solve, all three sigma interpolations and the Dupire ratio. The
sigma vector is deliberately non-flat — a flat one would make
PiecewiseConstant, Linear and CubicSpline agree and leave the interpolation
branch untested (probe.cpp:76-90).

**LM layer** (``lm`` / ``lm_adapters``) — the same quantities after a real
Levenberg-Marquardt calibration. C++ runs MINPACK ``lmdif``; this port
delegates to ``scipy.optimize.least_squares(method='lm')``, whose iterates are
not reproducible term by term (see
``pquantlib/tests/math/optimization/test_levenberg_marquardt_cpp_parity.py``).
Tested at LOOSE. In practice it does far better than that — the observed worst
relative gap over the whole LM block is 7e-12 — because for the exactly
determined slices (5 sigmas, 5 residuals) the least-squares problem has a
unique root that both implementations reach; but the tier reflects what is
*guaranteed*, not what today's SciPy happens to do.

Tolerance exceptions, both tighter than LOOSE, with their arithmetic:

* ``put_price`` under ``CalibrationType.Call``/``CallPut`` is reconstructed by
  put-call parity, ``price + strike/fwd - 1`` (interpl cpp:520-522). At
  ``t=0.10, K=70, fwd~100`` that is ``0.3009 - 0.2991``: two ~0.3 numbers
  cancelling to ~1e-4, so a 1-ulp error in the call term (~3e-17) is amplified
  by ~3e3 and then rescaled by ``df*fwd ~ 100``. Observed worst 2.2e-14
  absolute on a 0.011 value, i.e. 2.0e-12 relative — just past TIGHT's 1e-12
  and three orders inside the 1e-11 used here.
* ``local_vol`` is ``sqrt(2 dC/dT / d2C/dK2)``, a ratio of two FDM operators
  applied to a solve output; on the 150-point grid the numerator and
  denominator are each ~1e-5, so the quotient carries ~1e-12 relative noise.
  Observed worst 3.5e-12 relative.

The probe pins ``Settings::instance().evaluationDate()`` (probe.cpp:227); the
fixture below does the same and restores the previous value.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.end_criteria import Type as EndCriteriaType
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.andreasen_huge_local_vol_adapter import (
    AndreasenHugeLocalVolAdapter,
)
from pquantlib.termstructures.volatility.equity_fx.andreasen_huge_volatility_adapter import (
    AndreasenHugeVolatilityAdapter,
    black_formula_implied_std_dev_approximation_rs,
    black_formula_implied_std_dev_li_rs,
)
from pquantlib.termstructures.volatility.equity_fx.andreasen_huge_volatility_interpl import (
    AndreasenHugeVolatilityInterpl,
    CalibrationType,
    InterpolationType,
    SingleStepCalibrationResult,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date

_REF: dict[str, Any] = reference_reader.load("v143/ts/andreasenhuge")
_SETUP: dict[str, Any] = _REF["setup"]

_REF_DATE = Date(_SETUP["reference_date_serial"])
_EXPIRIES = [Date(s) for s in _SETUP["expiry_date_serials"]]
_CAL_STRIKES: list[float] = _SETUP["calibration_strikes"]
_N_STRIKES = len(_CAL_STRIKES)
_CAL_VOLS: list[list[float]] = [
    _SETUP["calibration_vols_row_major"][i * _N_STRIKES : (i + 1) * _N_STRIKES]
    for i in range(len(_EXPIRIES))
]
_SPOT: float = _SETUP["spot"]
_R: float = _SETUP["risk_free_rate"]
_Q: float = _SETUP["dividend_rate"]
_N_GRID: int = _SETUP["n_grid_points"]
_QUERY_TIMES: list[float] = _SETUP["query_times"]
_QUERY_STRIKES: list[float] = _SETUP["query_strikes"]

# Tighter than LOOSE; the derivations are in the module docstring.
_CANCELLATION_ABS = 1e-11
_CANCELLATION_REL = 1e-11
_CANCELLATION_REASON = (
    "put-call parity cancellation (interpl cpp:520-522) / Dupire operator ratio; "
    "see module docstring for the arithmetic"
)


@pytest.fixture(autouse=True)
def _pin_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin Settings to the probe's evaluation date (probe.cpp:227)."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _REF_DATE
    yield
    settings.evaluation_date = previous


class FrozenOptimizationMethod(OptimizationMethod):
    """Stamps the probe's fixed sigma vector; runs no search.

    # C++ parity: ``FrozenOptimizationMethod`` in
    # migration-harness/cpp/probes/v143_ts_andreasenhuge/probe.cpp:78-90.
    """

    @staticmethod
    def sigma(i: int) -> float:
        return 0.30 - 0.02 * float(i) + 0.01 * float(i % 2)

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> EndCriteriaType:
        del end_criteria
        x: npt.NDArray[np.float64] = np.asarray(problem.current_value, dtype=np.float64).copy()
        for i in range(x.size):
            x[i] = self.sigma(i)
        problem.set_current_value(x)
        return EndCriteriaType.None_


def _calibration_set() -> list[tuple[VanillaOption, Quote]]:
    out: list[tuple[VanillaOption, Quote]] = []
    for i, expiry in enumerate(_EXPIRIES):
        exercise = EuropeanExercise(expiry)
        for j, strike in enumerate(_CAL_STRIKES):
            out.append(
                (
                    VanillaOption(PlainVanillaPayoff(OptionType.Call, strike), exercise),
                    SimpleQuote(_CAL_VOLS[i][j]),
                )
            )
    return out


def _interpl(
    interpolation_type: InterpolationType = InterpolationType.CubicSpline,
    calibration_type: CalibrationType = CalibrationType.Call,
    *,
    frozen: bool = True,
    n_grid_points: int | None = None,
    calibration_set: list[tuple[VanillaOption, Quote]] | None = None,
    min_strike: float | None = None,
    max_strike: float | None = None,
) -> AndreasenHugeVolatilityInterpl:
    return AndreasenHugeVolatilityInterpl(
        calibration_set if calibration_set is not None else _calibration_set(),
        SimpleQuote(_SPOT),
        FlatForward.from_rate(_REF_DATE, _R, Actual365Fixed()),
        FlatForward.from_rate(_REF_DATE, _Q, Actual365Fixed()),
        interpolation_type,
        calibration_type,
        n_grid_points if n_grid_points is not None else _N_GRID,
        min_strike,
        max_strike,
        FrozenOptimizationMethod() if frozen else None,
        EndCriteria(500, 100, 1e-12, 1e-10, 1e-10),
    )


_NOOPT_CASES: dict[str, dict[str, Any]] = {
    "cubic_call": {},
    "cubic_put": {"calibration_type": CalibrationType.Put},
    "cubic_callput": {"calibration_type": CalibrationType.CallPut},
    "linear_call": {"interpolation_type": InterpolationType.Linear},
    "piecewise_constant_call": {"interpolation_type": InterpolationType.PiecewiseConstant},
    "cubic_call_150pts": {"n_grid_points": 150},
}

_LM_CASES: dict[str, dict[str, Any]] = {
    "cubic_call": {},
    "cubic_callput": {"calibration_type": CalibrationType.CallPut},
    "piecewise_constant_call": {"interpolation_type": InterpolationType.PiecewiseConstant},
}


# --------------------------------------------------------------------------
# 0. the FDM grid, one layer below the surface
# --------------------------------------------------------------------------


def test_mesher_matches_cpp() -> None:
    """The Concentrating1dMesher the surface builds (interpl cpp:353-359)."""
    expected = _SETUP["mesher_locations"]
    mesher = Concentrating1dMesher(
        math.log(_SETUP["default_min_strike"] / _SPOT),
        math.log(_SETUP["default_max_strike"] / _SPOT),
        _N_GRID,
        (0.0, 0.025),
    )
    locations = mesher.locations()
    assert len(locations) == len(expected)
    for got, exp in zip(locations, expected, strict=True):
        tolerance.tight(float(got), exp)


def test_default_strike_bounds() -> None:
    """``Null<Real>()`` bounds resolve to strikes[0]/8 and 8*strikes[-1]."""
    ah = _interpl()
    tolerance.exact(ah.min_strike(), _SETUP["default_min_strike"])
    tolerance.exact(ah.max_strike(), _SETUP["default_max_strike"])


def test_explicit_strike_bounds_override_the_defaults() -> None:
    ah = _interpl(min_strike=5.0, max_strike=500.0)
    tolerance.exact(ah.min_strike(), 5.0)
    tolerance.exact(ah.max_strike(), 500.0)


def test_frozen_sigmas_match_the_probe() -> None:
    """The stub optimiser's vector, as the probe emits it."""
    expected = _SETUP["frozen_sigmas"]
    for i, exp in enumerate(expected):
        tolerance.exact(FrozenOptimizationMethod.sigma(i), exp)


# --------------------------------------------------------------------------
# 1. blackFormulaImpliedStdDevLiRS, the adapter's inversion
# --------------------------------------------------------------------------


def test_li_rs_approximation_rs_matches_cpp() -> None:
    """The closed-form RS seed (blackformula.cpp:269-318)."""
    li = _REF["li_rs"]
    for i in range(len(li["input_forward"])):
        forward, strike = li["input_forward"][i], li["input_strike"][i]
        option_type = OptionType.Put if forward > strike else OptionType.Call
        tolerance.tight(
            black_formula_implied_std_dev_approximation_rs(
                option_type, strike, forward, li["input_price"][i], li["input_discount"][i], 0.0
            ),
            li["approximation_rs"][i],
        )


def test_li_rs_iteration_matches_cpp() -> None:
    """The Li rational-search fixed point (blackformula.cpp:483-543)."""
    li = _REF["li_rs"]
    for i in range(len(li["input_forward"])):
        forward, strike = li["input_forward"][i], li["input_strike"][i]
        option_type = OptionType.Put if forward > strike else OptionType.Call
        tolerance.tight(
            black_formula_implied_std_dev_li_rs(
                option_type,
                strike,
                forward,
                li["input_price"][i],
                li["input_discount"][i],
                0.0,
                None,
                1.0,
                1e-6,
                1000,
            ),
            li["li_rs_std_dev"][i],
        )


def test_li_rs_input_prices_match_cpp() -> None:
    """The Black prices the inversion is fed — pinned so a mismatch is not
    mistaken for an inversion error."""
    li = _REF["li_rs"]
    for i in range(len(li["input_forward"])):
        forward, strike = li["input_forward"][i], li["input_strike"][i]
        option_type = OptionType.Put if forward > strike else OptionType.Call
        tolerance.tight(
            black_formula(
                option_type, strike, forward, li["input_std_dev"][i], li["input_discount"][i]
            ),
            li["input_price"][i],
        )


def test_li_rs_recovers_the_input_std_dev() -> None:
    """Round trip: the inversion returns what generated the price, to ~1e-6.

    The accuracy argument is the iteration's own stopping criterion, so the
    round-trip error is bounded by it, not by float precision — C++'s own
    residuals are emitted alongside and agree.
    """
    li = _REF["li_rs"]
    for i in range(len(li["input_std_dev"])):
        assert abs(li["li_rs_minus_input"][i]) < 1e-6
        forward, strike = li["input_forward"][i], li["input_strike"][i]
        option_type = OptionType.Put if forward > strike else OptionType.Call
        got = black_formula_implied_std_dev_li_rs(
            option_type,
            strike,
            forward,
            li["input_price"][i],
            li["input_discount"][i],
            0.0,
            None,
            1.0,
            1e-6,
            1000,
        )
        tolerance.tight(got - li["input_std_dev"][i], li["li_rs_minus_input"][i])


def test_li_rs_deep_out_of_the_money_fails_in_both_languages() -> None:
    """Documented divergence: same breakdown, different exception.

    The RS seed's ``C`` term is a difference of two nearly equal squares
    (blackformula.cpp:293); at a ~1e-16 price it rounds negative, ``log(beta)``
    is NaN, and the iteration never converges. C++ surfaces that as a
    ``boost::math`` ``domain_error`` thrown by the Maddock quantile — the exact
    text is pinned in the reference. SciPy's ``ndtri`` returns NaN instead of
    throwing, so this port reaches the iteration's own
    ``QL_REQUIRE(dv <= accuracy)`` and reports "max iterations exceeded".
    Both refuse; only the message differs, and neither returns a number.
    """
    assert "boost::math::quantile" in _REF["throws"]["li_rs_deep_otm_breaks_down"]
    # The seed itself does not throw in either language (C++ records "").
    assert _REF["throws"]["li_rs_approximation_rs_deep_otm"] == ""
    price = black_formula(OptionType.Put, 70.0, 105.0, 0.05, 1.0)
    assert math.isnan(
        black_formula_implied_std_dev_approximation_rs(
            OptionType.Put, 70.0, 105.0, price, 1.0, 0.0
        )
    )
    with pytest.raises(LibraryException, match="max iterations exceeded"):
        black_formula_implied_std_dev_li_rs(
            OptionType.Put, 70.0, 105.0, price, 1.0, 0.0, None, 1.0, 1e-6, 1000
        )


def test_li_rs_rejects_a_negative_guess() -> None:
    with pytest.raises(LibraryException, match="must be non-negative"):
        black_formula_implied_std_dev_li_rs(
            OptionType.Call, 100.0, 100.0, 5.0, 1.0, 0.0, -0.1, 1.0, 1e-6, 100
        )


def test_li_rs_rejects_a_non_positive_discount() -> None:
    with pytest.raises(LibraryException, match="discount"):
        black_formula_implied_std_dev_li_rs(
            OptionType.Call, 100.0, 100.0, 5.0, 0.0, 0.0, None, 1.0, 1e-6, 100
        )


def test_li_rs_honours_an_explicit_guess() -> None:
    """A supplied guess bypasses the RS seed and reaches the same root."""
    li = _REF["li_rs"]
    for i in range(0, len(li["input_forward"]), 7):
        forward, strike = li["input_forward"][i], li["input_strike"][i]
        option_type = OptionType.Put if forward > strike else OptionType.Call
        got = black_formula_implied_std_dev_li_rs(
            option_type,
            strike,
            forward,
            li["input_price"][i],
            li["input_discount"][i],
            0.0,
            0.5,
            1.0,
            1e-10,
            1000,
        )
        tolerance.custom(
            got,
            li["li_rs_std_dev"][i],
            abs_tol=1e-6,
            rel_tol=1e-5,
            reason=(
                "different seed, same root: both runs stop on their own "
                "|v_{k+1}-v_k| criterion (1e-10 here, 1e-6 for the reference), "
                "so the two answers differ by at most the looser of the two"
            ),
        )


# --------------------------------------------------------------------------
# 2. the frozen (optimiser-independent) layer — TIGHT
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_NOOPT_CASES))
def test_frozen_calibration_error(case: str) -> None:
    expected = _REF["noopt"][case]
    ah = _interpl(frozen=True, **_NOOPT_CASES[case])
    min_error, max_error, avg_error = ah.calibration_error()
    tolerance.tight(min_error, expected["calibration_min_error"])
    tolerance.tight(max_error, expected["calibration_max_error"])
    tolerance.tight(avg_error, expected["calibration_avg_error"])


@pytest.mark.parametrize("case", sorted(_NOOPT_CASES))
def test_frozen_forward(case: str) -> None:
    expected = _REF["noopt"][case]["fwd"]
    ah = _interpl(frozen=True, **_NOOPT_CASES[case])
    for i, t in enumerate(_QUERY_TIMES):
        tolerance.tight(ah.fwd(t), expected[i])


@pytest.mark.parametrize("case", sorted(_NOOPT_CASES))
def test_frozen_call_price(case: str) -> None:
    expected = _REF["noopt"][case]["call_price"]
    ah = _interpl(frozen=True, **_NOOPT_CASES[case])
    i = 0
    for t in _QUERY_TIMES:
        for k in _QUERY_STRIKES:
            tolerance.tight(ah.option_price(t, k, OptionType.Call), expected[i])
            i += 1


@pytest.mark.parametrize("case", sorted(_NOOPT_CASES))
def test_frozen_put_price(case: str) -> None:
    expected = _REF["noopt"][case]["put_price"]
    ah = _interpl(frozen=True, **_NOOPT_CASES[case])
    i = 0
    for t in _QUERY_TIMES:
        for k in _QUERY_STRIKES:
            tolerance.custom(
                ah.option_price(t, k, OptionType.Put),
                expected[i],
                abs_tol=_CANCELLATION_ABS,
                rel_tol=_CANCELLATION_REL,
                reason=_CANCELLATION_REASON,
            )
            i += 1


@pytest.mark.parametrize("case", sorted(_NOOPT_CASES))
def test_frozen_local_vol(case: str) -> None:
    expected = _REF["noopt"][case]["local_vol"]
    ah = _interpl(frozen=True, **_NOOPT_CASES[case])
    i = 0
    for t in _QUERY_TIMES:
        for k in _QUERY_STRIKES:
            tolerance.custom(
                ah.local_vol(t, k),
                expected[i],
                abs_tol=_CANCELLATION_ABS,
                rel_tol=_CANCELLATION_REL,
                reason=_CANCELLATION_REASON,
            )
            i += 1


@pytest.mark.parametrize("case", sorted(_NOOPT_CASES))
def test_frozen_max_date_and_strike_range(case: str) -> None:
    expected = _REF["noopt"][case]
    ah = _interpl(frozen=True, **_NOOPT_CASES[case])
    assert ah.max_date().serial_number() == expected["max_date_serial"]
    tolerance.exact(ah.min_strike(), expected["min_strike"])
    tolerance.exact(ah.max_strike(), expected["max_strike"])


def test_interpolation_type_actually_changes_the_answer() -> None:
    """Guard against a port that ignores InterpolationType.

    With the frozen sigma vector non-flat, the three interpolations must
    disagree; if a port hard-wired one of them this test is what catches it.
    """
    cubic = _REF["noopt"]["cubic_call"]["call_price"]
    linear = _REF["noopt"]["linear_call"]["call_price"]
    piecewise = _REF["noopt"]["piecewise_constant_call"]["call_price"]
    assert cubic != linear
    assert cubic != piecewise
    assert linear != piecewise


# --------------------------------------------------------------------------
# 3. the frozen adapters — TIGHT
# --------------------------------------------------------------------------

_ADAPTER_CASES: dict[str, dict[str, Any]] = {
    "cubic_call": {},
    "cubic_put": {"calibration_type": CalibrationType.Put},
    "cubic_callput": {"calibration_type": CalibrationType.CallPut},
}


@pytest.mark.parametrize("case", sorted(_ADAPTER_CASES))
def test_frozen_volatility_adapter(case: str) -> None:
    """``AndreasenHugeVolatilityAdapter``: price -> LiRS -> variance."""
    expected = _REF["noopt_adapters"][case]
    ah = _interpl(frozen=True, **_ADAPTER_CASES[case])
    adapter = AndreasenHugeVolatilityAdapter(ah)
    i = 0
    for j, t in enumerate(_QUERY_TIMES):
        tolerance.tight(adapter.atm_level(t), expected["vol_adapter_atm_level"][j])
        for k in _QUERY_STRIKES:
            tolerance.tight(
                adapter.black_variance_at_time(t, k, True),
                expected["vol_adapter_black_variance"][i],
            )
            tolerance.tight(
                adapter.black_vol_at_time(t, k, True), expected["vol_adapter_black_vol"][i]
            )
            i += 1


@pytest.mark.parametrize("case", sorted(_ADAPTER_CASES))
def test_frozen_local_vol_adapter(case: str) -> None:
    """``AndreasenHugeLocalVolAdapter``: the clamped local-vol pass-through."""
    expected = _REF["noopt_adapters"][case]["local_vol_adapter_local_vol"]
    ah = _interpl(frozen=True, **_ADAPTER_CASES[case])
    adapter = AndreasenHugeLocalVolAdapter(ah)
    i = 0
    for t in _QUERY_TIMES:
        for k in _QUERY_STRIKES:
            tolerance.custom(
                adapter.local_vol_at_time(t, k, True),
                expected[i],
                abs_tol=_CANCELLATION_ABS,
                rel_tol=_CANCELLATION_REL,
                reason=_CANCELLATION_REASON,
            )
            i += 1


def test_adapter_inspectors_delegate_to_the_risk_free_curve() -> None:
    """Both adapters read their metadata off ``volInterpl_->riskFreeRate()``."""
    expected = _REF["noopt_adapters"]["cubic_call"]
    ah = _interpl(frozen=True)
    vol_adapter = AndreasenHugeVolatilityAdapter(ah)
    lv_adapter = AndreasenHugeLocalVolAdapter(ah)

    assert vol_adapter.max_date().serial_number() == expected["vol_adapter_max_date_serial"]
    tolerance.exact(vol_adapter.min_strike(), expected["vol_adapter_min_strike"])
    tolerance.exact(vol_adapter.max_strike(), expected["vol_adapter_max_strike"])
    assert vol_adapter.day_counter().name() == expected["vol_adapter_day_counter"]
    assert (
        vol_adapter.reference_date().serial_number()
        == expected["vol_adapter_reference_date_serial"]
    )

    assert (
        lv_adapter.max_date().serial_number() == expected["local_vol_adapter_max_date_serial"]
    )
    tolerance.exact(lv_adapter.min_strike(), expected["local_vol_adapter_min_strike"])
    tolerance.exact(lv_adapter.max_strike(), expected["local_vol_adapter_max_strike"])
    assert lv_adapter.day_counter().name() == expected["local_vol_adapter_day_counter"]
    assert (
        lv_adapter.reference_date().serial_number()
        == expected["local_vol_adapter_reference_date_serial"]
    )


def test_adapter_calendar_and_settlement_days_throw_like_cpp() -> None:
    """The delegation faithfully forwards the curve's *absence* of metadata.

    The reference curves are ``FlatForward``s on an explicit reference date, so
    C++ leaves ``calendar_`` empty and ``settlementDays_`` Null and both
    accessors throw. This port raises instead of inventing a NullCalendar,
    which would silently change what the adapter reports.
    """
    expected = _REF["noopt_adapters"]["cubic_call"]
    assert expected["vol_adapter_calendar_empty"] is True
    assert expected["local_vol_adapter_calendar_empty"] is True
    assert "settlement days not provided" in expected["vol_adapter_settlement_days_error"]
    assert "settlement days not provided" in expected["local_vol_adapter_settlement_days_error"]

    ah = _interpl(frozen=True)
    for adapter in (AndreasenHugeVolatilityAdapter(ah), AndreasenHugeLocalVolAdapter(ah)):
        with pytest.raises(LibraryException, match="calendar not provided"):
            adapter.calendar()
        with pytest.raises(LibraryException, match="settlement days not provided"):
            adapter.settlement_days()


def test_local_vol_adapter_clamps_the_strike() -> None:
    """Outside the interpolation's range the edge value is returned (cpp:46-48).

    The adapter's own ``max_strike`` is ``QL_MAX_REAL``, deliberately wider
    than the interpolation's 960, so the clamp is what bounds the query.
    """
    ah = _interpl(frozen=True)
    adapter = AndreasenHugeLocalVolAdapter(ah)
    t = _QUERY_TIMES[3]
    tolerance.exact(
        adapter.local_vol_at_time(t, 1.0e6, True),
        ah.local_vol(t, ah.max_strike()),
    )
    tolerance.exact(
        adapter.local_vol_at_time(t, 1.0e-6, True),
        ah.local_vol(t, ah.min_strike()),
    )


# --------------------------------------------------------------------------
# 4. the Levenberg-Marquardt layer — LOOSE
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_LM_CASES))
def test_lm_calibration_error(case: str) -> None:
    """Calibration quality after a real LM run.

    For the exactly determined slices C++ drives the residual to ~1e-15 and so
    does SciPy; the LOOSE absolute tolerance is what makes comparing two
    machine-noise numbers meaningful.
    """
    expected = _REF["lm"][case]
    ah = _interpl(frozen=False, **_LM_CASES[case])
    min_error, max_error, avg_error = ah.calibration_error()
    tolerance.loose(min_error, expected["calibration_min_error"])
    tolerance.loose(max_error, expected["calibration_max_error"])
    tolerance.loose(avg_error, expected["calibration_avg_error"])


@pytest.mark.parametrize("case", sorted(_LM_CASES))
def test_lm_prices_and_local_vol(case: str) -> None:
    expected = _REF["lm"][case]
    ah = _interpl(frozen=False, **_LM_CASES[case])
    i = 0
    for j, t in enumerate(_QUERY_TIMES):
        tolerance.tight(ah.fwd(t), expected["fwd"][j])
        for k in _QUERY_STRIKES:
            tolerance.loose(ah.option_price(t, k, OptionType.Call), expected["call_price"][i])
            tolerance.loose(ah.option_price(t, k, OptionType.Put), expected["put_price"][i])
            tolerance.loose(ah.local_vol(t, k), expected["local_vol"][i])
            i += 1


def test_lm_adapters() -> None:
    expected = _REF["lm_adapters"]["cubic_call"]
    ah = _interpl(frozen=False)
    vol_adapter = AndreasenHugeVolatilityAdapter(ah)
    lv_adapter = AndreasenHugeLocalVolAdapter(ah)
    i = 0
    for j, t in enumerate(_QUERY_TIMES):
        tolerance.loose(vol_adapter.atm_level(t), expected["vol_adapter_atm_level"][j])
        for k in _QUERY_STRIKES:
            tolerance.loose(
                vol_adapter.black_variance_at_time(t, k, True),
                expected["vol_adapter_black_variance"][i],
            )
            tolerance.loose(
                vol_adapter.black_vol_at_time(t, k, True),
                expected["vol_adapter_black_vol"][i],
            )
            tolerance.loose(
                lv_adapter.local_vol_at_time(t, k, True),
                expected["local_vol_adapter_local_vol"][i],
            )
            i += 1


def test_lm_is_the_default_optimizer() -> None:
    """Omitting the optimiser gives LevenbergMarquardt, as C++ does."""
    expected = _REF["lm"]["cubic_call"]
    ah = AndreasenHugeVolatilityInterpl(
        _calibration_set(),
        SimpleQuote(_SPOT),
        FlatForward.from_rate(_REF_DATE, _R, Actual365Fixed()),
        FlatForward.from_rate(_REF_DATE, _Q, Actual365Fixed()),
        InterpolationType.CubicSpline,
        CalibrationType.Call,
        _N_GRID,
    )
    tolerance.loose(
        ah.option_price(_QUERY_TIMES[4], _QUERY_STRIKES[2], OptionType.Call),
        expected["call_price"][4 * len(_QUERY_STRIKES) + 2],
    )


# --------------------------------------------------------------------------
# 5. structure: SingleStepCalibrationResult
# --------------------------------------------------------------------------


def test_single_step_calibration_result_holds_the_slice_state() -> None:
    """One result per expiry, carrying exactly what replays the slice.

    # C++ parity: ``SingleStepCalibrationResult``
    # (andreasenhugevolatilityinterpl.hpp:104-107) — the four fields
    # ``getPriceSlice`` / ``getLocalVolSlice`` read (cpp:501-505, 563-573).
    """
    ah = _interpl(frozen=True)
    ah.calculate()
    results = ah._calibration_results  # pyright: ignore[reportPrivateUsage]
    assert len(results) == len(_EXPIRIES)
    for result in results:
        assert isinstance(result, SingleStepCalibrationResult)
        assert result.put_npvs.size == _N_GRID
        assert result.call_npvs.size == _N_GRID
        assert result.sigmas.size == _N_STRIKES
        for i in range(_N_STRIKES):
            tolerance.exact(float(result.sigmas[i]), FrozenOptimizationMethod.sigma(i))
        # CalibrationType.Call stores the CALL cost function (cpp:400).
        assert result.cost_function is not None


def test_call_put_calibration_stores_the_put_cost_function() -> None:
    """# C++ parity: cpp:398-401 — anything but ``Call`` stores the put side.

    Not a typo in C++: ``solveFor`` depends only on the mesher and the sigmas,
    so ``getPriceSlice`` can drive the *put* cost function with the *call* NPV
    vector and still get call prices out. Asserted structurally — the stored
    cost function's market NPVs are the put ones under ``CallPut``, the call
    ones under ``Call`` — and then behaviourally against the reference.
    """
    call_only = _interpl(InterpolationType.CubicSpline, CalibrationType.Call)
    call_put = _interpl(InterpolationType.CubicSpline, CalibrationType.CallPut)
    put_only = _interpl(InterpolationType.CubicSpline, CalibrationType.Put)
    for ah in (call_only, call_put, put_only):
        ah.calculate()

    call_results = call_only._calibration_results  # pyright: ignore[reportPrivateUsage]
    callput_results = call_put._calibration_results  # pyright: ignore[reportPrivateUsage]
    put_results = put_only._calibration_results  # pyright: ignore[reportPrivateUsage]

    for slice_idx in range(len(_EXPIRIES)):
        stored_call = call_results[slice_idx].cost_function
        stored_callput = callput_results[slice_idx].cost_function
        stored_put = put_results[slice_idx].cost_function
        assert stored_call is not None
        assert stored_callput is not None
        assert stored_put is not None
        # CallPut keeps the put side, and its market prices are the put ones.
        assert np.allclose(
            stored_callput._market_npvs,  # pyright: ignore[reportPrivateUsage]
            stored_put._market_npvs,  # pyright: ignore[reportPrivateUsage]
            atol=0.0,
            rtol=0.0,
        )
        assert not np.allclose(
            stored_callput._market_npvs,  # pyright: ignore[reportPrivateUsage]
            stored_call._market_npvs,  # pyright: ignore[reportPrivateUsage]
        )

    # ...and the call prices it produces are still the reference's.
    expected = _REF["noopt"]["cubic_callput"]["call_price"]
    i = 0
    for t in _QUERY_TIMES:
        for k in _QUERY_STRIKES:
            tolerance.tight(call_put.option_price(t, k, OptionType.Call), expected[i])
            i += 1


# --------------------------------------------------------------------------
# 6. constructor preconditions
# --------------------------------------------------------------------------


def test_rejects_an_empty_calibration_set() -> None:
    with pytest.raises(LibraryException) as exc:
        _interpl(calibration_set=[])
    assert _REF["throws"]["empty_calibration_set"] in str(exc.value)


def test_rejects_too_few_grid_points() -> None:
    with pytest.raises(LibraryException) as exc:
        _interpl(n_grid_points=2)
    assert _REF["throws"]["too_few_grid_points"] in str(exc.value)


def test_rejects_a_non_european_exercise() -> None:
    with pytest.raises(LibraryException) as exc:
        _interpl(
            calibration_set=[
                (
                    VanillaOption(
                        PlainVanillaPayoff(OptionType.Call, 100.0),
                        AmericanExercise(_REF_DATE, _EXPIRIES[0]),
                    ),
                    SimpleQuote(0.2),
                )
            ]
        )
    assert _REF["throws"]["american_exercise_rejected"] in str(exc.value)


def test_rejects_min_strike_above_max_strike() -> None:
    """Deferred to ``performCalculations`` in C++, so it fires on first use."""
    ah = _interpl(min_strike=200.0, max_strike=50.0)
    with pytest.raises(LibraryException) as exc:
        ah.calibration_error()
    assert _REF["throws"]["min_strike_above_max_strike"] in str(exc.value)
