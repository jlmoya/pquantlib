"""Cross-validated tests for GlobalBootstrap / MultiCurveBootstrap / SimpleQuoteVariables.

# C++ parity: ql/termstructures/globalbootstrap.{hpp,cpp} and
   ql/termstructures/globalbootstrapvars.{hpp,cpp} (v1.43).

Reference values come from
``migration-harness/cpp/probes/v143_ts_globalbootstrap/probe.cpp``, run
against QuantLib v1.43 and stored verbatim at
``migration-harness/references/v143/ts/globalbootstrap.json``.

**Why the layers are tested separately.** ``GlobalBootstrap`` defaults its
optimizer to ``LevenbergMarquardt``, and pquantlib's ``LevenbergMarquardt``
is a scipy delegation rather than a port of MINPACK ``lmdif``; its own
C++-parity tests are xfailed (``tests/math/optimization/
test_levenberg_marquardt_cpp_parity.py``). So the CONVERGED curve cannot
reproduce C++ bit for bit. Every layer below the optimizer is therefore
pinned on its own at a FIXED input — ``initialize``, ``setup_cost_function``,
``set_cost_function_argument`` + ``evaluate_cost_function``, and the
``MultiCurveBootstrap`` concatenation layout — all at TIGHT, leaving the
optimizer as the single loose joint. See
``test_converged_single_curve_reproduces_cpp`` for the tolerance the
converged layer actually reaches and why.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.optimization.end_criteria import EndCriteria, Type
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.global_bootstrap import (
    GlobalBootstrap,
    MultiCurveBootstrap,
)
from pquantlib.termstructures.global_bootstrap_vars import SimpleQuoteVariables
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

REF: Final[dict[str, Any]] = reference_reader.load("v143/ts/globalbootstrap")

# probe.cpp:87 — ``const Date kToday(23, October, 2025);``
_TODAY: Final[Date] = Date.from_ymd(23, Month.October, 2025)
# probe.cpp:89 — ``TARGET().advance(kToday, 2, Days)``.
_SETTLEMENT: Final[Date] = TARGET().advance(_TODAY, 2, TimeUnit.Days)


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    s = ObservableSettings()
    prev = s.evaluation_date
    s.evaluation_date = _TODAY  # probe.cpp:87
    yield
    s.evaluation_date = prev


# ---------------------------------------------------------------------------
# fixtures mirroring probe.cpp
# ---------------------------------------------------------------------------


def _deposit(rate: float, n: int, unit: TimeUnit, fixing_days: int) -> DepositRateHelper:
    """probe.cpp:105-109 — ``deposit(rate, n, unit, fixingDays)``."""
    return DepositRateHelper(
        SimpleQuote(rate),
        tenor=Period(n, unit),
        fixing_days=fixing_days,
        calendar=TARGET(),
        convention=BusinessDayConvention.ModifiedFollowing,
        end_of_month=True,
        day_counter=Actual360(),
        evaluation_date=_TODAY,
    )


def _make_instruments() -> list[DepositRateHelper]:
    """probe.cpp:113-117. [0] is dead — it matures before the curve's reference date."""
    return [
        _deposit(0.021, 1, TimeUnit.Days, 0),
        _deposit(0.022, 1, TimeUnit.Months, 2),
        _deposit(0.024, 3, TimeUnit.Months, 2),
        _deposit(0.026, 6, TimeUnit.Months, 2),
        _deposit(0.028, 1, TimeUnit.Years, 2),
    ]


def _instrument_weights() -> list[float]:
    """probe.cpp:121."""
    return [0.5, 1.0, 2.0, 1.0, 1.5]


def _make_additional_helpers() -> list[DepositRateHelper]:
    """probe.cpp:125-127. [0] is dead — its pillar lands ON the reference date."""
    return [
        _deposit(0.020, 2, TimeUnit.Days, 0),
        _deposit(0.027, 9, TimeUnit.Months, 2),
    ]


def _additional_dates() -> list[Date]:
    """probe.cpp:132-141 — unsorted, three expired entries and one duplicate."""
    cal = TARGET()
    return [
        _TODAY - 1,
        _SETTLEMENT,
        cal.advance(_SETTLEMENT, 2, TimeUnit.Months),
        cal.advance(_SETTLEMENT, 1, TimeUnit.Months),
        cal.advance(_SETTLEMENT, 4, TimeUnit.Months),
        _TODAY - 2,
    ]


def _penalties(times: list[float], data: list[float]) -> npt.NDArray[np.float64]:
    """probe.cpp:145-150."""
    return np.array(
        [
            10.0 * (data[1] - data[0]) + times[1],
            5.0 * (data[-1] - data[-2]) - times[-1],
        ],
        dtype=np.float64,
    )


def _constant_penalties() -> npt.NDArray[np.float64]:
    """probe.cpp:155-160 — the zero-argument penalty shape."""
    return np.array([0.25, -0.5], dtype=np.float64)


def _make_curve(instruments: list[DepositRateHelper]) -> PiecewiseYieldCurve:
    """probe.cpp:172-176 — Discount / LogLinear, Actual365Fixed, ref = settlement."""
    return PiecewiseYieldCurve(
        Discount, _SETTLEMENT, instruments, Actual365Fixed(),
    )


def _full_bootstrap() -> GlobalBootstrap:
    """probe.cpp:180-183 — additional helpers, dates, two-argument penalties, weights."""
    return GlobalBootstrap(
        Discount,
        additional_helpers=_make_additional_helpers(),
        additional_dates=_additional_dates,
        additional_penalties=_penalties,
        instrument_weights=_instrument_weights(),
    )


def _serials(dates: list[Date]) -> list[int]:
    return [d.serial_number() for d in dates]


# ---------------------------------------------------------------------------
# SimpleQuoteVariables — globalbootstrapvars.{hpp,cpp}
# ---------------------------------------------------------------------------


def test_simple_quote_variables_no_bounds() -> None:
    """Empty bounds -> ``detail::get`` yields Null -> both transforms identity.

    # C++ parity: globalbootstrapvars.cpp:40-48; probe section
    # ``simple_quote_variables.no_bounds``. Tier: EXACT — every value is
    # either copied through or written unchanged.
    """
    ref = REF["simple_quote_variables"]["no_bounds"]
    quotes = [SimpleQuote(0.11), SimpleQuote(0.22), SimpleQuote(0.33)]
    v = SimpleQuoteVariables(quotes)

    invalid = v.initialize(valid_data=False)
    assert len(invalid) == len(ref["initialize_invalid"])
    for got, want in zip(invalid, ref["initialize_invalid"], strict=True):
        exact(float(got), want)
    for q, want in zip(quotes, ref["quotes_after_initialize_invalid"], strict=True):
        exact(q.value(), want)

    x = np.array(ref["update_x"], dtype=np.float64)
    v.update(x)
    for q, want in zip(quotes, ref["quotes_after_update"], strict=True):
        exact(q.value(), want)

    valid = v.initialize(valid_data=True)
    for got, want in zip(valid, ref["initialize_valid"], strict=True):
        exact(float(got), want)


def test_simple_quote_variables_short_vectors() -> None:
    """A SHORT bounds/guess vector falls back to its LAST entry, not the default.

    # C++ parity: ql/utilities/vectors.hpp:33-42 — ``detail::get`` returns
    # ``v.back()`` past the end of a non-empty vector. Quote 2 therefore
    # inherits quote 1's guess (0.5) AND its lower bound (-1.0); probe
    # section ``simple_quote_variables.short_vectors``. Tier: TIGHT (exp/log
    # round trips).
    """
    ref = REF["simple_quote_variables"]["short_vectors"]
    quotes = [SimpleQuote(), SimpleQuote(), SimpleQuote()]
    v = SimpleQuoteVariables(quotes, ref["initial_guesses"], ref["lower_bounds"])

    invalid = v.initialize(valid_data=False)
    for got, want in zip(invalid, ref["initialize_invalid"], strict=True):
        tight(float(got), want)
    # The tail quote picked up 0.5, the LAST configured guess.
    for q, want in zip(quotes, ref["quotes_after_initialize_invalid"], strict=True):
        exact(q.value(), want)

    v.update(np.array(ref["update_x"], dtype=np.float64))
    for q, want in zip(quotes, ref["quotes_after_update"], strict=True):
        tight(q.value(), want)

    valid = v.initialize(valid_data=True)
    for got, want in zip(valid, ref["initialize_valid"], strict=True):
        tight(float(got), want)


def test_simple_quote_variables_guesses_without_bounds() -> None:
    """Guesses with an EMPTY bounds vector: identity transform, guess broadcast.

    # C++ parity: globalbootstrapvars.cpp:26-29; probe section
    # ``simple_quote_variables.guesses_no_bounds``. Tier: EXACT.
    """
    ref = REF["simple_quote_variables"]["guesses_no_bounds"]
    quotes = [SimpleQuote(), SimpleQuote()]
    v = SimpleQuoteVariables(quotes, ref["initial_guesses"])
    invalid = v.initialize(valid_data=False)
    for got, want in zip(invalid, ref["initialize_invalid"], strict=True):
        exact(float(got), want)
    for q, want in zip(quotes, ref["quotes_after_initialize_invalid"], strict=True):
        exact(q.value(), want)


def test_simple_quote_variables_rejects_over_long_vectors() -> None:
    """# C++ parity: globalbootstrapvars.cpp:15-16 — the two QL_REQUIREs."""
    ref = REF["simple_quote_variables"]
    quotes = [SimpleQuote(0.1)]
    with pytest.raises(LibraryException) as guess_err:
        SimpleQuoteVariables(quotes, [1.0, 2.0])
    assert ref["too_many_initial_guesses_message"] in str(guess_err.value)
    with pytest.raises(LibraryException) as bound_err:
        SimpleQuoteVariables(quotes, [], [1.0, 2.0])
    assert ref["too_many_lower_bounds_message"] in str(bound_err.value)


# ---------------------------------------------------------------------------
# setup() — defaults and the instrument-weight contract
# ---------------------------------------------------------------------------


def test_setup_builds_default_optimizer_and_end_criteria() -> None:
    """# C++ parity: globalbootstrap.hpp:216-229.

    The accuracy falls back to the CURVE's accuracy (1e-12) when the
    bootstrap was given none, and both the optimizer and the end criteria
    are built from it. Tier: EXACT on the end-criteria fields, which the
    probe reads back off a C++ ``EndCriteria(1000, 10, a, a, a)``.
    """
    ref = REF["setup_defaults"]
    curve = _make_curve(_make_instruments())
    gb = GlobalBootstrap(Discount)
    gb.setup(curve)

    ec = gb.end_criteria()
    assert ec is not None
    assert ec.max_iterations == ref["end_criteria_max_iterations"]
    assert ec.max_stationary_state == ref["end_criteria_max_stationary_state_iterations"]
    exact(ec.root_epsilon, ref["end_criteria_root_epsilon"])
    exact(ec.function_epsilon, ref["end_criteria_function_epsilon"])
    exact(ec.gradient_norm_epsilon, ref["end_criteria_gradient_norm_epsilon"])

    optimizer = gb.optimizer()
    assert isinstance(optimizer, LevenbergMarquardt)
    # C++ globalbootstrap.hpp:225 — LevenbergMarquardt(a, a, a); the three
    # arguments are epsfcn / xtol / gtol in that order.
    exact(optimizer.epsfcn, ref["curve_accuracy"])
    exact(optimizer.xtol, ref["curve_accuracy"])
    exact(optimizer.gtol, ref["curve_accuracy"])


def test_multi_curve_bootstrap_null_accuracy_default() -> None:
    """# C++ parity: globalbootstrap.cpp:34-38 — ``constexpr auto accuracy = 1E-10``."""
    ref = REF["setup_defaults"]
    mcb = MultiCurveBootstrap()
    ec = mcb.end_criteria()
    exact(ec.root_epsilon, ref["multi_curve_end_criteria_root_epsilon"])
    exact(ec.function_epsilon, ref["multi_curve_null_accuracy"])
    assert ec.max_iterations == ref["end_criteria_max_iterations"]
    assert ec.max_stationary_state == ref["end_criteria_max_stationary_state_iterations"]


def test_setup_resizes_instrument_weights_to_one() -> None:
    """# C++ parity: globalbootstrap.hpp:236 — ``resize(n, 1.0)``."""
    instruments = _make_instruments()
    gb = GlobalBootstrap(Discount)
    gb.setup(_make_curve(instruments))
    assert gb.instrument_weights() == [1.0] * len(instruments)


def test_setup_rejects_wrong_number_of_instrument_weights() -> None:
    """# C++ parity: globalbootstrap.hpp:232-235 — message pinned by the probe."""
    gb = GlobalBootstrap(Discount, instrument_weights=[1.0, 2.0])
    with pytest.raises(LibraryException) as err:
        gb.setup(_make_curve(_make_instruments()))
    assert REF["instrument_weights_require_message"] in str(err.value)


# ---------------------------------------------------------------------------
# initialize() — globalbootstrap.hpp:242-317
# ---------------------------------------------------------------------------


def test_helper_pillar_dates_match_cpp() -> None:
    """Precondition for everything below: identical helper sets.

    # C++ parity: probe section ``single_curve.instrument_pillar_dates`` /
    # ``additional_helper_pillar_dates``. Tier: EXACT (dates).
    """
    ref = REF["single_curve"]
    assert _serials([h.pillar_date() for h in _make_instruments()]) == (
        ref["instrument_pillar_dates"]
    )
    assert _serials([h.pillar_date() for h in _make_additional_helpers()]) == (
        ref["additional_helper_pillar_dates"]
    )
    assert _serials(_additional_dates()) == ref["raw_additional_dates"]
    assert _SETTLEMENT.serial_number() == REF["settlement_date"]


def test_initialize_grid_and_alive_selection() -> None:
    """Dates/times/max-date and the alive splits, all optimizer-free.

    # C++ parity: globalbootstrap.hpp:242-317. The grid is
    # ``{firstDate} + alive pillars + surviving additional dates``, sorted
    # and uniqued: the three additional dates at or before the reference
    # date are dropped (hpp:271) and the one that duplicates the 1M pillar
    # is collapsed (hpp:288). Tier: EXACT on dates, TIGHT on times.
    """
    ref = REF["single_curve"]
    instruments = _make_instruments()
    curve = _make_curve(instruments)
    gb = _full_bootstrap()
    gb.setup(curve)
    gb.setup_cost_function()  # hpp:331 — this is what triggers initialize()

    assert _serials(curve.dates()) == ref["dates"]
    for got, want in zip(curve.times(), ref["times"], strict=True):
        tight(got, want)
    assert curve.max_date().serial_number() == ref["max_date"]

    # The overnight deposit's pillar (45954) is before the reference date
    # (45957), so it is dropped together with its weight (0.5).
    assert len(gb.alive_instruments()) == ref["n_alive_instruments"]
    assert gb.alive_instruments() == instruments[1:]
    assert gb.alive_instrument_weights() == _instrument_weights()[1:]

    # The 2-day additional helper's pillar lands ON the reference date, and
    # the test is ``pillarDate() > firstDate``, so it is dead too.
    assert len(gb.alive_additional_helpers()) == 1
    assert (
        gb.alive_additional_helpers()[0].pillar_date().serial_number()
        == ref["additional_helper_pillar_dates"][1]
    )


def test_initialize_installs_initial_value_and_bare_grid() -> None:
    """A bootstrap with no additional machinery: grid = first date + alive pillars.

    # C++ parity: globalbootstrap.hpp:313 — ``data_ = vector(n, initialValue)``;
    # probe section ``bare_curve``. Tier: EXACT on dates, TIGHT on times/guess.
    """
    ref = REF["bare_curve"]
    curve = _make_curve(_make_instruments())
    gb = GlobalBootstrap(Discount, instrument_weights=_instrument_weights())
    gb.setup(curve)
    guess = gb.setup_cost_function()

    assert _serials(curve.dates()) == ref["dates"]
    for got, want in zip(curve.times(), ref["times"], strict=True):
        tight(got, want)
    assert curve.max_date().serial_number() == ref["max_date"]
    exact(Discount().initial_value(curve), ref["initial_value"])
    for got, want in zip(guess, ref["guess"], strict=True):
        tight(float(got), want)


def test_initialize_requires_enough_curve_points() -> None:
    """# C++ parity: globalbootstrap.hpp:291-294.

    A single deposit whose pillar is before the curve's reference date
    leaves the grid with only the reference date itself, one point short of
    LogLinear's ``requiredPoints == 2``.
    """
    curve = _make_curve([_deposit(0.021, 1, TimeUnit.Days, 0)])
    gb = GlobalBootstrap(Discount)
    gb.setup(curve)
    with pytest.raises(LibraryException, match="not enough curve points"):
        gb.setup_cost_function()


# ---------------------------------------------------------------------------
# setupCostFunction() — globalbootstrap.hpp:319-376
# ---------------------------------------------------------------------------


def test_setup_cost_function_guess() -> None:
    """The initial guess, i.e. traits ``guess`` -> ``updateGuess`` -> ``transformInverse``.

    # C++ parity: globalbootstrap.hpp:366-374. For ``Discount`` the
    # transform is ``log``, so ``guess[i] == log(data[i+1])`` and the guesses
    # are chained: ``Discount::guess`` for pillar i reads ``data[i-1]``,
    # which the previous ``updateGuess`` has just written. Tier: TIGHT.
    """
    ref = REF["single_curve"]
    curve = _make_curve(_make_instruments())
    gb = _full_bootstrap()
    gb.setup(curve)
    guess = gb.setup_cost_function()

    assert len(guess) == len(ref["guess"])
    assert len(guess) == len(curve.times()) - 1
    for got, want in zip(guess, ref["guess"], strict=True):
        tight(float(got), want)
    # The guess really is log of the data the traits wrote back.
    for i, g in enumerate(guess):
        tight(float(np.exp(g)), curve.data()[i + 1])


# ---------------------------------------------------------------------------
# setCostFunctionArgument + evaluateCostFunction, at a FIXED x
# ---------------------------------------------------------------------------


def test_evaluate_cost_function_at_fixed_x() -> None:
    """Residuals at the probe's fixed ``x``: weighted quote errors, then penalties.

    # C++ parity: globalbootstrap.hpp:378-403. The reference ``x`` is used
    # verbatim rather than recomputed from the guess, so this layer is
    # pinned independently of ``setup_cost_function``. Tier: TIGHT.
    """
    ref = REF["single_curve"]
    curve = _make_curve(_make_instruments())
    gb = _full_bootstrap()
    gb.setup(curve)
    gb.setup_cost_function()

    gb.set_cost_function_argument(np.array(ref["x"], dtype=np.float64))
    # hpp:383 — updateGuess(transformDirect(x[i])), i.e. data[i+1] = exp(x[i]).
    for got, want in zip(curve.data(), ref["data_at_x"], strict=True):
        tight(got, want)

    residuals = gb.evaluate_cost_function()
    assert len(residuals) == len(ref["residuals_at_x"])
    # 4 alive instruments + 2 penalty terms (hpp:397).
    assert len(residuals) == ref["n_alive_instruments"] + 2
    for got, want in zip(residuals, ref["residuals_at_x"], strict=True):
        tight(float(got), want)


def test_evaluate_cost_function_applies_instrument_weights() -> None:
    """Each residual is ``quoteError() * weight`` — globalbootstrap.hpp:399.

    Tier: TIGHT. Checked against the helper's own quote error so a port that
    forgot the weight (or used the unfiltered weight vector) is caught even
    if the reference happened to agree.
    """
    ref = REF["single_curve"]
    curve = _make_curve(_make_instruments())
    gb = _full_bootstrap()
    gb.setup(curve)
    gb.setup_cost_function()
    gb.set_cost_function_argument(np.array(ref["x"], dtype=np.float64))
    residuals = gb.evaluate_cost_function()
    for i, helper in enumerate(gb.alive_instruments()):
        tight(
            float(residuals[i]),
            helper.quote_error() * gb.alive_instrument_weights()[i],
        )


def test_zero_argument_penalties_are_wrapped() -> None:
    """The third C++ constructor discards ``times``/``data`` and calls ``f()``.

    # C++ parity: globalbootstrap.hpp:196-206; probe section
    # ``constant_penalties``. Tier: TIGHT.
    """
    ref = REF["constant_penalties"]
    curve = _make_curve(_make_instruments())
    gb = GlobalBootstrap(
        Discount,
        additional_penalties=_constant_penalties,
        instrument_weights=_instrument_weights(),
    )
    gb.setup(curve)
    guess = gb.setup_cost_function()
    assert _serials(curve.dates()) == ref["dates"]
    for got, want in zip(guess, ref["guess"], strict=True):
        tight(float(got), want)

    x = np.array([g + 0.001 * (i + 1) for i, g in enumerate(guess)], dtype=np.float64)
    gb.set_cost_function_argument(x)
    residuals = gb.evaluate_cost_function()
    for got, want in zip(residuals, ref["residuals_at_x"], strict=True):
        tight(float(got), want)


# ---------------------------------------------------------------------------
# MultiCurveBootstrap concatenation layout — globalbootstrap.cpp:50-116
# ---------------------------------------------------------------------------


class _CapturingMethod(OptimizationMethod):
    """Records the problem's guess, evaluates once at a fixed point, stops.

    Mirrors ``CapturingMethod`` in probe.cpp:199-215: the point of the
    exercise is the concatenation LAYOUT, so no optimization is wanted.
    ``set_current_value`` is called with the probe point because that is
    what a real optimizer does with its accepted iterate
    (C++ levenbergmarquardt.cpp:136).
    """

    def __init__(self) -> None:
        self.guess: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.x: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.values: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        del end_criteria
        problem.reset()
        self.guess = np.array(problem.current_value, dtype=np.float64)
        self.x = np.array(
            [g + 0.001 * (i + 1) for i, g in enumerate(self.guess)], dtype=np.float64
        )
        self.values = np.array(problem.values(self.x), dtype=np.float64)
        problem.set_current_value(self.x)
        return Type.StationaryPoint


def test_multi_curve_bootstrap_concatenation_layout() -> None:
    """Guess sizes, slice offsets and the concatenated residual vector.

    # C++ parity: globalbootstrap.cpp:50-116. Curve B carries penalties, so
    # its RESIDUAL count (4) differs from its GUESS count (2): a port that
    # slices the results with the guess sizes gets a different answer.
    # Tier: TIGHT.
    """
    ref = REF["multi_curve_layout"]
    instruments_a = _make_instruments()
    # probe.cpp:510-511 — a shorter second curve.
    instruments_b = [
        _deposit(0.019, 1, TimeUnit.Months, 2),
        _deposit(0.023, 6, TimeUnit.Months, 2),
    ]
    curve_a = _make_curve(instruments_a)
    curve_b = _make_curve(instruments_b)
    gb_a = GlobalBootstrap(Discount, instrument_weights=_instrument_weights())
    gb_b = GlobalBootstrap(Discount, additional_penalties=_constant_penalties)
    gb_a.setup(curve_a)
    gb_b.setup(curve_b)

    method = _CapturingMethod()
    mcb = MultiCurveBootstrap(
        optimizer=method, end_criteria=EndCriteria(1000, 10, 1e-10, 1e-10, 1e-10)
    )
    mcb.add(gb_a)
    mcb.add(gb_b)
    mcb.run_multi_curve_bootstrap()

    guess_sizes = [len(curve_a.times()) - 1, len(curve_b.times()) - 1]
    assert guess_sizes == ref["guess_sizes"]
    assert [0, guess_sizes[0]] == ref["offsets"]

    for got, want in zip(method.guess, ref["global_guess"], strict=True):
        tight(float(got), want)
    for got, want in zip(method.x, ref["x"], strict=True):
        tight(float(got), want)
    assert len(method.values) == len(ref["global_values_at_x"])
    for got, want in zip(method.values, ref["global_values_at_x"], strict=True):
        tight(float(got), want)

    # Each contributor received ITS slice: had the offsets been wrong the
    # curve data would not agree.
    assert _serials(curve_a.dates()) == ref["curve_a_dates"]
    assert _serials(curve_b.dates()) == ref["curve_b_dates"]
    for got, want in zip(curve_a.data(), ref["curve_a_data_at_x"], strict=True):
        tight(got, want)
    for got, want in zip(curve_b.data(), ref["curve_b_data_at_x"], strict=True):
        tight(got, want)

    # cpp:114-115 — every contributor is marked valid once the solve returns.
    assert gb_a.valid_curve()
    assert gb_b.valid_curve()


def test_multi_curve_bootstrap_guess_sizes_accessor() -> None:
    """``guess_sizes()`` reproduces the per-contributor split of cpp:55-59."""
    ref = REF["multi_curve_layout"]
    gb_a = GlobalBootstrap(Discount, instrument_weights=_instrument_weights())
    gb_b = GlobalBootstrap(Discount, additional_penalties=_constant_penalties)
    gb_a.setup(_make_curve(_make_instruments()))
    gb_b.setup(
        _make_curve(
            [
                _deposit(0.019, 1, TimeUnit.Months, 2),
                _deposit(0.023, 6, TimeUnit.Months, 2),
            ]
        )
    )
    mcb = MultiCurveBootstrap(accuracy=1e-10)
    mcb.add(gb_a)
    mcb.add(gb_b)
    assert mcb.guess_sizes() == ref["guess_sizes"]


def test_multi_curve_bootstrap_rejects_failed_optimization() -> None:
    """# C++ parity: globalbootstrap.cpp:107-110."""

    class _Failing(OptimizationMethod):
        def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
            del problem, end_criteria
            return Type.MaxIterations

    gb = GlobalBootstrap(Discount)
    gb.setup(_make_curve(_make_instruments()))
    mcb = MultiCurveBootstrap(optimizer=_Failing())
    mcb.add(gb)
    with pytest.raises(LibraryException, match="during multi curve bootstrap"):
        mcb.run_multi_curve_bootstrap()


def test_multi_curve_bootstrap_rejects_accuracy_with_explicit_optimizer() -> None:
    """Not expressible in C++ (two disjoint constructors), so rejected here."""
    with pytest.raises(LibraryException, match="accuracy cannot be combined"):
        MultiCurveBootstrap(accuracy=1e-10, optimizer=LevenbergMarquardt())


# ---------------------------------------------------------------------------
# the converged curve — the only optimizer-dependent layer
# ---------------------------------------------------------------------------


def test_converged_single_curve_reproduces_cpp() -> None:
    """The converged curve, at LOOSE.

    # C++ parity: globalbootstrap.hpp:405-429; probe section
    # ``converged_single_curve``.

    **Tolerance rationale.** This is the ONLY layer whose value depends on
    the optimizer, and the two optimizers are not the same code: C++ runs
    the MINPACK ``lmdif`` translation embedded in
    ql/math/optimization/levenbergmarquardt.cpp, pquantlib delegates to
    ``scipy.optimize.least_squares(method='lm')``. Both drive the same
    least-squares problem to the same minimum, but by different iterate
    sequences, so agreement is bounded by how flat the objective is near
    the solution rather than by floating-point reproducibility. The problem
    here is exactly determined (4 residuals, 4 unknowns) with a residual
    norm at the solution of order 1e-16, so both land on the same root.

    Measured agreement on this fixture is 1.1e-16 relative on the worst
    ``data`` entry — i.e. the two optimizers reach the same double. The
    assertion is nevertheless LOOSE, deliberately: the number is a property
    of where scipy's MINPACK stops on a flat objective, not of arithmetic
    the two codes share, so it is not a value this port controls across
    scipy / BLAS versions. LOOSE is the contract; the measured margin is
    recorded here so a future 1e-9 drift is recognisable as a regression
    rather than assumed to be normal optimizer noise.
    """
    ref = REF["converged_single_curve"]
    instruments = _make_instruments()
    curve = _make_curve(instruments)
    gb = GlobalBootstrap(Discount, instrument_weights=_instrument_weights())
    gb.setup(curve)
    gb.calculate()

    assert _serials(curve.dates()) == ref["dates"]
    for got, want in zip(curve.times(), ref["times"], strict=True):
        tight(got, want)
    for got, want in zip(curve.data(), ref["data"], strict=True):
        loose(got, want)
    for d, want in zip(curve.dates(), ref["discounts_at_pillars"], strict=True):
        loose(curve.discount(d), want)


def test_converged_single_curve_reprices_its_helpers() -> None:
    """Every alive helper is repriced — the property the bootstrap exists for.

    # C++ parity: probe section ``converged_single_curve.quote_errors``,
    # which is ~1e-16 for all four alive helpers. Asserted as "small"
    # against an absolute bound rather than against the C++ value, because
    # the C++ number is a rounding-noise residual with no reproducible
    # digits; 1e-10 is the accuracy the default EndCriteria asks for
    # (globalbootstrap.hpp:228 with the curve accuracy of 1e-12).
    """
    instruments = _make_instruments()
    curve = _make_curve(instruments)
    gb = GlobalBootstrap(Discount, instrument_weights=_instrument_weights())
    gb.setup(curve)
    gb.calculate()
    for helper in gb.alive_instruments():
        assert abs(helper.quote_error()) < 1e-10


def test_calculate_rejects_failed_optimization() -> None:
    """# C++ parity: globalbootstrap.hpp:426-427."""

    class _Failing(OptimizationMethod):
        def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
            del problem, end_criteria
            return Type.MaxIterations

    curve = _make_curve(_make_instruments())
    gb = GlobalBootstrap(Discount, optimizer=_Failing())
    gb.setup(curve)
    with pytest.raises(LibraryException, match="failed to minimize"):
        gb.calculate()


def test_calculate_delegates_to_parent_bootstrapper() -> None:
    """# C++ parity: globalbootstrap.hpp:408-411."""
    curve = _make_curve(_make_instruments())
    gb = GlobalBootstrap(Discount)
    gb.setup(curve)
    method = _CapturingMethod()
    mcb = MultiCurveBootstrap(optimizer=method)
    mcb.add(gb)
    gb.calculate()
    # The joint solve ran, not the single-curve one: the capturing method
    # only ever sees a problem raised by run_multi_curve_bootstrap.
    assert len(method.guess) == len(curve.times()) - 1
    assert gb.valid_curve()
