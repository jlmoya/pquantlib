"""Cross-validate the v1.43 basket pricing engines against the C++ probe.

Reference: ``migration-harness/references/v143/pe/basket`` produced by
``migration-harness/cpp/probes/v143_pe_basket/probe.cpp``.

Covers :class:`VectorBsmProcessExtractor`, :class:`SumExponentialsRootSolver`,
:class:`SingleFactorBsmBasketEngine`, :class:`BjerksundStenslandSpreadEngine`,
:class:`OperatorSplittingSpreadEngine`, :class:`ChoiBasketEngine`,
:class:`DengLiZhouBasketEngine`, :class:`MCEuropeanBasketEngine` /
:class:`EuropeanMultiPathPricer` / :class:`MakeMCEuropeanBasketEngine`, and
:class:`MCAmericanBasketEngine` / :class:`AmericanBasketPathPricer` /
:class:`MakeMCAmericanBasketEngine`.

Every case carries its whole market description — evaluation date, curves,
vols, correlation, weights, payoff, and each engine's tuning knobs — so the
sweeps below reconstruct the market rather than restating constants. Six
markets are covered, all taken from the upstream v1.43 test-suite where one
exists: the ``{200, 50, -125}`` single-factor basket with its *negative* spot
and *negative* volatilities; the Bjerksund-Stensland futures pair; Chi-Fai Lo's
spread market; the four-asset golden Choi basket; the Deng-Li-Zhou
mixed-sign basket; and a three-leg Merton array for the Monte Carlo engines.

Tolerances
----------
TIGHT (``1e-14`` abs / ``1e-12`` rel) throughout, including every Monte Carlo
value. Nothing here is a statistical band: the C++ RNG
(``MersenneTwisterUniformRng`` + ``InverseCumulativeNormal``) is fully
reproducible, so a fixed non-zero seed pins the exact NPV *and* the exact error
estimate. The largest deviation observed across the whole reference is
``1.4e-14`` scaled — round-off in the last two bits of a sum, not a modelling
difference — and the tightest engine (``VectorBsmProcessExtractor``) agrees
bit-for-bit.

Monte Carlo reproducibility is not free, and two alignments were needed to get
it; both are asserted here so they cannot regress:

* ``StochasticProcessArray`` correlates its Brownian increments with
  ``pseudoSqrt(rho, Spectral)``, whose *value* — not merely its Gram product —
  is observable through the generated paths. It must come from
  ``SymmetricSchurDecomposition`` (eigenvalues descending, C++'s sign
  convention) plus C++'s row normalisation, not from ``numpy.linalg.eigh``.
  See :func:`test_spectral_pseudo_sqrt_matches_cpp`.
* ``MCLongstaffSchwartzEngine``'s calibration seed is *not*
  ``seed + 1768237423``; that branch is unreachable in C++. See
  :func:`test_lsm_calibration_seed_is_null_size`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import (
    AverageBasketPayoff,
    BasketOption,
    BasketPayoff,
    MaxBasketPayoff,
    MinBasketPayoff,
    SpreadBasketPayoff,
)
from pquantlib.math.array import Array
from pquantlib.math.matrix import Matrix
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer import (
    LongstaffSchwartzPathPricer,
)
from pquantlib.methods.montecarlo.lsm_basis_system import PolynomialType
from pquantlib.methods.montecarlo.monte_carlo_model import MonteCarloModel
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.bjerksund_stensland_spread_engine import (
    BjerksundStenslandSpreadEngine,
)
from pquantlib.pricingengines.basket.choi_basket_engine import (
    MAX_NR_INTEGRATION_STEPS_UNBOUNDED,
    ChoiBasketEngine,
)
from pquantlib.pricingengines.basket.deng_li_zhou_basket_engine import (
    DengLiZhouBasketEngine,
)
from pquantlib.pricingengines.basket.mc_american_basket_engine import (
    AmericanBasketPathPricer,
    MakeMCAmericanBasketEngine,
)
from pquantlib.pricingengines.basket.mc_european_basket_engine import (
    MakeMCEuropeanBasketEngine,
)
from pquantlib.pricingengines.basket.operator_splitting_spread_engine import (
    OperatorSplittingSpreadEngine,
)
from pquantlib.pricingengines.basket.single_factor_bsm_basket_engine import (
    SingleFactorBsmBasketEngine,
    SumExponentialsRootSolver,
)
from pquantlib.pricingengines.basket.vector_bsm_process_extractor import (
    VectorBsmProcessExtractor,
)
from pquantlib.processes.black_process import BlackProcess
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.processes.stochastic_process_array import (
    StochasticProcessArray,
    _spectral_sqrt,  # pyright: ignore[reportPrivateUsage]
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid

_DC = Actual365Fixed()
_EPS = float(np.finfo(np.float64).eps)


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Save/restore the evaluation date.

    Every case sets its own date from the reference's ``today`` field, matching
    ``Settings::instance().evaluationDate() = ...`` in each section of
    probe.cpp (see the ``emit*`` functions there). ``BasketOption::isExpired()``
    short-circuits the NPV to 0 past maturity, so a wall-clock-dependent test
    would silently pass or fail with the calendar.
    """
    settings = ObservableSettings()
    saved = settings.evaluation_date
    yield
    settings.evaluation_date = saved


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/basket")


# --- market reconstruction ---------------------------------------------------


def _date(iso: str) -> Date:
    year, month, day = (int(part) for part in iso.split("-"))
    return Date.from_ymd(day, Month(month), year)


def _set_today(inputs: dict[str, Any]) -> Date:
    today = _date(str(inputs["today"]))
    ObservableSettings().evaluation_date = today
    return today


def _flat_curve(reference_date: Date, rate: float) -> YieldTermStructure:
    return FlatForward.from_rate(
        reference_date=reference_date, forward_rate=rate, day_counter=_DC
    )


def _flat_vol(reference_date: Date, volatility: float) -> BlackConstantVol:
    return BlackConstantVol(
        reference_date=reference_date,
        calendar=NullCalendar(),
        day_counter=_DC,
        volatility=volatility,
    )


def _merton_legs(
    inputs: dict[str, Any], today: Date, risk_free: YieldTermStructure
) -> list[GeneralizedBlackScholesProcess]:
    """One Merton leg per (spot, dividend yield, volatility) triple.

    The risk-free curve is shared, as C++ does: ``VectorBsmProcessExtractor``
    checks that every leg agrees on the discount factor at maturity.
    """
    return [
        BlackScholesMertonProcess(
            x0=SimpleQuote(float(s)),
            dividend_ts=_flat_curve(today, float(q)),
            risk_free_ts=risk_free,
            black_vol_ts=_flat_vol(today, float(v)),
        )
        for s, q, v in zip(
            inputs["spots"], inputs["dividend_yields"], inputs["volatilities"],
            strict=True,
        )
    ]


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _basket_payoff(
    kind: str, option_type: str, strike: float, weights: list[float]
) -> BasketPayoff:
    base = PlainVanillaPayoff(_option_type(option_type), strike)
    match kind:
        case "max":
            return MaxBasketPayoff(base)
        case "min":
            return MinBasketPayoff(base)
        case "spread":
            return SpreadBasketPayoff(base)
        case _:
            return AverageBasketPayoff(base, weights)


def _constant_correlation(n: int, rho: float) -> Matrix:
    m: Matrix = np.full((n, n), rho, dtype=np.float64)
    np.fill_diagonal(m, 1.0)
    return m


def _mc_process_array(inputs: dict[str, Any], today: Date) -> StochasticProcessArray:
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
    legs: list[StochasticProcess1D] = list(_merton_legs(inputs, today, risk_free))
    return StochasticProcessArray(
        legs,
        _constant_correlation(int(inputs["n_assets"]), float(inputs["correlation"])),
    )


def _throws(f: Callable[[], object]) -> bool:
    try:
        f()
    except LibraryException:
        return True
    return False


def _cases(cpp: dict[str, Any], prefix: str, *, priced: bool = True) -> list[str]:
    """Case names with ``prefix``, restricted to priced (or guard) cases."""
    return [
        name
        for name, case in cpp.items()
        if name.startswith(prefix)
        and (("npv" in case["expected"]) is priced)
    ]


# --- VectorBsmProcessExtractor ------------------------------------------------


def test_vector_bsm_process_extractor(cpp: dict[str, Any]) -> None:
    """Every accessor, on a market with a negative spot and a negative vol.

    Agreement is bit-exact: each value is one arithmetic step off a quote or a
    discount factor, so there is nothing to accumulate.
    """
    case = cpp["vx_extract"]
    inputs, expected = case["inputs"], case["expected"]
    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
    extractor = VectorBsmProcessExtractor(_merton_legs(inputs, today, risk_free))

    for key, actual in (
        ("spot", extractor.get_spot()),
        ("dividend_yield_df", extractor.get_dividend_yield_df(maturity)),
        ("black_variance", extractor.get_black_variance(maturity)),
        ("black_std_dev", extractor.get_black_std_dev(maturity)),
    ):
        for got, want in zip(actual, expected[key], strict=True):
            tolerance.exact(float(got), float(want), reason=f"vx.{key}")
    tolerance.exact(
        extractor.get_interest_rate_df(maturity), float(expected["interest_rate_df"])
    )


def test_black_std_dev_is_signed_but_variance_is_not(cpp: dict[str, Any]) -> None:
    """A negative volatility gives a negative std-dev and a positive variance.

    ``getBlackStdDev`` is ``blackVol * sqrt(t)``; ``getBlackVariance`` is
    ``vol**2 * t``. The asymmetry is not incidental —
    ``SingleFactorBsmBasketEngine`` relies on it to admit a negative-weight leg
    through its ``a * sig >= 0`` guard, and the upstream ``{200, 50, -125}``
    market exists to exercise exactly that. A port that returned
    ``sqrt(variance)`` here would pass nothing downstream.
    """
    case = cpp["vx_signed_std_dev"]
    inputs, expected = case["inputs"], case["expected"]
    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
    extractor = VectorBsmProcessExtractor(_merton_legs(inputs, today, risk_free))

    std_dev = extractor.get_black_std_dev(maturity)
    variance = extractor.get_black_variance(maturity)

    assert bool(expected["std_dev_leg2_is_negative"])
    assert float(std_dev[2]) < 0.0
    tolerance.exact(float(std_dev[2]), float(expected["std_dev_leg2"]))
    tolerance.exact(float(variance[2]), float(expected["variance_leg2"]))
    tolerance.exact(
        math.sqrt(float(variance[2])), float(expected["sqrt_variance_leg2"])
    )
    assert float(std_dev[2]) == -math.sqrt(float(variance[2]))


def test_interest_rate_check_compares_discount_factors_not_curves(
    cpp: dict[str, Any],
) -> None:
    """Two *distinct* curves at the same rate are accepted; different rates throw.

    The C++ check is ``close_enough`` over the per-leg discount factors, not an
    identity comparison of ``riskFreeRate().currentLink()`` —
    ``GaussianCopulaSpreadEngine`` does the latter, and copying that check here
    would reject a market C++ prices. This is the only case that can tell the
    two apart, since nothing about the priced result differs.
    """
    ok_case = cpp["vx_distinct_but_equal_curves_ok"]
    bad_case = cpp["vx_different_rates_throws"]
    assert bool(ok_case["expected"]["throws"]) is False
    assert bool(bad_case["expected"]["throws"]) is True

    inputs = ok_case["inputs"]
    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))

    def build(rate2: float) -> None:
        curve_a = _flat_curve(today, float(inputs["rate_1"]))
        curve_b = _flat_curve(today, rate2)
        assert curve_a is not curve_b
        legs = [
            BlackScholesMertonProcess(
                x0=SimpleQuote(100.0),
                dividend_ts=_flat_curve(today, 0.0),
                risk_free_ts=curve_a,
                black_vol_ts=_flat_vol(today, 0.2),
            ),
            BlackScholesMertonProcess(
                x0=SimpleQuote(50.0),
                dividend_ts=_flat_curve(today, 0.0),
                risk_free_ts=curve_b,
                black_vol_ts=_flat_vol(today, 0.3),
            ),
        ]
        VectorBsmProcessExtractor(legs).get_interest_rate_df(maturity)

    assert _throws(lambda: build(float(inputs["rate_1"]))) is False
    with pytest.raises(
        LibraryException, match="interest rates need to be the same for all underlyings"
    ):
        build(float(bad_case["inputs"]["rate_2"]))


# --- SumExponentialsRootSolver ------------------------------------------------


def test_sum_exponentials_values(cpp: dict[str, Any]) -> None:
    """``f``, ``f'`` and ``f''`` at fixed abscissae, plus the counters.

    A fresh solver counts one increment per evaluation, so five abscissae give
    exactly five of each — the counters are what make the port's iteration
    observable at all.
    """
    checked = 0
    for name in _cases(cpp, "sumexp_values_", priced=False):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        solver = SumExponentialsRootSolver(inputs["a"], inputs["sig"], inputs["k"])
        for idx, x in enumerate(expected["x"]):
            tolerance.tight(solver(float(x)), float(expected["f"][idx]), reason=name)
        for idx, x in enumerate(expected["x"]):
            tolerance.tight(
                solver.derivative(float(x)), float(expected["derivative"][idx]),
                reason=name,
            )
        for idx, x in enumerate(expected["x"]):
            tolerance.tight(
                solver.second_derivative(float(x)),
                float(expected["second_derivative"][idx]),
                reason=name,
            )
        assert solver.get_f_ctr() == int(expected["f_ctr"])
        assert solver.get_derivative_ctr() == int(expected["derivative_ctr"])
        assert solver.get_second_derivative_ctr() == int(
            expected["second_derivative_ctr"]
        )
        checked += 1
    assert checked >= 8, f"expected the probe to carry value cases, got {checked}"


def test_sum_exponentials_root_and_evaluation_counters(cpp: dict[str, Any]) -> None:
    """Every ``Strategy`` at every configuration, root *and* counters.

    The counters are the point. A root can be reached by many iterations; only
    the evaluation counts prove the port runs QuantLib's own
    Brent / Newton / Ridder / Halley from QuantLib's own start point
    (``clamp((K - sum a) / sum(a sig), -10, 10)``, or ``0`` when the
    denominator underflows) rather than delegating to scipy.
    """
    checked = 0
    for name in _cases(cpp, "sumexp_root_", priced=False):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        strategy = SumExponentialsRootSolver.Strategy[str(inputs["strategy"])]
        solver = SumExponentialsRootSolver(inputs["a"], inputs["sig"], inputs["k"])
        root = solver.get_root(float(inputs["x_tol"]), strategy)

        tolerance.tight(root, float(expected["root"]), reason=f"{name}: root")
        assert solver.get_f_ctr() == int(expected["f_ctr"]), name
        assert solver.get_derivative_ctr() == int(expected["derivative_ctr"]), name
        assert solver.get_second_derivative_ctr() == int(
            expected["second_derivative_ctr"]
        ), name

        # The residual is not a TIGHT quantity and cannot be: f(x) = sum_i a_i
        # exp(sig_i x) - K is evaluated as a difference of two nearly equal
        # numbers at the root, so it is pure cancellation. Two bounds govern it:
        #   * round-off, |f| ~ eps * S with S = sum_i |a_i| exp(sig_i x) the
        #     gross magnitude of the summands (~ |K| near the root); and
        #   * the solver's own accuracy, |f| ~ |f'(x)| * x_tol, since the root
        #     is only located to x_tol.
        # Both are computed here from this case's own numbers rather than
        # guessed. A structural error — wrong start point, wrong iteration —
        # moves the root itself, which is asserted at TIGHT above.
        if "residual" not in expected:
            # The x_tol sweep pins the root and the counters only.
            checked += 1
            continue
        probe = SumExponentialsRootSolver(inputs["a"], inputs["sig"], inputs["k"])
        residual = probe(root)
        gross = sum(
            abs(float(a)) * math.exp(float(s) * root)
            for a, s in zip(inputs["a"], inputs["sig"], strict=True)
        )
        floor = 8.0 * _EPS * (gross + abs(float(inputs["k"]))) + 4.0 * abs(
            probe.derivative(root)
        ) * float(inputs["x_tol"])
        tolerance.custom(
            residual,
            float(expected["residual"]),
            abs_tol=floor,
            rel_tol=0.0,
            reason=f"{name}: residual is cancellation- and x_tol-limited",
        )
        checked += 1
    assert checked >= 30, f"expected the probe to carry root cases, got {checked}"


def test_sum_exponentials_strategy_enum_is_v143(cpp: dict[str, Any]) -> None:
    """v1.43 declares exactly ``{Ridder, Newton, Brent, Halley}``, in that order.

    No ``SuperHalley``: that value does not exist at this pin. The declaration
    order is asserted because ``Strategy`` is an ``IntEnum`` and a reordering
    would silently change what an integer-valued caller selects.
    """
    assert [s.name for s in SumExponentialsRootSolver.Strategy] == [
        "Ridder",
        "Newton",
        "Brent",
        "Halley",
    ]
    assert [int(s) for s in SumExponentialsRootSolver.Strategy] == [0, 1, 2, 3]
    used = {str(cpp[n]["inputs"]["strategy"]) for n in _cases(cpp, "sumexp_root_", priced=False)}
    assert used == {"Ridder", "Newton", "Brent", "Halley"}


@pytest.mark.parametrize(
    "case_name",
    [
        "sumexp_rejects_negative_a_times_sig",
        "sumexp_rejects_negative_a_times_sig_2",
        "sumexp_rejects_non_positive_strike_when_all_a_positive",
        "sumexp_rejects_negative_strike_when_all_a_positive",
    ],
)
def test_sum_exponentials_guards(cpp: dict[str, Any], case_name: str) -> None:
    inputs, expected = cpp[case_name]["inputs"], cpp[case_name]["expected"]
    assert bool(expected["throws"])
    with pytest.raises(LibraryException, match=str(inputs["why"])):
        SumExponentialsRootSolver(
            inputs["a"], inputs["sig"], float(inputs["k"])
        ).get_root()


def test_sum_exponentials_ctor_rejects_size_mismatch(cpp: dict[str, Any]) -> None:
    expected = cpp["sumexp_ctor_rejects_size_mismatch"]["expected"]
    assert bool(expected["throws"])
    with pytest.raises(LibraryException, match="Arrays must have the same size"):
        SumExponentialsRootSolver([1.0, 2.0], [0.1], 1.0)


# --- SingleFactorBsmBasketEngine ----------------------------------------------


def _single_factor_option(inputs: dict[str, Any], maturity: Date) -> BasketOption:
    return BasketOption(
        AverageBasketPayoff(
            PlainVanillaPayoff(
                _option_type(str(inputs["option_type"])), float(inputs["strike"])
            ),
            [float(w) for w in inputs["weights"]],
        ),
        EuropeanExercise(maturity),
    )


def test_single_factor_bsm_basket_engine(cpp: dict[str, Any]) -> None:
    """Every priced case, including the all-zero-vol intrinsic branch.

    ``additionalResults["d"]`` is written only on the solver branch: when every
    std-dev is ``close_enough(0)`` the engine returns the discounted intrinsic
    and writes nothing. That absence is asserted, not just the presence.
    """
    checked = 0
    for name in _cases(cpp, "sf_"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        legs = _merton_legs(inputs, today, risk_free)

        option = _single_factor_option(inputs, maturity)
        engine = (
            SingleFactorBsmBasketEngine(legs, float(inputs["x_tol"]))
            if "x_tol" in inputs
            else SingleFactorBsmBasketEngine(legs)
        )
        option.set_pricing_engine(engine)

        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        extra = option.additional_results()
        if "has_d" in expected:
            assert ("d" in extra) is bool(expected["has_d"]), name
            assert len(extra) == int(expected["additional_results_count"]), name
        if "d" in expected:
            tolerance.tight(float(extra["d"]), float(expected["d"]), reason=f"{name}: d")
        checked += 1
    assert checked >= 10, f"expected the probe to carry sf cases, got {checked}"


def test_single_factor_zero_vol_takes_the_intrinsic_branch(cpp: dict[str, Any]) -> None:
    """All-zero vols: discounted intrinsic on the summed forward basket.

    Checked against the arithmetic rather than only against the reference, so
    the branch is pinned as a *fact* about the engine and not just as a number.
    """
    name = "sf_2asset_zero_vol_call"
    inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
    assert bool(expected["has_d"]) is False
    assert int(expected["additional_results_count"]) == 0

    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
    legs = _merton_legs(inputs, today, risk_free)
    option = _single_factor_option(inputs, maturity)
    option.set_pricing_engine(SingleFactorBsmBasketEngine(legs))

    dr0 = risk_free.discount(maturity)
    forward_basket = sum(
        float(w) * float(s) * legs[k].dividend_yield().discount(maturity) / dr0
        for k, (w, s) in enumerate(zip(inputs["weights"], inputs["spots"], strict=True))
    )
    intrinsic = dr0 * max(0.0, forward_basket - float(inputs["strike"]))
    tolerance.tight(option.npv(), intrinsic, reason="zero-vol intrinsic")
    tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
    assert "d" not in option.additional_results()


def test_single_factor_x_tol_moves_the_answer(cpp: dict[str, Any]) -> None:
    """The root-finder tolerance is a real knob, not decoration.

    A port that hardcoded the default would still pass every default-tolerance
    case; these three differ from each other in ``d``, which is what forces the
    argument to be plumbed through.
    """
    roots = [
        float(cpp[name]["expected"]["d"])
        for name in sorted(cpp)
        if name.startswith("sf_3asset_call_xtol")
    ]
    assert len(roots) == 3
    assert len(set(roots)) == 3, "the probe must separate the three tolerances"


@pytest.mark.parametrize(
    "case_name",
    [
        "sf_rejects_non_average_payoff",
        "sf_rejects_wrong_weight_count",
        "sf_rejects_american_exercise",
    ],
)
def test_single_factor_guards(cpp: dict[str, Any], case_name: str) -> None:
    inputs, expected = cpp[case_name]["inputs"], cpp[case_name]["expected"]
    assert bool(expected["throws"])

    base = cpp["sf_3asset_call"]["inputs"]
    today = _set_today(base)
    maturity = _date(str(base["maturity"]))
    risk_free = _flat_curve(today, float(base["risk_free_rate"]))
    legs = _merton_legs(base, today, risk_free)

    payoff: BasketPayoff
    exercise: Exercise = EuropeanExercise(maturity)
    match case_name:
        case "sf_rejects_non_average_payoff":
            payoff = MaxBasketPayoff(PlainVanillaPayoff(OptionType.Call, 100.0))
        case "sf_rejects_wrong_weight_count":
            payoff = AverageBasketPayoff(
                PlainVanillaPayoff(OptionType.Call, 100.0), [1.0, 1.0]
            )
        case _:
            payoff = AverageBasketPayoff(
                PlainVanillaPayoff(OptionType.Call, 100.0), [1.0, 1.0, 1.0]
            )
            exercise = AmericanExercise(today, maturity)

    option = BasketOption(payoff, exercise)
    option.set_pricing_engine(SingleFactorBsmBasketEngine(legs))
    with pytest.raises(LibraryException, match=str(inputs["why"])):
        option.npv()


# --- spread engines: Bjerksund-Stensland and operator splitting ---------------


def _spread_legs(
    inputs: dict[str, Any], today: Date, risk_free: YieldTermStructure
) -> tuple[GeneralizedBlackScholesProcess, GeneralizedBlackScholesProcess]:
    if str(inputs["market"]) == "bs_merton":
        legs = _merton_legs(inputs, today, risk_free)
        return legs[0], legs[1]
    return (
        BlackProcess(
            x0=SimpleQuote(float(inputs["forward1"])),
            risk_free_ts=risk_free,
            black_vol_ts=_flat_vol(today, float(inputs["volatility1"])),
        ),
        BlackProcess(
            x0=SimpleQuote(float(inputs["forward2"])),
            risk_free_ts=risk_free,
            black_vol_ts=_flat_vol(today, float(inputs["volatility2"])),
        ),
    )


def _spread_option(inputs: dict[str, Any], maturity: Date) -> BasketOption:
    return BasketOption(
        SpreadBasketPayoff(
            PlainVanillaPayoff(
                _option_type(str(inputs["option_type"])), float(inputs["strike"])
            )
        ),
        EuropeanExercise(maturity),
    )


def test_bjerksund_stensland_spread_engine(cpp: dict[str, Any]) -> None:
    """Every case: both option types, strikes spanning the spread, rho in [-1, 1]."""
    checked = 0
    for name in _cases(cpp, "bs_"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        p1, p2 = _spread_legs(inputs, today, risk_free)

        option = _spread_option(inputs, maturity)
        option.set_pricing_engine(
            BjerksundStenslandSpreadEngine(p1, p2, float(inputs["correlation"]))
        )
        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        if "additional_results_count" in expected:
            assert len(option.additional_results()) == int(
                expected["additional_results_count"]
            )
        checked += 1
    assert checked >= 20, f"expected the probe to carry bs cases, got {checked}"


def test_bjerksund_stensland_satisfies_put_call_parity(cpp: dict[str, Any]) -> None:
    """``C - P == df (F1 - F2 - K)`` to round-off.

    Model-independent, so this is a fact about the engine rather than about the
    reference numbers: the put is obtained by flipping ``cp``, which flips every
    argument of Phi, and parity falls out. Upstream asserts the same identity at
    ``100 * QL_EPSILON``.
    """
    call = cpp["bs_call_k5_rho075"]
    put = cpp["bs_put_k5_rho075"]
    inputs = call["inputs"]
    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
    df = risk_free.discount(maturity)
    f1, f2 = float(inputs["forward1"]), float(inputs["forward2"])
    strike = float(inputs["strike"])

    def npv(case: dict[str, Any]) -> float:
        p1, p2 = _spread_legs(case["inputs"], today, risk_free)
        option = _spread_option(case["inputs"], maturity)
        option.set_pricing_engine(
            BjerksundStenslandSpreadEngine(
                p1, p2, float(case["inputs"]["correlation"])
            )
        )
        return option.npv()

    forward = (npv(call) - npv(put)) / df
    tolerance.tight(forward, f1 - f2 - strike, reason="put-call parity")


def test_operator_splitting_spread_engine(cpp: dict[str, Any]) -> None:
    """Every case, both orders, including the degenerate second-order branch."""
    checked = 0
    for name in _cases(cpp, "os_"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        p1, p2 = _spread_legs(inputs, today, risk_free)

        option = _spread_option(inputs, maturity)
        order_name = str(inputs["order"])
        engine = (
            OperatorSplittingSpreadEngine(p1, p2, float(inputs["correlation"]))
            if order_name == "default"
            else OperatorSplittingSpreadEngine(
                p1,
                p2,
                float(inputs["correlation"]),
                OperatorSplittingSpreadEngine.Order[order_name],
            )
        )
        option.set_pricing_engine(engine)
        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        checked += 1
    assert checked >= 35, f"expected the probe to carry os cases, got {checked}"


def test_operator_splitting_degenerate_branch_is_reached(cpp: dict[str, Any]) -> None:
    """``rs = (rho vol1 - sig2)**2`` is *exactly* zero in the degenerate market.

    The generic second-order expression divides by powers of
    ``e = rho vol1 - sig2``; at ``rs == 0`` it is ``0/0``. C++ switches to a
    closed form below ``QL_EPSILON**0.625``, and the market is built so the
    switch is unavoidable: ``f2 = 100``, ``k = 25`` give ``f2/(f2+k) = 0.8``
    exactly, ``vol2 = 0.25`` gives ``sig2 = 0.2`` exactly, and
    ``rho vol1 = 0.5 * 0.4 = 0.2``. A port with only the generic branch returns
    NaN here rather than being slightly wrong.
    """
    inputs = cpp["os_degenerate_call_Second"]["inputs"]
    vol1, vol2 = float(inputs["volatility1"]), float(inputs["volatility2"])
    f2, k, rho = (
        float(inputs["forward2"]),
        float(inputs["strike"]),
        float(inputs["correlation"]),
    )
    sig2 = vol2 * f2 / (f2 + k)
    assert (rho * vol1 - sig2) ** 2 == 0.0

    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
    p1, p2 = _spread_legs(inputs, today, risk_free)
    option = _spread_option(inputs, maturity)
    option.set_pricing_engine(
        OperatorSplittingSpreadEngine(
            p1, p2, rho, OperatorSplittingSpreadEngine.Order.Second
        )
    )
    npv = option.npv()
    assert math.isfinite(npv)
    tolerance.tight(
        npv, float(cpp["os_degenerate_call_Second"]["expected"]["npv"]),
        reason="degenerate branch",
    )

    # The nearby non-degenerate correlation takes the generic branch and lands
    # within 1e-5 — which is what makes the two branches a continuation of one
    # another rather than two unrelated formulas.
    near = float(cpp["os_near_degenerate_call_Second"]["expected"]["npv"])
    assert abs(near - npv) < 1e-5


def test_operator_splitting_put_is_call_minus_forward(cpp: dict[str, Any]) -> None:
    """The put is ``call - df (F1 - F2 - K)`` by construction, at both orders.

    ``callPutParityPrice`` never evaluates a put formula, so parity is exact
    rather than approximate — including on the degenerate branch, where the
    call itself is a different closed form.
    """
    for order in ("First", "Second"):
        call = cpp[f"os_call_k20_rho0.00_{order}"]
        put = cpp[f"os_put_k20_rho0.00_{order}"]
        inputs = call["inputs"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        df = risk_free.discount(maturity)

        def npv(
            case: dict[str, Any],
            order: str = order,
            today: Date = today,
            risk_free: YieldTermStructure = risk_free,
            maturity: Date = maturity,
        ) -> float:
            p1, p2 = _spread_legs(case["inputs"], today, risk_free)
            option = _spread_option(case["inputs"], maturity)
            option.set_pricing_engine(
                OperatorSplittingSpreadEngine(
                    p1,
                    p2,
                    float(case["inputs"]["correlation"]),
                    OperatorSplittingSpreadEngine.Order[order],
                )
            )
            return option.npv()

        expected_gap = df * (
            float(inputs["forward1"]) - float(inputs["forward2"]) - float(inputs["strike"])
        )
        tolerance.tight(npv(call) - npv(put), expected_gap, reason=f"parity {order}")


def test_operator_splitting_default_order_is_second(cpp: dict[str, Any]) -> None:
    """The C++ default is ``Order::Second``, and the two orders differ here."""
    default_npv = float(cpp["os_default_order_is_second"]["expected"]["npv"])
    second = float(cpp["os_call_k20_rho0.50_Second"]["expected"]["npv"])
    first = float(cpp["os_call_k20_rho0.50_First"]["expected"]["npv"])
    assert default_npv == second
    assert default_npv != first
    assert OperatorSplittingSpreadEngine.Order.Second == 1


# --- ChoiBasketEngine ---------------------------------------------------------


def _choi_engine(
    inputs: dict[str, Any], legs: list[GeneralizedBlackScholesProcess]
) -> ChoiBasketEngine:
    steps = int(inputs["max_nr_integration_steps"])
    return ChoiBasketEngine(
        legs,
        np.asarray(inputs["rho"], dtype=np.float64),
        float(inputs["lambda"]),
        MAX_NR_INTEGRATION_STEPS_UNBOUNDED if steps < 0 else steps,
        bool(inputs["calc_fwd_delta"]),
        bool(inputs["control_variate"]),
    )


def test_choi_basket_engine(cpp: dict[str, Any]) -> None:
    """Every priced case: golden market, knob sweeps, smaller baskets.

    Also asserts the ``additionalResults`` surface exactly — the count and each
    ``forwardDelta k`` — because the deltas are the only part of this engine
    that a wrong Householder rotation would leave visibly wrong while the NPV
    stayed plausible.
    """
    checked = 0
    for name in _cases(cpp, "choi_"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        legs = _merton_legs(inputs, today, risk_free)

        option = BasketOption(
            _basket_payoff(
                str(inputs["payoff_kind"]),
                str(inputs["option_type"]),
                float(inputs["strike"]),
                [float(w) for w in inputs["weights"]],
            ),
            EuropeanExercise(maturity),
        )
        option.set_pricing_engine(_choi_engine(inputs, legs))

        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        extra = option.additional_results()
        assert len(extra) == int(expected["additional_results_count"]), name
        for k, want in enumerate(expected.get("forward_deltas", [])):
            tolerance.tight(
                float(extra[f"forwardDelta {k}"]), float(want),
                reason=f"{name}: forwardDelta {k}",
            )
        checked += 1
    assert checked >= 18, f"expected the probe to carry choi cases, got {checked}"


def test_choi_control_variate_implies_forward_deltas(cpp: dict[str, Any]) -> None:
    """``calcFwdDelta_ = calcfwdDelta || controlVariate``.

    Turning the control variate on silently turns the deltas on as well, *and*
    moves the value; turning only the deltas on leaves the value untouched.
    Four combinations, so neither implication can be satisfied by accident.
    """
    ff = cpp["choi_knobs_ff"]["expected"]
    tf = cpp["choi_knobs_tf"]["expected"]
    ft = cpp["choi_knobs_ft"]["expected"]
    tt = cpp["choi_knobs_tt"]["expected"]

    assert int(ff["additional_results_count"]) == 0
    assert int(tf["additional_results_count"]) == 4
    assert int(ft["additional_results_count"]) == 4
    assert int(tt["additional_results_count"]) == 4

    # deltas alone do not change the value; the control variate does
    assert float(ff["npv"]) == float(tf["npv"])
    assert float(ft["npv"]) == float(tt["npv"])
    assert float(ff["npv"]) != float(ft["npv"])

    for name in ("choi_knobs_ff", "choi_knobs_ft"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        legs = _merton_legs(inputs, today, risk_free)
        option = BasketOption(
            _basket_payoff(
                str(inputs["payoff_kind"]),
                str(inputs["option_type"]),
                float(inputs["strike"]),
                [float(w) for w in inputs["weights"]],
            ),
            EuropeanExercise(maturity),
        )
        option.set_pricing_engine(_choi_engine(inputs, legs))
        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        assert len(option.additional_results()) == int(
            expected["additional_results_count"]
        )


def test_choi_lambda_and_max_steps_are_separate_knobs(cpp: dict[str, Any]) -> None:
    """Both knobs move the answer, and they interact.

    ``lambda`` sets the per-dimension quadrature orders; when their product
    exceeds ``maxNrIntegrationSteps`` C++ *rescales* lambda by 0.9 and retries
    rather than truncating. A port that ignored either would reproduce one
    sweep and fail the other.
    """
    lambdas = [
        float(cpp[n]["expected"]["npv"])
        for n in sorted(cpp)
        if n.startswith("choi_lambda")
    ]
    assert len(lambdas) == 5
    assert len(set(lambdas)) == 5

    caps = [
        float(cpp[n]["expected"]["npv"])
        for n in sorted(cpp)
        if n.startswith("choi_maxsteps")
    ]
    assert len(caps) == 4
    assert len(set(caps)) == 4


def test_choi_rewrites_a_spread_payoff_as_average_weights(cpp: dict[str, Any]) -> None:
    """``SpreadBasketPayoff`` is silently rewritten as ``AverageBasketPayoff{1, -1}``.

    Same market, same strike, two payoff spellings — identical prices. Pinned so
    the rewrite is observable rather than assumed.
    """
    for kind in ("call", "put"):
        spread = float(cpp[f"choi_spread_payoff_{kind}"]["expected"]["npv"])
        average = float(cpp[f"choi_2asset_{kind}"]["expected"]["npv"])
        assert spread == average

        inputs = cpp[f"choi_spread_payoff_{kind}"]["inputs"]
        assert str(inputs["payoff_kind"]) == "spread"
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        legs = _merton_legs(inputs, today, risk_free)
        option = BasketOption(
            SpreadBasketPayoff(
                PlainVanillaPayoff(
                    _option_type(str(inputs["option_type"])), float(inputs["strike"])
                )
            ),
            EuropeanExercise(maturity),
        )
        option.set_pricing_engine(_choi_engine(inputs, legs))
        tolerance.tight(option.npv(), spread, reason=f"choi spread {kind}")


@pytest.mark.parametrize(
    "case_name",
    [
        "choi_ctor_rejects_empty_processes",
        "choi_ctor_rejects_rho_size_mismatch",
        "choi_ctor_rejects_zero_lambda",
        "choi_ctor_rejects_negative_lambda",
        "choi_rejects_single_asset",
        "choi_rejects_wrong_weight_count",
        "choi_rejects_min_basket_payoff",
        "choi_rejects_american_exercise",
        "choi_rejects_unfittable_max_integration_steps",
    ],
)
def test_choi_guards(cpp: dict[str, Any], case_name: str) -> None:
    inputs, expected = cpp[case_name]["inputs"], cpp[case_name]["expected"]
    assert bool(expected["throws"])

    base = cpp["choi_golden_call"]["inputs"]
    today = _set_today(base)
    maturity = _date(str(base["maturity"]))
    risk_free = _flat_curve(today, float(base["risk_free_rate"]))
    legs = _merton_legs(base, today, risk_free)
    rho4 = np.asarray(base["rho"], dtype=np.float64)
    weights4 = [float(w) for w in base["weights"]]

    def run() -> None:
        match case_name:
            case "choi_ctor_rejects_empty_processes":
                ChoiBasketEngine([], np.zeros((0, 0), dtype=np.float64))
            case "choi_ctor_rejects_rho_size_mismatch":
                ChoiBasketEngine(legs, rho4[:3, :3])
            case "choi_ctor_rejects_zero_lambda":
                ChoiBasketEngine(legs, rho4, 0.0)
            case "choi_ctor_rejects_negative_lambda":
                ChoiBasketEngine(legs, rho4, -1.0)
            case "choi_rejects_single_asset":
                option = BasketOption(
                    AverageBasketPayoff(
                        PlainVanillaPayoff(OptionType.Call, 100.0), [1.0]
                    ),
                    EuropeanExercise(maturity),
                )
                option.set_pricing_engine(
                    ChoiBasketEngine(legs[:1], rho4[:1, :1])
                )
                option.npv()
            case "choi_rejects_wrong_weight_count":
                option = BasketOption(
                    AverageBasketPayoff(
                        PlainVanillaPayoff(OptionType.Call, 20.0), [1.0, 1.0, 1.0]
                    ),
                    EuropeanExercise(maturity),
                )
                option.set_pricing_engine(ChoiBasketEngine(legs, rho4))
                option.npv()
            case "choi_rejects_min_basket_payoff":
                option = BasketOption(
                    MinBasketPayoff(PlainVanillaPayoff(OptionType.Call, 20.0)),
                    EuropeanExercise(maturity),
                )
                option.set_pricing_engine(ChoiBasketEngine(legs, rho4))
                option.npv()
            case "choi_rejects_american_exercise":
                option = BasketOption(
                    AverageBasketPayoff(
                        PlainVanillaPayoff(OptionType.Call, 20.0), weights4
                    ),
                    AmericanExercise(today, maturity),
                )
                option.set_pricing_engine(ChoiBasketEngine(legs, rho4))
                option.npv()
            case _:
                option = BasketOption(
                    AverageBasketPayoff(
                        PlainVanillaPayoff(OptionType.Call, 20.0), weights4
                    ),
                    EuropeanExercise(maturity),
                )
                option.set_pricing_engine(ChoiBasketEngine(legs, rho4, 10.0, 0))
                option.npv()

    with pytest.raises(LibraryException, match=str(inputs["why"])):
        run()


# --- DengLiZhouBasketEngine ---------------------------------------------------


def test_deng_li_zhou_basket_engine(cpp: dict[str, Any]) -> None:
    """Every priced case: both branches, both payoff spellings, negative strikes."""
    checked = 0
    for name in _cases(cpp, "dlz_"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
        legs = _merton_legs(inputs, today, risk_free)

        option = BasketOption(
            _basket_payoff(
                str(inputs["payoff_kind"]),
                str(inputs["option_type"]),
                float(inputs["strike"]),
                [float(w) for w in inputs["weights"]],
            ),
            EuropeanExercise(maturity),
        )
        option.set_pricing_engine(
            DengLiZhouBasketEngine(legs, np.asarray(inputs["rho"], dtype=np.float64))
        )
        # A put is priced as ``max(0, call - fwd)`` where both terms are the
        # gross basket notional, so a deep-OTM put is a small number recovered
        # from two large ones. The floor is that cancellation scale:
        # ``8 eps sum_i |w_i| s_i``. For the 3-asset put it is 2.5e-13 against a
        # 0.0021 value, where a purely relative tier would demand 2e-15 — an
        # accuracy the subtraction cannot deliver in either library.
        gross = sum(
            abs(float(w)) * abs(float(s))
            for w, s in zip(inputs["weights"], inputs["spots"], strict=True)
        )
        tolerance.custom(
            option.npv(),
            float(expected["npv"]),
            abs_tol=max(1e-14, 8.0 * _EPS * gross),
            rel_tol=1e-12,
            reason=f"{name}: put/call gap is cancellation-limited at the basket notional",
        )
        assert len(option.additional_results()) == int(
            expected["additional_results_count"]
        )
        checked += 1
    assert checked >= 20, f"expected the probe to carry dlz cases, got {checked}"


def test_deng_li_zhou_covers_both_positive_weight_branches(cpp: dict[str, Any]) -> None:
    """``M == 1`` and ``M > 1`` are genuinely different code paths.

    With one positive weight the engine uses the raw leg; with more it collapses
    them onto a single log-normal proxy via C.F. Lo's WKB approximation and
    rebuilds the correlation row. The probe carries cases on both sides.
    """

    def positives(name: str) -> int:
        return sum(1 for w in cpp[name]["inputs"]["weights"] if float(w) > 0.0)

    assert positives("dlz_upstream_put") == 1
    assert positives("dlz_m2_call") == 2
    assert positives("dlz_m3_call") == 3
    # A weight of exactly 0 is NOT positive: lower_bound uses `w > 0`.
    assert positives("dlz_zero_weights_call") == 1
    assert any(float(w) == 0.0 for w in cpp["dlz_zero_weights_call"]["inputs"]["weights"])


def test_deng_li_zhou_negative_strike_adds_a_synthetic_asset(
    cpp: dict[str, Any],
) -> None:
    """A negative strike is priced as a zero-strike basket plus one extra leg.

    C++ appends ``(1.0, n, -K, dr0, 0.0)`` — weight 1, spot ``-K``, dividend
    discount ``dr0``, zero variance — uncorrelated with everything, then prices
    at ``max(0, K) == 0``. The value must therefore exceed the ``K == 0`` value
    for a call, which is what makes the branch observable in the price rather
    than only in the code.
    """
    km1 = float(cpp["dlz_negative_strike_call_km1"]["expected"]["npv"])
    km10 = float(cpp["dlz_negative_strike_call_km10"]["expected"]["npv"])
    k0 = float(cpp["dlz_upstream_call_k0"]["expected"]["npv"])
    assert k0 < km1 < km10

    inputs = cpp["dlz_negative_strike_call_km10"]["inputs"]
    assert float(inputs["strike"]) < 0.0
    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))
    legs = _merton_legs(inputs, today, risk_free)
    option = BasketOption(
        AverageBasketPayoff(
            PlainVanillaPayoff(OptionType.Call, float(inputs["strike"])),
            [float(w) for w in inputs["weights"]],
        ),
        EuropeanExercise(maturity),
    )
    option.set_pricing_engine(
        DengLiZhouBasketEngine(legs, np.asarray(inputs["rho"], dtype=np.float64))
    )
    tolerance.tight(option.npv(), km10, reason="dlz negative strike")


@pytest.mark.parametrize(
    "case_name",
    [
        "dlz_ctor_rejects_empty_processes",
        "dlz_ctor_rejects_rho_size_mismatch",
        "dlz_rejects_all_positive_weights",
        "dlz_rejects_all_negative_weights",
        "dlz_rejects_wrong_weight_count",
        "dlz_rejects_min_basket_payoff",
        "dlz_rejects_american_exercise",
    ],
)
def test_deng_li_zhou_guards(cpp: dict[str, Any], case_name: str) -> None:
    inputs, expected = cpp[case_name]["inputs"], cpp[case_name]["expected"]
    assert bool(expected["throws"])

    base = cpp["dlz_upstream_call"]["inputs"]
    today = _set_today(base)
    maturity = _date(str(base["maturity"]))
    risk_free = _flat_curve(today, float(base["risk_free_rate"]))
    legs = _merton_legs(base, today, risk_free)
    rho4 = np.asarray(base["rho"], dtype=np.float64)

    def price(weights: list[float]) -> None:
        option = BasketOption(
            AverageBasketPayoff(PlainVanillaPayoff(OptionType.Call, 5.0), weights),
            EuropeanExercise(maturity),
        )
        option.set_pricing_engine(DengLiZhouBasketEngine(legs, rho4))
        option.npv()

    def run() -> None:
        match case_name:
            case "dlz_ctor_rejects_empty_processes":
                DengLiZhouBasketEngine([], np.zeros((0, 0), dtype=np.float64))
            case "dlz_ctor_rejects_rho_size_mismatch":
                DengLiZhouBasketEngine(legs, rho4[:3, :3])
            case "dlz_rejects_all_positive_weights":
                price([1.0, 2.0, 3.0, 4.0])
            case "dlz_rejects_all_negative_weights":
                price([-1.0, -2.0, -3.0, -4.0])
            case "dlz_rejects_wrong_weight_count":
                price([1.0, -1.0])
            case "dlz_rejects_min_basket_payoff":
                option = BasketOption(
                    MaxBasketPayoff(PlainVanillaPayoff(OptionType.Call, 5.0)),
                    EuropeanExercise(maturity),
                )
                option.set_pricing_engine(DengLiZhouBasketEngine(legs, rho4))
                option.npv()
            case _:
                option = BasketOption(
                    AverageBasketPayoff(
                        PlainVanillaPayoff(OptionType.Call, 5.0),
                        [float(w) for w in base["weights"]],
                    ),
                    AmericanExercise(today, maturity),
                )
                option.set_pricing_engine(DengLiZhouBasketEngine(legs, rho4))
                option.npv()

    with pytest.raises(LibraryException, match=str(inputs["why"])):
        run()


# --- Monte Carlo: shared machinery -------------------------------------------


def test_pseudo_random_stream_matches_cpp(cpp: dict[str, Any]) -> None:
    """``PseudoRandom::make_sequence_generator(dim, seed)``, variate by variate.

    Everything below rests on this, so it is pinned first: a Monte Carlo
    failure is then never ambiguous between "wrong RNG" and "wrong pricer".

    The Mersenne-Twister uniforms *are* bit-identical — they are integer
    arithmetic. The Gaussians are not, and cannot be. Both libraries evaluate
    the same Acklam rational approximation with the same constants in the same
    Horner order (the optional Halley refinement is ``#ifdef``-ed out in C++),
    but the reference binary is compiled ``-O3`` on arm64, where clang
    contracts each ``a*r + b`` into an FMA. Ten of the fifteen pinned variates
    differ from PQuantLib's by at most 40 ulp — 7.4e-15 relative, five orders
    inside the TIGHT tier and thirteen orders inside any statistical band. It
    propagates no further: every downstream MC value below still agrees to
    2e-15.
    """
    for name in ("mc_rng_pseudo_random_dim3_seed42", "mc_rng_pseudo_random_dim6_seed7"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        generator = make_pseudo_random_rsg(int(inputs["dimension"]), int(inputs["seed"]))
        drawn: list[float] = []
        for _ in range(int(inputs["draws"])):
            drawn.extend(float(x) for x in generator.next_sequence().value)
        for got, want in zip(drawn, expected["gaussians"], strict=True):
            tolerance.tight(got, float(want), reason=name)
            assert abs(got - float(want)) <= 64 * math.ulp(abs(float(want)))


@pytest.mark.parametrize("n_assets", [2, 3])
@pytest.mark.parametrize("rho", [0.5, 0.0, -0.5])
def test_spectral_pseudo_sqrt_matches_cpp(
    cpp: dict[str, Any], n_assets: int, rho: float
) -> None:
    """``pseudoSqrt(rho, Spectral)`` bit-for-bit, not merely up to a Gram product.

    ``StochasticProcessArray`` uses ``dz = M @ dw`` for a *specific* ``dw``, so
    any other square root of the same correlation gives a statistically
    equivalent but path-wise different simulation. ``numpy.linalg.eigh`` returns
    eigenvalues ascending and eigenvectors with different signs, and produces a
    valid — but wrong — ``M``. This case is what forces
    ``SymmetricSchurDecomposition`` plus C++'s row normalisation.
    """
    name = f"mc_spectral_pseudo_sqrt_n{n_assets}_rho{rho:.2f}"
    inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
    corr = np.asarray(inputs["rho"], dtype=np.float64)
    root = _spectral_sqrt(corr)
    want = np.asarray(expected["pseudo_sqrt"], dtype=np.float64)
    assert root.shape == want.shape
    for i in range(root.shape[0]):
        for j in range(root.shape[1]):
            tolerance.tight(float(root[i, j]), float(want[i, j]), reason=name)
    # Any square root reproduces the correlation; only C++'s reproduces the paths.
    reconstructed = root @ root.T
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            tolerance.tight(float(reconstructed[i, j]), float(corr[i, j]))


def test_process_array_evolve_matches_cpp(cpp: dict[str, Any]) -> None:
    """``evolve(t, x, dt, dw)`` for a fixed ``dw`` — the correlation in action."""
    inputs, expected = cpp["mc_process_array_evolve"]["inputs"], cpp[
        "mc_process_array_evolve"
    ]["expected"]
    today = _set_today(inputs)
    array = _mc_process_array(inputs, today)

    x0: Array = array.initial_values()
    for got, want in zip(x0, expected["x0"], strict=True):
        tolerance.tight(float(got), float(want), reason="x0")

    evolved = array.evolve(
        float(inputs["t"]),
        x0,
        float(inputs["dt"]),
        np.asarray(inputs["dw"], dtype=np.float64),
    )
    for got, want in zip(evolved, expected["evolved"], strict=True):
        tolerance.tight(float(got), float(want), reason="evolve")


def test_multi_path_generator_first_draw_matches_cpp(cpp: dict[str, Any]) -> None:
    """The whole first ``MultiPath``, asset by asset and step by step."""
    inputs, expected = cpp["mc_multipath_first_draw"]["inputs"], cpp[
        "mc_multipath_first_draw"
    ]["expected"]
    today = _set_today(inputs)
    array = _mc_process_array(inputs, today)
    grid = TimeGrid.regular(float(inputs["horizon"]), int(inputs["time_steps"]))
    generator = make_pseudo_random_rsg(
        int(inputs["n_assets"]) * (len(grid) - 1), int(inputs["seed"])
    )
    path: MultiPath = MultiPathGenerator(array, grid, generator).next().value

    assert path.asset_number() == int(expected["asset_number"])
    assert path.path_size() == int(expected["path_size"])
    flat = [
        float(path[j].values[t])
        for j in range(path.asset_number())
        for t in range(path.path_size())
    ]
    for got, want in zip(flat, expected["values_row_major"], strict=True):
        tolerance.tight(got, float(want), reason="multipath")


# --- MCEuropeanBasketEngine ---------------------------------------------------


def _make_mc_european(
    inputs: dict[str, Any], array: StochasticProcessArray
) -> MakeMCEuropeanBasketEngine:
    maker = MakeMCEuropeanBasketEngine(array)
    if int(inputs.get("steps", -1)) >= 0:
        maker.with_steps(int(inputs["steps"]))
    if int(inputs.get("steps_per_year", -1)) >= 0:
        maker.with_steps_per_year(int(inputs["steps_per_year"]))
    if "samples" in inputs:
        maker.with_samples(int(inputs["samples"]))
    if "absolute_tolerance" in inputs:
        maker.with_absolute_tolerance(float(inputs["absolute_tolerance"]))
        maker.with_max_samples(int(inputs["max_samples"]))
    if bool(inputs.get("antithetic", False)):
        maker.with_antithetic_variate()
    return maker.with_seed(int(inputs["seed"]))


def test_mc_european_basket_engine(cpp: dict[str, Any]) -> None:
    """Every case: exact NPV *and* exact error estimate, never a band.

    Covers max / min / average / spread payoffs, 2 and 3 assets, fixed steps vs
    steps-per-year, antithetic on and off, two seeds, two sample counts, three
    correlations, and the adaptive ``withAbsoluteTolerance`` loop.
    """
    checked = 0
    for name in _cases(cpp, "mceb_"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        array = _mc_process_array(inputs, today)

        option = BasketOption(
            _basket_payoff(
                str(inputs["payoff_kind"]),
                str(inputs["option_type"]),
                float(inputs["strike"]),
                [float(w) for w in inputs.get("weights", [])],
            ),
            EuropeanExercise(maturity),
        )
        option.set_pricing_engine(_make_mc_european(inputs, array).engine())

        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        tolerance.tight(
            option.error_estimate(), float(expected["error_estimate"]),
            reason=f"{name}: error estimate",
        )
        checked += 1
    assert checked >= 14, f"expected the probe to carry mceb cases, got {checked}"


def test_mc_european_rejects_brownian_bridge(cpp: dict[str, Any]) -> None:
    """``withBrownianBridge(true)`` throws at pricing time.

    ``MultiPathGenerator`` has no Brownian-bridge implementation; the flag is
    accepted by the builder and then rejected by the generator. Reproduced
    rather than "supported": inventing a bridge here would silently price
    something C++ cannot.
    """
    inputs, expected = cpp["mceb_rejects_brownian_bridge"]["inputs"], cpp[
        "mceb_rejects_brownian_bridge"
    ]["expected"]
    assert bool(expected["throws"])

    base = cpp["mceb_max_call_2a"]["inputs"]
    today = _set_today(base)
    maturity = _date(str(base["maturity"]))
    array = _mc_process_array(base, today)
    option = BasketOption(
        MaxBasketPayoff(PlainVanillaPayoff(OptionType.Call, 100.0)),
        EuropeanExercise(maturity),
    )
    option.set_pricing_engine(
        MakeMCEuropeanBasketEngine(array)
        .with_steps_per_year(1)
        .with_samples(1023)
        .with_brownian_bridge()
        .with_seed(42)
        .engine()
    )
    with pytest.raises(LibraryException, match=str(inputs["why"])):
        option.npv()


@pytest.mark.parametrize(
    "case_name",
    [
        "mceb_rejects_no_steps",
        "mceb_rejects_overspecified_steps",
        "mceb_rejects_samples_and_tolerance",
        "mceb_rejects_tolerance_and_samples",
    ],
)
def test_mc_european_builder_guards(cpp: dict[str, Any], case_name: str) -> None:
    inputs, expected = cpp[case_name]["inputs"], cpp[case_name]["expected"]
    assert bool(expected["throws"])

    base = cpp["mceb_max_call_2a"]["inputs"]
    today = _set_today(base)
    array = _mc_process_array(base, today)

    def run() -> None:
        match case_name:
            case "mceb_rejects_no_steps":
                MakeMCEuropeanBasketEngine(array).with_samples(1023).with_seed(
                    42
                ).engine()
            case "mceb_rejects_overspecified_steps":
                MakeMCEuropeanBasketEngine(array).with_steps(2).with_steps_per_year(
                    1
                ).with_samples(1023).with_seed(42).engine()
            case "mceb_rejects_samples_and_tolerance":
                MakeMCEuropeanBasketEngine(array).with_steps_per_year(
                    1
                ).with_absolute_tolerance(0.02).with_samples(1023)
            case _:
                MakeMCEuropeanBasketEngine(array).with_steps_per_year(1).with_samples(
                    1023
                ).with_absolute_tolerance(0.02)

    with pytest.raises(LibraryException, match=str(inputs["why"])):
        run()


def test_mc_european_accepts_a_basket_payoff(cpp: dict[str, Any]) -> None:
    """The "non-basket payoff" guard is unreachable through ``BasketOption``.

    ``BasketOption`` only accepts a ``BasketPayoff``, so C++ records the happy
    path here rather than a throw. Pinned so a port does not invent a rejection
    C++ never performs.
    """
    assert bool(cpp["mceb_rejects_non_basket_payoff"]["expected"]["throws"]) is False


# --- MCAmericanBasketEngine ---------------------------------------------------


def test_american_basket_path_pricer_basis_system(cpp: dict[str, Any]) -> None:
    """The basis is multivariate over the *scaled asset vector*, plus the payoff.

    Size is ``multi_path_basis_system(n_assets, order, type).size() + 1`` — the
    payoff is appended as an extra regressor — and the values are pinned at a
    fixed state for orders 1, 2 and 3. This is the assertion that rules out the
    plausible-but-wrong reading "a univariate basis over the max of the basket".
    """
    for order in (1, 2, 3):
        name = f"mcab_path_pricer_basis_order{order}"
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        pricer = AmericanBasketPathPricer(
            int(inputs["asset_number"]),
            MaxBasketPayoff(
                PlainVanillaPayoff(
                    _option_type(str(inputs["option_type"])), float(inputs["strike"])
                )
            ),
            int(inputs["polynomial_order"]),
            PolynomialType[str(inputs["polynomial_type"])],
        )
        basis = pricer.basis_system()
        assert len(basis) == int(expected["basis_size"]), name
        state: Array = np.asarray(inputs["state"], dtype=np.float64)
        for k, f in enumerate(basis):
            tolerance.tight(
                f(state), float(expected["basis_values"][k]), reason=f"{name}[{k}]"
            )


@pytest.mark.parametrize(
    ("case_name", "polynomial_type", "payoff_is_basket"),
    [
        ("mcab_path_pricer_rejects_legendre", PolynomialType.Legendre, True),
        ("mcab_path_pricer_rejects_non_basket_payoff", PolynomialType.Monomial, False),
    ],
)
def test_american_basket_path_pricer_guards(
    cpp: dict[str, Any],
    case_name: str,
    polynomial_type: PolynomialType,
    payoff_is_basket: bool,
) -> None:
    inputs, expected = cpp[case_name]["inputs"], cpp[case_name]["expected"]
    assert bool(expected["throws"])
    payoff = (
        MaxBasketPayoff(PlainVanillaPayoff(OptionType.Put, 100.0))
        if payoff_is_basket
        else PlainVanillaPayoff(OptionType.Put, 100.0)
    )
    with pytest.raises(LibraryException, match=str(inputs["why"])):
        AmericanBasketPathPricer(2, payoff, 2, polynomial_type)


def test_lsm_calibration_seed_is_null_size(cpp: dict[str, Any]) -> None:
    """The calibration seed is ``Null<Size>()``, *not* ``seed + 1768237423``.

    ``MCLongstaffSchwartzEngine`` writes

        seedCalibration_ = (seedCalibration != Null<Real>())
                               ? seedCalibration
                               : (seed == 0 ? 0 : seed + 1768237423L);

    and the condition is always true — ``seedCalibration`` is a ``BigNatural``
    while ``Null<Real>()`` is ``FLT_MAX`` (3.4e38), so they can never compare
    equal. The ``seed + 1768237423`` branch is dead code, and what reaches the
    calibration generator is the parameter default ``Null<Size>()``, which for
    an integral type is ``INT_MAX == 2147483647``, independent of the pricing
    seed.

    This is reproduced rather than corrected: diverging would make the two
    libraries price differently. The check below is not cosmetic — the
    reference's ``hand_npv``, rebuilt here from the seed, equals the engine's
    NPV to 16 digits, whereas the ``seed + 1768237423`` reading gives 3.5591
    against C++'s 3.7504, a 5% error.
    """
    inputs, expected = cpp["mcab_lsm_internals"]["inputs"], cpp["mcab_lsm_internals"][
        "expected"
    ]
    assert int(inputs["seed_calibration"]) == 2147483647
    assert int(inputs["seed"]) == 1

    today = _set_today(inputs)
    maturity = _date(str(inputs["maturity"]))
    array = _mc_process_array(inputs, today)
    risk_free = _flat_curve(today, float(inputs["risk_free_rate"]))

    grid = TimeGrid.with_mandatory_and_steps(
        [array.time(maturity)], int(inputs["steps"])
    )
    for got, want in zip(list(grid), expected["time_grid"], strict=True):
        tolerance.tight(float(got), float(want), reason="time grid")
    assert len(grid) == int(expected["len"])

    early_pricer = AmericanBasketPathPricer(
        2,
        MaxBasketPayoff(
            PlainVanillaPayoff(
                _option_type(str(inputs["option_type"])), float(inputs["strike"])
            )
        ),
        int(inputs["polynomial_order"]),
        PolynomialType[str(inputs["polynomial_type"])],
    )
    lsm = LongstaffSchwartzPathPricer[MultiPath, Array](grid, early_pricer, risk_free)

    dimension = 2 * (len(grid) - 1)
    calibration = MonteCarloModel[MultiPath](
        path_generator=MultiPathGenerator(
            array,
            grid,
            make_pseudo_random_rsg(dimension, int(inputs["seed_calibration"])),
        ),
        path_pricer=lsm,
        sample_accumulator=GeneralStatistics(),
        antithetic_variate=False,
    )
    calibration.add_samples(int(inputs["calibration_samples"]))
    lsm.calibrate()

    # The regression coefficients are the one quantity here that is not TIGHT.
    # The monomial design matrix over states clustered near 1 is ill
    # conditioned — the fitted coefficients reach ~1e3 while the values being
    # fitted are ~1e1, so the normal system cancels by ~1e2 — and the two
    # libraries solve it with different SVD routines (QuantLib's own Jacobi
    # implementation vs LAPACK ``gelsd`` behind ``numpy.linalg.lstsq``). Their
    # agreement is bounded by ``cond(A) * eps``, measured at 1.0e-11 relative
    # across all seven vectors. 1e-9 leaves two orders in hand and is still
    # sharp: a wrong basis ordering or a wrong ITM filter moves these
    # coefficients by 100%, not by 1e-9. The quantity that matters — the NPV
    # below — is asserted at TIGHT.
    for k, coefficients in enumerate(lsm.coefficients()):
        for got, want in zip(coefficients, expected[f"coeff_{k}"], strict=True):
            tolerance.custom(
                float(got),
                float(want),
                abs_tol=1e-9,
                rel_tol=1e-9,
                reason=f"coeff_{k}: ill-conditioned least squares, two SVD routines",
            )

    accumulator = GeneralStatistics()
    pricing = MonteCarloModel[MultiPath](
        path_generator=MultiPathGenerator(
            array, grid, make_pseudo_random_rsg(dimension, int(inputs["seed"]))
        ),
        path_pricer=lsm,
        sample_accumulator=accumulator,
        antithetic_variate=False,
    )
    pricing.add_samples(2048)

    tolerance.tight(accumulator.mean(), float(expected["hand_npv"]), reason="hand npv")
    tolerance.tight(
        accumulator.error_estimate(), float(expected["hand_error_estimate"]),
        reason="hand error",
    )
    # ... and the hand-assembled run reproduces the engine exactly.
    tolerance.tight(
        accumulator.mean(), float(cpp["mcab_max_put_2a"]["expected"]["npv"]),
        reason="hand run == engine",
    )


def test_mc_american_basket_engine(cpp: dict[str, Any]) -> None:
    """Every case: exact NPV, error estimate and exercise probability.

    Sweeps the two knobs C++ exposes on this engine — ``polynomialOrder`` (1, 2,
    3) and ``polynomialType`` (Monomial, Laguerre, Hermite, Chebyshev2nd) — plus
    payoff kind, asset count, step count, antithetic and seed.
    """
    checked = 0
    for name in _cases(cpp, "mcab_"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        maturity = _date(str(inputs["maturity"]))
        array = _mc_process_array(inputs, today)

        maker = (
            MakeMCAmericanBasketEngine(array)
            .with_steps(int(inputs["steps"]))
            .with_samples(int(inputs["samples"]))
            .with_calibration_samples(int(inputs["calibration_samples"]))
            .with_polynomial_order(int(inputs["polynomial_order"]))
            .with_basis_system(PolynomialType[str(inputs["polynomial_type"])])
            .with_seed(int(inputs["seed"]))
        )
        if bool(inputs["antithetic"]):
            maker.with_antithetic_variate()

        option = BasketOption(
            _basket_payoff(
                str(inputs["payoff_kind"]),
                str(inputs["option_type"]),
                float(inputs["strike"]),
                [float(w) for w in inputs.get("weights", [])],
            ),
            AmericanExercise(today, maturity),
        )
        option.set_pricing_engine(maker.engine())

        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        tolerance.tight(
            option.error_estimate(), float(expected["error_estimate"]),
            reason=f"{name}: error estimate",
        )
        extra = option.additional_results()
        assert len(extra) == int(expected["additional_results_count"]), name
        tolerance.tight(
            float(extra["exerciseProbability"]),
            float(expected["exercise_probability"]),
            reason=f"{name}: exercise probability",
        )
        checked += 1
    assert checked >= 13, f"expected the probe to carry mcab cases, got {checked}"


def test_mc_american_polynomial_knobs_move_the_answer(cpp: dict[str, Any]) -> None:
    """Order and basis family are real knobs.

    All five configurations share every other input, so a port that discarded
    either argument would give one value five times. (One engine in an earlier
    wave did exactly that — accepted a constructor argument and dropped it.)
    """
    values = {
        name: float(cpp[name]["expected"]["npv"])
        for name in (
            "mcab_max_put_2a",
            "mcab_max_put_2a_order1",
            "mcab_max_put_2a_order3",
            "mcab_max_put_2a_laguerre",
            "mcab_max_put_2a_hermite",
            "mcab_max_put_2a_chebyshev2nd",
        )
    }
    assert len(set(values.values())) == len(values)


def test_mc_american_rejects_european_exercise(cpp: dict[str, Any]) -> None:
    inputs, expected = cpp["mcab_rejects_european_exercise"]["inputs"], cpp[
        "mcab_rejects_european_exercise"
    ]["expected"]
    assert bool(expected["throws"])

    base = cpp["mcab_max_put_2a"]["inputs"]
    today = _set_today(base)
    maturity = _date(str(base["maturity"]))
    array = _mc_process_array(base, today)
    option = BasketOption(
        MaxBasketPayoff(PlainVanillaPayoff(OptionType.Put, 100.0)),
        EuropeanExercise(maturity),
    )
    option.set_pricing_engine(
        MakeMCAmericanBasketEngine(array)
        .with_steps(4)
        .with_samples(1023)
        .with_calibration_samples(256)
        .with_seed(1)
        .engine()
    )
    with pytest.raises(LibraryException, match=str(inputs["why"])):
        option.npv()
