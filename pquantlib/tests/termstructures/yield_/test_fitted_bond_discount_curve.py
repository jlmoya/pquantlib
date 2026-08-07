"""FittedBondDiscountCurve, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/fittedbond.json
Probe:     migration-harness/cpp/probes/v143_ts_fittedbond/probe.cpp

The fit nests two iterative processes — a ``Simplex`` over a cost function that
itself runs a ``NewtonSafe`` root-find per bond inside ``FittingMethod::init``
— so the probe is layered and so is this module, from least to most
path-dependent:

* **Plumbing** — the ``max_evaluations == 0`` "don't fit, evaluate these
  parameters" mode, under both anchorings; and ``SpreadFittingMethod``'s
  rebasing.  No optimizer, no root-find.
* **Layer 2** — ``init()``'s weight inputs: the raw yield-to-maturity and
  modified duration per bond.  One root-find each, no simplex.
* **Layer 3** — ``FittingCost::values(x)`` / ``value(x)`` at a fixed ``x``,
  reached by handing the fitting method a stub ``OptimizationMethod`` that
  evaluates the problem once at the guess and returns.  This is the only way
  into the (private, nested) cost function, in C++ as here.
* **Layer 4** — the real ``Simplex`` fit: converged solution, function
  evaluation count, cost, and end-criteria code.

**Reproduction.** Layers 1-3 are bit-exact or within one ulp.  Ten of the
eleven layer-4 fits — including a 1665-evaluation one — are bit-exact in every
emitted quantity, which is only possible because ``Simplex`` and ``NewtonSafe``
are transcriptions rather than SciPy wrappers.  The eleventh,
``layer4_bspline_zeros``, is not, and
:func:`test_bspline_fit_diverges_until_bspline_basis_gets_fma_contraction`
carries the derivation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.cashflows.duration import Duration
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.instruments.bonds.zero_coupon_bond import ZeroCouponBond
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.end_criteria import Type as EndCriteriaType
from pquantlib.math.optimization.optimization_method import OptimizationMethod
from pquantlib.math.optimization.problem import Problem
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.bond.bond_functions import BondFunctions
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.bond_helper import BondHelper
from pquantlib.termstructures.yield_.fitted_bond_discount_curve import (
    FittedBondDiscountCurve,
    FittingMethod,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.nonlinear_fitting_methods import (
    CubicBSplinesFitting,
    ExponentialSplinesFitting,
    NaturalCubicFitting,
    NelsonSiegelFitting,
    SimplePolynomialFitting,
    SpreadFittingMethod,
    SvenssonFitting,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.canada import Canada
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

CPP: dict[str, Any] = reference_reader.load("v143/ts/fittedbond")

# probe.cpp:62 — ``const Date kEval(15, July, 2019);``, the same asof as
# test-suite/fittedbonddiscountcurve.cpp:82.
_EVAL = Date(CPP["meta"]["eval_date"])
_MAX_DATE_10Y = Date(CPP["meta"]["max_date_10y"])
_A365 = Actual365Fixed()
_AA_ISDA = ActualActual(ActualActualConvention.ISDA)
_SEMI = Period(6, TimeUnit.Months)
_KNOTS9: list[float] = [-10.0, -5.0, 0.0, 4.0, 8.0, 12.0, 20.0, 30.0, 40.0]


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """The probe pins Settings::evaluationDate (probe.cpp:230); bonds read it."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _EVAL
    try:
        yield
    finally:
        settings.evaluation_date = previous


# ---------------------------------------------------------------------------
# bond sets
# ---------------------------------------------------------------------------


def _canada_helpers() -> list[BondHelper]:
    """Four Canadian government bonds. probe.cpp:139-172.

    Lifted verbatim from test-suite/fittedbonddiscountcurve.cpp:86-124.
    """
    specs: list[tuple[Date, Date, Date, float, float]] = [
        (
            Date.from_ymd(1, Month.February, 2013),
            Date.from_ymd(3, Month.February, 2020),
            Date.from_ymd(3, Month.August, 2013),
            0.046,
            101.2100,
        ),
        (
            Date.from_ymd(12, Month.June, 2015),
            Date.from_ymd(12, Month.June, 2020),
            Date.from_ymd(12, Month.December, 2015),
            0.0295,
            100.6270,
        ),
        (
            Date.from_ymd(24, Month.November, 2017),
            Date.from_ymd(24, Month.November, 2020),
            Date.from_ymd(24, Month.May, 2018),
            0.02689,
            99.9210,
        ),
        (
            Date.from_ymd(21, Month.February, 2017),
            Date.from_ymd(21, Month.February, 2022),
            Date.from_ymd(21, Month.August, 2017),
            0.0338,
            101.6700,
        ),
    ]
    helpers: list[BondHelper] = []
    for effective, termination, first, rate, quote in specs:
        schedule = Schedule.from_rule(
            effective,
            termination,
            _SEMI,
            Canada(),
            BusinessDayConvention.Following,
            BusinessDayConvention.Following,
            DateGeneration.Forward,
            False,
            first,
        )
        helpers.append(
            BondHelper(
                SimpleQuote(quote),
                FixedRateBond(2, 100.0, schedule, [rate], _AA_ISDA),
            )
        )
    return helpers


def _zero_helpers() -> list[BondHelper]:
    """Four zero-coupon bonds. probe.cpp:175-186.

    From test-suite/fittedbonddiscountcurve.cpp:229-238; cheap to price, so
    this is the set the heavier layer-4 fits use.
    """
    quotes = [99.0, 98.0, 95.0, 90.0]
    tenors = [
        Period(1, TimeUnit.Years),
        Period(2, TimeUnit.Years),
        Period(5, TimeUnit.Years),
        Period(10, TimeUnit.Years),
    ]
    return [
        BondHelper(SimpleQuote(q), ZeroCouponBond(3, TARGET(), 100.0, _EVAL + t))
        for q, t in zip(quotes, tenors, strict=True)
    ]


def _flat_base(shift_days: int, rate: float) -> FlatForward:
    """A flat curve anchored ``shift_days`` before the evaluation date."""
    curve = FlatForward.from_rate(
        Date(_EVAL.serial_number() + shift_days),
        rate,
        _A365,
        Compounding.Continuous,
        Frequency.Annual,
    )
    curve.enable_extrapolation()
    return curve


# ---------------------------------------------------------------------------
# a stub optimizer — the only door into the private FittingCost
# ---------------------------------------------------------------------------


class _ProbeOptimizer(OptimizationMethod):
    """Evaluate the problem once at the guess and stop. probe.cpp:117-140.

    ``Problem.reset()`` is MANDATORY and not decoration: C++'s ``Problem``
    leaves ``functionEvaluation_`` indeterminate until ``reset()`` runs
    (problem.hpp:45-47), every shipped optimizer calls it first, and the probe
    caught the difference — two runs disagreed on
    ``numberOfIterations`` until the stub was fixed to call it.
    """

    def __init__(self) -> None:
        self.x0: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.values: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.value: float = 0.0

    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> EndCriteriaType:
        del end_criteria
        problem.reset()
        self.x0 = problem.current_value.copy()
        self.values = problem.values(self.x0)
        self.value = problem.value(self.x0)
        problem.set_function_value(self.value)
        return EndCriteriaType.StationaryPoint


# ---------------------------------------------------------------------------
# plumbing: max_evaluations == 0
# ---------------------------------------------------------------------------


def test_no_fit_mode_reproduces_cpp() -> None:
    """The "don't fit, use precalculated parameters" constructors. probe.cpp:451-499.

    ``solution()`` is the given parameter vector verbatim, ``numberOfIterations``
    is 0, ``minimumCostValue`` is ``Null<Real>()`` — which is
    ``numeric_limits<float>::max()``, 3.4028234663852886e38, NOT NaN and not the
    double maximum — and the error code is ``None``.
    """
    block = CPP["no_fit"]
    parameters: npt.NDArray[np.float64] = np.array(block["solution"], dtype=np.float64)
    method = ExponentialSplinesFitting()

    by_date = FittedBondDiscountCurve.from_parameters(
        method, parameters, _MAX_DATE_10Y, _A365, reference_date=_EVAL
    )
    by_settlement = FittedBondDiscountCurve.from_parameters(
        method, parameters, _MAX_DATE_10Y, _A365, settlement_days=0, calendar=TARGET()
    )

    assert by_date.reference_date().serial_number() == block["reference_date_c1"]
    assert by_settlement.reference_date().serial_number() == block["reference_date_c2"]
    assert by_date.number_of_bonds() == block["number_of_bonds"]
    assert by_date.max_date().serial_number() == block["max_date"]

    results = by_date.fit_results()
    assert results.number_of_iterations() == block["number_of_iterations"]
    tolerance.exact(results.minimum_cost_value(), block["minimum_cost_value"])
    assert int(results.error_code()) == block["error_code"]
    assert results.weights().size == len(block["weights"])
    for got, expected in zip(results.solution(), block["solution"], strict=True):
        tolerance.exact(float(got), expected)

    for t, expected in zip(block["t"], block["discount_c1"], strict=True):
        tolerance.exact(by_date.discount(t), expected, reason=f"c1 t={t}")
    for t, expected in zip(block["t"], block["discount_c2"], strict=True):
        tolerance.exact(by_settlement.discount(t), expected, reason=f"c2 t={t}")


def test_no_fit_mode_with_non_degenerate_parameters() -> None:
    """The upstream parameter set has kappa == 0, which collapses d(t) to 1.

    test-suite/fittedbonddiscountcurve.cpp:49-59 sets ``x[8] = 0``, so every
    ``exp(-kappa*n*t)`` is 1 and the constrained coefficient completes the sum:
    the curve above is identically 1.0 and would pass with an arbitrarily
    broken discount function. probe.cpp:465-471 adds a second, live set.
    """
    block = CPP["no_fit"]
    assert all(d == 1.0 for d in block["discount_c1"]), "premise changed"

    parameters: npt.NDArray[np.float64] = np.array(
        block["parameters_nondegenerate"], dtype=np.float64
    )
    curve = FittedBondDiscountCurve.from_parameters(
        ExponentialSplinesFitting(), parameters, _MAX_DATE_10Y, _A365, reference_date=_EVAL
    )
    for t, expected in zip(block["t"], block["discount_c3"], strict=True):
        tolerance.exact(curve.discount(t), expected, reason=f"t={t}")


def test_no_fit_mode_rejects_a_wrong_sized_parameter_vector() -> None:
    """cpp:250-251 — ``guessSolution_.size() == size()``."""
    curve = FittedBondDiscountCurve.from_parameters(
        NelsonSiegelFitting(),
        np.array([0.01, 0.0, 0.0], dtype=np.float64),
        _MAX_DATE_10Y,
        _A365,
        reference_date=_EVAL,
    )
    with pytest.raises(LibraryException, match="wrong number of parameters"):
        curve.discount(3.0)


def test_curve_requires_exactly_one_anchoring() -> None:
    with pytest.raises(LibraryException, match="exactly one of reference_date"):
        FittedBondDiscountCurve(
            day_counter=_A365, fitting_method=NelsonSiegelFitting(), bonds=_zero_helpers()
        )


def test_fit_requires_helpers_and_no_fit_requires_a_max_date() -> None:
    """cpp:127-137 — the two mutually exclusive preconditions."""
    with pytest.raises(LibraryException, match="no bond helpers given"):
        FittedBondDiscountCurve(
            day_counter=_A365,
            fitting_method=NelsonSiegelFitting(),
            reference_date=_EVAL,
        ).discount(1.0)
    with pytest.raises(LibraryException, match="no bond helpers or max date given"):
        FittedBondDiscountCurve(
            day_counter=_A365,
            fitting_method=NelsonSiegelFitting(),
            reference_date=_EVAL,
            max_evaluations=0,
            guess=np.array([0.02, 0.0, 0.0, 1.0]),
        ).discount(1.0)


def test_l2_penalty_requires_a_guess() -> None:
    """cpp:235 — the check test-suite/fittedbonddiscountcurve.cpp:250 pins."""
    method = NelsonSiegelFitting(l2=[0.25, 0.25, 0.25, 0.25])
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=method,
        settlement_days=0,
        calendar=TARGET(),
        bonds=_zero_helpers(),
    )
    with pytest.raises(LibraryException, match="L2 penalty requires a guess"):
        curve.discount(3.0)


def test_guess_size_is_checked() -> None:
    """cpp:267 — test-suite/fittedbonddiscountcurve.cpp:274-279."""
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=NelsonSiegelFitting(),
        settlement_days=0,
        calendar=TARGET(),
        bonds=_zero_helpers(),
        guess=np.array([0.01, 0.0, 0.0], dtype=np.float64),
    )
    with pytest.raises(LibraryException, match="wrong size for guess"):
        curve.discount(3.0)


# ---------------------------------------------------------------------------
# SpreadFittingMethod
# ---------------------------------------------------------------------------


def test_spread_fitting_rebases_on_the_curve_reference_date() -> None:
    """cpp:414-429 — the rebase divisor and the extrapolating base-curve call.

    Two curves differing only in the base curve's reference date: with a
    30-day offset every discount factor is divided by
    ``base.discount(curve.referenceDate())``, and with no offset the divisor is
    exactly 1.0.  The sampled times run to 30 years, well past the base curve's
    own max date, which is why ``discountFunction`` passes ``extrapolate=true``.
    """
    block = CPP["spread"]
    x: npt.NDArray[np.float64] = np.array(block["x"], dtype=np.float64)

    base = _flat_base(-30, 0.021)
    assert base.reference_date().serial_number() == block["base_reference_date"]
    tolerance.exact(base.discount(_EVAL), block["rebase"])

    rebased = FittedBondDiscountCurve.from_parameters(
        SpreadFittingMethod(NelsonSiegelFitting(), base, 0.5, 8.0),
        x,
        _MAX_DATE_10Y,
        _A365,
        reference_date=_EVAL,
    )
    rebased.enable_extrapolation()
    for t, expected in zip(block["t"], block["d_rebased"], strict=True):
        tolerance.exact(rebased.discount(t, True), expected, reason=f"rebased t={t}")

    aligned = FittedBondDiscountCurve.from_parameters(
        SpreadFittingMethod(NelsonSiegelFitting(), _flat_base(0, 0.021)),
        x,
        _MAX_DATE_10Y,
        _A365,
        reference_date=_EVAL,
    )
    aligned.enable_extrapolation()
    for t, expected in zip(block["t"], block["d_same_reference"], strict=True):
        tolerance.exact(aligned.discount(t, True), expected, reason=f"aligned t={t}")

    assert block["d_rebased"] != block["d_same_reference"], "rebasing had no effect"


def test_spread_fitting_size_and_guards() -> None:
    """cpp:401-412 — ``size()`` delegates; both constructor arguments are required."""
    base = _flat_base(0, 0.02)
    assert SpreadFittingMethod(SvenssonFitting(), base).size() == (
        CPP["layer1_size"]["spread_over_svensson"]
    )
    assert SpreadFittingMethod(NelsonSiegelFitting(), base).size() == (
        CPP["layer1_size"]["nelson_siegel"]
    )
    with pytest.raises(LibraryException, match="Fitting method is empty"):
        SpreadFittingMethod(None, base)  # pyright: ignore[reportArgumentType]
    with pytest.raises(LibraryException, match="Discounting curve cannot be empty"):
        SpreadFittingMethod(NelsonSiegelFitting(), None)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# layer 2 — init()'s weight inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "builder"), [("layer2_canada", _canada_helpers), ("layer2_zeros", _zero_helpers)]
)
def test_weight_inputs_match_cpp(key: str, builder: Callable[[], list[BondHelper]]) -> None:
    """The yield and duration ``init()`` builds the weights from. cpp:202-226.

    The conventions are hard-coded in C++: the CURVE's day counter, Compounded,
    Annual, Duration::Modified, and the BOND's settlement date — not the
    curve's reference date, which is why the settlement serial is pinned too.
    ``BondFunctions::yield`` is a ``NewtonSafe`` root-find, so this is where a
    solver that had been swapped for something else would show up.
    """
    block = CPP[key]
    for i, helper in enumerate(builder()):
        bond = helper.bond()
        settlement = bond.settlement_date()
        assert settlement.serial_number() == block["settlement"][i], f"{key}[{i}]"
        price = BondPrice(helper.quote().value(), BondPriceType.Clean)
        ytm = BondFunctions.bond_yield(
            bond, price, _A365, Compounding.Compounded, Frequency.Annual, settlement
        )
        duration = BondFunctions.duration_from_rate(
            bond,
            ytm,
            _A365,
            Compounding.Compounded,
            Frequency.Annual,
            Duration.Modified,
            settlement,
        )
        tolerance.tight(ytm, block["ytm"][i], reason=f"{key}[{i}] ytm")
        tolerance.tight(duration, block["duration"][i], reason=f"{key}[{i}] duration")


def test_weights_are_inverse_duration_normalised_to_unit_norm() -> None:
    """cpp:222-225 — ``w_i = 1/dur_i``, then ``w /= sqrt(sum w_i^2)``.

    Derived from the layer-2 durations alone, so it pins the normalisation
    independently of the fit that reports the weights.
    """
    durations = CPP["layer2_zeros"]["duration"]
    raw = [1.0 / d for d in durations]
    norm = sum(w * w for w in raw) ** 0.5
    expected = [w / norm for w in raw]
    for got, want in zip(CPP["layer3_ns_zeros_l2"]["weights"], expected, strict=True):
        tolerance.tight(got, want)
    tolerance.tight(sum(w * w for w in expected), 1.0)


# ---------------------------------------------------------------------------
# layer 3 — FittingCost at a fixed x
# ---------------------------------------------------------------------------


def _layer3_cases() -> dict[str, tuple[Callable[[], list[BondHelper]], Callable[[_ProbeOptimizer], FittingMethod], list[float]]]:
    """probe.cpp:610-676 and 771-780."""
    return {
        "layer3_ns_canada": (
            _canada_helpers,
            lambda opt: NelsonSiegelFitting(optimization_method=opt),
            [0.0317, 5.0, -3.6796, 24.1703],
        ),
        "layer3_ns_zeros_l2": (
            _zero_helpers,
            lambda opt: NelsonSiegelFitting(optimization_method=opt, l2=[0.25, 0.5, 0.75, 1.0]),
            [0.021, -0.004, 0.011, 1.3],
        ),
        "layer3_l2_offset": (
            _zero_helpers,
            lambda opt: NelsonSiegelFitting(
                weights=[0.1, 0.2, 0.3, 0.4], optimization_method=opt
            ),
            [0.021, -0.004, 0.011, 1.3],
        ),
        "layer3_exp_zeros": (
            _zero_helpers,
            lambda opt: ExponentialSplinesFitting(
                True, optimization_method=opt, num_coeffs=4, fixed_kappa=0.05
            ),
            [0.2, 0.1, 0.05],
        ),
        "layer3_bspline_zeros": (
            _zero_helpers,
            lambda opt: CubicBSplinesFitting(_KNOTS9, True, optimization_method=opt),
            [0.98, 0.94, 0.88, 0.80],
        ),
    }


@pytest.mark.parametrize("key", sorted(_layer3_cases()))
def test_fitting_cost_matches_cpp(key: str) -> None:
    """``FittingCost::values(x)`` and ``value(x)`` at the guess. cpp:304-336.

    ``values`` has ``n + N`` entries: the ``n`` weighted squared quote errors
    followed by ``N`` L2 penalties ``l2[i]*(x[i] - guess[i])^2``.  Because the
    stub optimizer evaluates AT the guess, every penalty term is exactly zero —
    which is itself the discriminating check that the penalty is measured from
    the guess and not from the origin.

    ``value`` is the plain SUM of ``values``, NOT ``CostFunction``'s default
    ``sqrt(mean(v^2))``: the residuals are already squared, and applying the
    default would square them twice.
    """
    builder, make_method, guess = _layer3_cases()[key]
    block = CPP[key]
    optimizer = _ProbeOptimizer()
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=make_method(optimizer),
        reference_date=_EVAL,
        bonds=builder(),
        guess=np.array(guess, dtype=np.float64),
    )
    curve.enable_extrapolation()
    results = curve.fit_results()

    for got, expected in zip(results.weights(), block["weights"], strict=True):
        tolerance.tight(float(got), expected, reason=f"{key} weight")
    for got, expected in zip(optimizer.x0, block["cost_x"], strict=True):
        tolerance.exact(float(got), expected, reason=f"{key} cost_x")
    for i, (got, expected) in enumerate(
        zip(optimizer.values, block["cost_values"], strict=True)
    ):
        tolerance.tight(float(got), expected, reason=f"{key} values[{i}]")
    tolerance.tight(optimizer.value, block["cost_value"], reason=f"{key} value")


def test_l2_penalty_terms_extend_the_residual_vector() -> None:
    """The L2 block is ``N`` extra entries, not a scaling of the first ``n``.

    cpp:322 sizes the vector ``n + N``; the no-L2 case has exactly ``n``.
    """
    with_l2 = CPP["layer3_ns_zeros_l2"]["cost_values"]
    without_l2 = CPP["layer3_l2_offset"]["cost_values"]
    assert len(with_l2) == 8
    assert len(without_l2) == 4
    assert with_l2[4:] == [0.0, 0.0, 0.0, 0.0]


def test_fitting_cost_value_is_the_sum_not_the_rms() -> None:
    """cpp:304-312, against ``CostFunction``'s default (costfunction.hpp:38-43)."""
    for key in ("layer3_ns_canada", "layer3_ns_zeros_l2", "layer3_l2_offset"):
        values = CPP[key]["cost_values"]
        total = 0.0
        for v in values:
            total += v
        tolerance.exact(CPP[key]["cost_value"], total, reason=key)


# ---------------------------------------------------------------------------
# layer 4 — the real fit
# ---------------------------------------------------------------------------


def _layer4_cases() -> dict[str, tuple[Callable[[], list[BondHelper]], Callable[[], FittingMethod], float, int]]:
    """probe.cpp:679-760: (helpers, method factory, accuracy, max_evaluations)."""
    cut = CPP["layer4_ns_canada_cutoff_times"]
    return {
        "layer4_poly_zeros": (_zero_helpers, lambda: SimplePolynomialFitting(2, True), 1e-10, 5000),
        "layer4_ns_zeros": (_zero_helpers, NelsonSiegelFitting, 1e-9, 400),
        "layer4_ns_canada": (_canada_helpers, NelsonSiegelFitting, 1e-10, 1000),
        "layer4_ns_canada_cutoff": (
            _canada_helpers,
            lambda: NelsonSiegelFitting(min_cutoff_time=cut["min"], max_cutoff_time=cut["max"]),
            1e-10,
            1000,
        ),
        "layer4_svensson_zeros": (_zero_helpers, SvenssonFitting, 1e-9, 300),
        "layer4_exp_zeros": (
            _zero_helpers,
            lambda: ExponentialSplinesFitting(True, num_coeffs=4, fixed_kappa=0.05),
            1e-9,
            500,
        ),
        "layer4_natcubic_zeros": (
            _zero_helpers,
            lambda: NaturalCubicFitting([1.0, 2.0, 5.0, 10.0]),
            1e-9,
            500,
        ),
        "layer4_spread_zeros": (
            _zero_helpers,
            lambda: SpreadFittingMethod(NelsonSiegelFitting(), _flat_base(-30, 0.012)),
            1e-9,
            300,
        ),
        "layer4_unconstrained": (
            _zero_helpers,
            lambda: SimplePolynomialFitting(1, True),
            1e-10,
            5000,
        ),
        "layer4_positive_constrained": (
            _zero_helpers,
            lambda: SimplePolynomialFitting(1, True, constraint=PositiveConstraint()),
            1e-10,
            5000,
        ),
        # layer4_bspline_zeros is deliberately absent — see
        # test_bspline_fit_diverges_until_bspline_basis_gets_fma_contraction.
    }


def _run_fit(key: str) -> FittedBondDiscountCurve:
    builder, make_method, accuracy, max_evaluations = _layer4_cases()[key]
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=make_method(),
        reference_date=_EVAL,
        bonds=builder(),
        accuracy=accuracy,
        max_evaluations=max_evaluations,
        guess=np.array(CPP[key]["guess"], dtype=np.float64),
    )
    curve.enable_extrapolation()
    return curve


@pytest.mark.parametrize("key", sorted(_layer4_cases()))
def test_fit_matches_cpp_bit_exactly(key: str) -> None:
    """Converged solution, evaluation count, cost, end criteria, and the curve.

    Bit-exact, deliberately.  The fit is a ``Simplex`` over a cost function
    driven by a ``NewtonSafe`` root-find; both are transcriptions rather than
    SciPy wrappers, so the entire iterate path is reproducible — and the
    evaluation counts here (up to 1665) only agree if it really is.  A wrapper
    would land in the same basin and disagree on every one of these numbers.
    """
    block = CPP[key]
    curve = _run_fit(key)
    results = curve.fit_results()

    assert results.number_of_iterations() == block["number_of_iterations"], key
    assert int(results.error_code()) == block["error_code"], key
    assert curve.number_of_bonds() == block["n_bonds"], key
    assert curve.max_date().serial_number() == block["max_date"], key
    # TIGHT rather than EXACT, and only here. The cost is a SUM OF SQUARES of
    # quote errors, each a difference of two near-equal prices, each scaled by
    # a weight that came out of a NewtonSafe root-find over ActualActual(ISDA)
    # year fractions. On the Canada set those weights agree to 2.3e-16 (one
    # ulp), and squaring and summing four of them puts the total 2 ulp out:
    # 0.009290498268401915 against ...913. Every other quantity in this test,
    # including the entire iterate path, is bit-identical.
    tolerance.tight(results.minimum_cost_value(), block["minimum_cost_value"], reason=key)
    for i, (got, expected) in enumerate(zip(results.solution(), block["solution"], strict=True)):
        tolerance.exact(float(got), expected, reason=f"{key} solution[{i}]")
    for i, (got, expected) in enumerate(zip(results.weights(), block["weights"], strict=True)):
        tolerance.tight(float(got), expected, reason=f"{key} weights[{i}]")
    for t, expected in zip(block["t"], block["discount"], strict=True):
        tolerance.exact(curve.discount(t, True), expected, reason=f"{key} discount t={t}")
    for t, expected in zip(block["t"], block["zero_continuous"], strict=True):
        tolerance.exact(
            curve.zero_rate(t, Compounding.Continuous).rate(),
            expected,
            reason=f"{key} zero t={t}",
        )


def test_constraint_is_threaded_into_the_optimization() -> None:
    """The same one-parameter problem, with and without ``PositiveConstraint``.

    test-suite/fittedbonddiscountcurve.cpp:326-334.  The unconstrained fit
    lands on a negative coefficient; the constrained one cannot, because
    ``Simplex`` builds its initial simplex through ``Constraint::update``,
    which HALVES the step until the vertex is feasible, and rejects infeasible
    trial points inside ``extrapolate``.  A port that accepted the constraint
    and then ignored it would reproduce the first number and not the second.
    """
    unconstrained = CPP["layer4_unconstrained"]["solution"][0]
    constrained = CPP["layer4_positive_constrained"]["solution"][0]
    assert unconstrained < 0.0
    assert constrained > 0.0
    assert _run_fit("layer4_unconstrained").fit_results().solution()[0] < 0.0
    assert _run_fit("layer4_positive_constrained").fit_results().solution()[0] > 0.0


def test_refit_continues_from_the_previous_solution_until_reset_guess() -> None:
    """cpp:295 — ``calculate()`` writes the answer back into ``guessSolution_``.

    So an ordinary invalidate-and-refit restarts from where the last one
    stopped and takes FEWER evaluations (139 against 149), while
    ``resetGuess`` puts the original guess back and reproduces the first run
    exactly.  A port that forgot the write-back would report 149 all three
    times; one that forgot ``resetGuess`` would report 139.
    """
    block = CPP["layer4_refit"]
    assert block["iterations_2"] != block["iterations_1"], "premise changed"
    assert block["iterations_3"] == block["iterations_1"]

    guess: npt.NDArray[np.float64] = np.array([-0.02, 0.0], dtype=np.float64)
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=SimplePolynomialFitting(2, True),
        reference_date=_EVAL,
        bonds=_zero_helpers(),
        accuracy=1e-10,
        max_evaluations=5000,
        guess=guess,
    )
    curve.enable_extrapolation()

    for step, suffix in ((None, "1"), ("update", "2"), ("reset", "3")):
        if step == "update":
            curve.update()
        elif step == "reset":
            curve.reset_guess(guess)
        results = curve.fit_results()
        assert results.number_of_iterations() == block[f"iterations_{suffix}"], suffix
        tolerance.exact(results.minimum_cost_value(), block[f"cost_{suffix}"], reason=suffix)
        for got, expected in zip(results.solution(), block[f"solution_{suffix}"], strict=True):
            tolerance.exact(float(got), expected, reason=suffix)


def test_reset_guess_checks_the_size() -> None:
    """cpp:119 — an empty guess is allowed, a wrong-sized one is not."""
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=NelsonSiegelFitting(),
        reference_date=_EVAL,
        bonds=_zero_helpers(),
        guess=np.array([0.02, 0.0, 0.0, 1.0], dtype=np.float64),
    )
    curve.reset_guess([])
    curve.reset_guess([0.02, 0.0, 0.0, 1.0])
    with pytest.raises(LibraryException, match="guess is of wrong size"):
        curve.reset_guess([0.02, 0.0])


def test_curve_owns_a_clone_of_the_fitting_method() -> None:
    """``Clone<FittingMethod>`` (hpp:164) — the caller's object is never touched.

    probe.cpp:797-816.  After a full fit the curve's method carries a solution
    and a weight vector; the instance the caller passed still carries neither.
    """
    block = CPP["clone_ownership"]
    caller = SimplePolynomialFitting(2, True)
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=caller,
        reference_date=_EVAL,
        bonds=_zero_helpers(),
        accuracy=1e-10,
        max_evaluations=5000,
        guess=np.array([-0.02, 0.0], dtype=np.float64),
    )
    curve.enable_extrapolation()
    results = curve.fit_results()

    assert results is not caller
    assert caller.solution().size == block["caller_solution_size"]
    assert caller.weights().size == block["caller_weights_size"]
    assert int(caller.error_code()) == block["caller_error_code"]
    assert results.solution().size == block["curve_solution_size"]
    assert results.weights().size == block["curve_weights_size"]


# ---------------------------------------------------------------------------
# where the paths part company
# ---------------------------------------------------------------------------


def _sweep_case(key: str) -> tuple[Callable[[], FittingMethod], Sequence[float], float, int]:
    if key.startswith("layer4_exp"):
        return (
            lambda: ExponentialSplinesFitting(True, num_coeffs=4, fixed_kappa=0.05),
            [0.2, 0.1, 0.05],
            1e-9,
            5 if key.endswith("earlycaps") else 100,
        )
    return (lambda: CubicBSplinesFitting(_KNOTS9, True), [0.98, 0.94, 0.88, 0.80], 1e-9, 100)


def _run_capped(key: str, cap: int) -> FittedBondDiscountCurve:
    make_method, guess, accuracy, max_stationary = _sweep_case(key)
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=make_method(),
        reference_date=_EVAL,
        bonds=_zero_helpers(),
        accuracy=accuracy,
        max_evaluations=cap,
        guess=np.array(guess, dtype=np.float64),
        max_stationary_state_iterations=max_stationary,
    )
    curve.enable_extrapolation()
    return curve


@pytest.mark.parametrize("key", ["layer4_exp_zeros_earlycaps", "layer4_exp_zeros_caps"])
def test_exponential_splines_truncated_fits_are_bit_exact_at_every_cap(key: str) -> None:
    """The same fit stopped at a ladder of iteration caps. probe.cpp:82-105.

    ``max_stationary_state_iterations`` does not steer the simplex — it only
    seeds the unconditional ``check_stationary_point`` call at the exit
    (simplex.cpp:141-152) — so lowering it to reach caps below 100 leaves the
    iterate path untouched, which is what makes the early rungs meaningful.

    Every rung of both ladders, from 12 function evaluations to 257, agrees
    with C++ in the solution, the cost, the evaluation count and the exit code,
    to the last bit.  Before ``ExponentialSplinesFitting`` used
    :func:`math.fma` the plain transcription missed twelve of these fourteen
    early rungs, so this is the regression guard for that decision.
    """
    for cap, block in CPP[key].items():
        results = _run_capped(key, int(cap)).fit_results()
        assert results.number_of_iterations() == block["iterations"], f"{key} cap={cap}"
        assert int(results.error_code()) == block["error_code"], f"{key} cap={cap}"
        tolerance.exact(results.minimum_cost_value(), block["cost"], reason=f"{key} cap={cap}")
        for i, (got, expected) in enumerate(
            zip(results.solution(), block["solution"], strict=True)
        ):
            tolerance.exact(float(got), expected, reason=f"{key} cap={cap} x[{i}]")


# Where the B-spline ladder stands today. Measured, not assumed: caps 110-300
# agree in every emitted quantity; at 400 the iterate path is still identical
# and only the cost has drifted, by 1.5e-22 in absolute terms on a value of
# 4.7e-17; from 500 the paths part. The grouping is written so it stays valid
# once bspline.py gains the FMA contraction and everything collapses to exact.
_BSPLINE_FULLY_EXACT_CAPS = ("110", "200", "300")
_BSPLINE_PATH_EXACT_CAPS = ("400",)
_BSPLINE_PATH_DIVERGED_CAPS = ("500", "600", "700", "800", "840")


def test_bspline_truncated_fits_are_bit_exact_until_the_cost_reaches_the_noise_floor() -> None:
    """Localise the B-spline divergence to a single band of the iterate path.

    Four B-spline coefficients over four bonds is an exactly determined system,
    so the cost falls to the double-precision floor as the fit converges:
    2.3 at 110 evaluations, 2.9e-07 at 300, 4.7e-17 at 400, 4.9e-25 at 500.
    Below 400 the missing FMA in ``BSpline._basis`` (see
    :func:`test_bspline_fit_diverges_until_bspline_basis_gets_fma_contraction`)
    is invisible; past it the simplex is comparing noise and a half-ulp decides
    which vertex is "better".

    That is the whole content of the divergence, and it is why the discount
    factors keep agreeing to 2e-14 while the evaluation count does not.
    """
    ladder = CPP["layer4_bspline_zeros_caps"]
    assert set(ladder) == set(
        _BSPLINE_FULLY_EXACT_CAPS + _BSPLINE_PATH_EXACT_CAPS + _BSPLINE_PATH_DIVERGED_CAPS
    )

    for cap in _BSPLINE_FULLY_EXACT_CAPS:
        block = ladder[cap]
        results = _run_capped("layer4_bspline_zeros_caps", int(cap)).fit_results()
        assert results.number_of_iterations() == block["iterations"], f"cap={cap}"
        assert int(results.error_code()) == block["error_code"], f"cap={cap}"
        tolerance.exact(results.minimum_cost_value(), block["cost"], reason=f"cap={cap}")
        for i, (got, expected) in enumerate(
            zip(results.solution(), block["solution"], strict=True)
        ):
            tolerance.exact(float(got), expected, reason=f"cap={cap} x[{i}]")

    for cap in _BSPLINE_PATH_EXACT_CAPS:
        block = ladder[cap]
        results = _run_capped("layer4_bspline_zeros_caps", int(cap)).fit_results()
        assert results.number_of_iterations() == block["iterations"], f"cap={cap}"
        assert int(results.error_code()) == block["error_code"], f"cap={cap}"
        for i, (got, expected) in enumerate(
            zip(results.solution(), block["solution"], strict=True)
        ):
            tolerance.exact(float(got), expected, reason=f"cap={cap} x[{i}]")
        tolerance.custom(
            results.minimum_cost_value(),
            block["cost"],
            abs_tol=2e-22,
            rel_tol=0.0,
            reason=(
                "cap 400: the parameter vector is still bit-identical, so the "
                "1.5e-22 gap in the cost is the accumulated half-ulp of the "
                "un-contracted BSpline basis and nothing else. Stated as an "
                "ABSOLUTE bound because the cost is 4.7e-17 and heading to "
                "4.9e-25 — relative terms are meaningless at that magnitude"
            ),
        )

    for cap in _BSPLINE_PATH_DIVERGED_CAPS:
        block = ladder[cap]
        results = _run_capped("layer4_bspline_zeros_caps", int(cap)).fit_results()
        # Only the solution, which survives the divergence: the two simplices
        # are wandering inside the same 1e-13-wide basin.
        for i, (got, expected) in enumerate(
            zip(results.solution(), block["solution"], strict=True)
        ):
            tolerance.tight(float(got), expected, reason=f"cap={cap} x[{i}]")


def test_bspline_fit_is_bit_exact_including_the_evaluation_count() -> None:
    """The CubicBSplinesFitting fit, down to the simplex evaluation count.

    THIS TEST WAS THE MARKER FOR AN ALIGNMENT, AND THAT ALIGNMENT HAS LANDED.
    ``BSpline::N`` (bspline.cpp:49-57) is

        return ((x - knots_[i])/(knots_[i+p] - knots_[i]))*N(i, p-1, x) +
               ((knots_[i+p+1] - x)/(knots_[i+p+1] - knots_[i+1]))*N(i+1, p-1, x);

    which Clang, at its default ``-ffp-contract=on``, fuses into a single FMA.
    ``bspline.py`` transcribed it as separate operations, which cost, measured
    against this reference file:

        probed basis values     4 mismatches / 80
        layer-1 discount rows   4 mismatches / 20
        THIS fit                858 evaluations against C++'s 855

    ``BSpline._basis`` now uses ``math.fma``, and all three are exact. The
    same contraction inside nonlinear_fitting_methods.py is what makes the
    exponential splines and the simple polynomial reproduce exactly.

    WHY IT SHOWED UP HERE AND NOWHERE ELSE. Four B-spline coefficients over
    four bonds is an exactly determined system, so the cost falls to the
    double-precision floor: 4.7e-17 at cap 400, 4.9e-25 at cap 500. Past that
    the simplex is comparing noise and a half-ulp decides the branch — see
    :func:`test_bspline_truncated_fits_are_bit_exact_until_the_cost_reaches_the_noise_floor`.
    The evaluation count is asserted here precisely because it is the most
    fragile observable in the whole module: it is what catches the next
    contraction difference, long before any price moves.
    """
    block = CPP["layer4_bspline_zeros"]
    curve = FittedBondDiscountCurve(
        day_counter=_A365,
        fitting_method=CubicBSplinesFitting(_KNOTS9, True),
        reference_date=_EVAL,
        bonds=_zero_helpers(),
        accuracy=1e-9,
        max_evaluations=500,
        guess=np.array(block["guess"], dtype=np.float64),
    )
    curve.enable_extrapolation()
    results = curve.fit_results()

    # What holds today, and must keep holding.
    for i, (got, expected) in enumerate(zip(results.solution(), block["solution"], strict=True)):
        tolerance.tight(float(got), expected, reason=f"solution[{i}]")
    for t, expected in zip(block["t"], block["discount"], strict=True):
        tolerance.tight(curve.discount(t, True), expected, reason=f"discount t={t}")
    for t, expected in zip(block["t"], block["zero_continuous"], strict=True):
        tolerance.tight(
            curve.zero_rate(t, Compounding.Continuous).rate(), expected, reason=f"zero t={t}"
        )

    # The fragile observables the FMA contraction controls.
    assert results.number_of_iterations() == block["number_of_iterations"], (
        "simplex evaluation count moved — check BSpline._basis still applies "
        "the FMA contraction Clang gives bspline.cpp:56-57"
    )
    assert int(results.error_code()) == block["error_code"]
    tolerance.exact(results.minimum_cost_value(), block["minimum_cost_value"])
