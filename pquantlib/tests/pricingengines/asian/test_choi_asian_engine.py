"""Cross-validate :class:`ChoiAsianEngine` against the C++ probe.

Reference: ``migration-harness/references/v143/pe/basket`` (the ``ca_*`` cases),
produced by ``migration-harness/cpp/probes/v143_pe_basket/probe.cpp``. The
engine lives in ``ql/pricingengines/asian/`` but is a client of
``ChoiBasketEngine``, so it is pinned in the same reference as the basket
cluster.

Each case carries its whole market — evaluation date, spot, curves, vol,
fixing dates, running accumulator, past fixings, and both engine knobs — so the
sweep reconstructs the option rather than restating constants.

All three branches of ``calculate()`` are covered:

* ``future_fixings > 1`` — the Choi basket;
* ``future_fixings == 1`` — a plain :func:`black_formula`;
* ``future_fixings == 0`` — the discounted intrinsic on the running average,
  reachable only by having today's fixing folded into the past.

Tolerances
----------
TIGHT (``1e-14`` abs / ``1e-12`` rel). The largest deviation across the
reference is 5.0e-15 scaled; the Gauss-Hermite quadrature underneath is
deterministic, so nothing here needs a looser tier.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise, Exercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOption
from pquantlib.instruments.average_type import AverageType
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import (
    CashOrNothingPayoff,
    OptionType,
    PlainVanillaPayoff,
    StrikedTypePayoff,
)
from pquantlib.pricingengines.asian.choi_asian_engine import ChoiAsianEngine
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_DC = Actual365Fixed()
_DEFAULT_MAX_STEPS = 2 << 21


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Save/restore the evaluation date.

    Every case sets its own from the reference's ``today`` field, matching
    ``Settings::instance().evaluationDate() = kCaToday`` in
    ``emitChoiAsianEngine()``. It matters here beyond expiry checks: the engine
    folds a fixing at ``process.time(d) == 0`` into the past fixings, and that
    comparison is against the evaluation date.
    """
    settings = ObservableSettings()
    saved = settings.evaluation_date
    yield
    settings.evaluation_date = saved


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/basket")


def _date(iso: str) -> Date:
    year, month, day = (int(part) for part in iso.split("-"))
    return Date.from_ymd(day, Month(month), year)


def _process(inputs: dict[str, Any], today: Date) -> GeneralizedBlackScholesProcess:
    return BlackScholesMertonProcess(
        x0=SimpleQuote(float(inputs["spot"])),
        dividend_ts=FlatForward.from_rate(
            reference_date=today,
            forward_rate=float(inputs["dividend_yield"]),
            day_counter=_DC,
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=today,
            forward_rate=float(inputs["risk_free_rate"]),
            day_counter=_DC,
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            day_counter=_DC,
            volatility=float(inputs["volatility"]),
        ),
    )


def _set_today(inputs: dict[str, Any]) -> Date:
    today = _date(str(inputs["today"]))
    ObservableSettings().evaluation_date = today
    return today


def _option(
    inputs: dict[str, Any],
    *,
    payoff: StrikedTypePayoff | None = None,
    exercise: Exercise | None = None,
    fixing_dates: list[Date] | None = None,
    running_accumulator: float | None = None,
    past_fixings: int | None = None,
    average_type: AverageType = AverageType.Arithmetic,
) -> DiscreteAveragingAsianOption:
    return DiscreteAveragingAsianOption(
        average_type=average_type,
        running_accumulator=(
            float(inputs["running_accumulator"])
            if running_accumulator is None
            else running_accumulator
        ),
        past_fixings=(
            int(inputs["past_fixings"]) if past_fixings is None else past_fixings
        ),
        fixing_dates=(
            [_date(str(d)) for d in inputs["fixing_dates"]]
            if fixing_dates is None
            else fixing_dates
        ),
        payoff=(
            PlainVanillaPayoff(
                OptionType.Call
                if str(inputs["option_type"]) == "Call"
                else OptionType.Put,
                float(inputs["strike"]),
            )
            if payoff is None
            else payoff
        ),
        exercise=(
            EuropeanExercise(_date(str(inputs["exercise_date"])))
            if exercise is None
            else exercise
        ),
    )


def _engine(
    inputs: dict[str, Any], process: GeneralizedBlackScholesProcess
) -> ChoiAsianEngine:
    steps = int(inputs["max_nr_integration_steps"])
    return ChoiAsianEngine(
        process,
        float(inputs["lambda"]),
        _DEFAULT_MAX_STEPS if steps < 0 else steps,
    )


def test_choi_asian_engine(cpp: dict[str, Any]) -> None:
    """Every priced case, across all three branches.

    Also asserts the results surface: like every engine in this cluster,
    ``calculate()`` writes only the value — no greeks, no additional results.
    """
    checked = 0
    for name, case in cpp.items():
        if not name.startswith("ca_") or "npv" not in case["expected"]:
            continue
        inputs, expected = case["inputs"], case["expected"]
        today = _set_today(inputs)
        option = _option(inputs)
        option.set_pricing_engine(_engine(inputs, _process(inputs, today)))

        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)
        assert len(option.additional_results()) == int(
            expected["additional_results_count"]
        )
        checked += 1
    assert checked >= 20, f"expected the probe to carry ca cases, got {checked}"


def test_todays_fixing_is_folded_into_the_past(cpp: dict[str, Any]) -> None:
    """A fixing at ``t == 0`` moves from future to past, carrying ``x0`` with it.

    ``future_fixings`` drops, ``past_fixings`` rises, and
    ``running_accumulator`` picks up the spot — which then feeds the effective
    strike and the basket weights. The fold is visible in the price: the folded
    5-fixing option is worth strictly less than the unfolded 6-fixing one at
    the same strike, because one of its fixings is now known.
    """
    folded = float(cpp["ca_today_fixing_folded_call"]["expected"]["npv"])
    unfolded = float(cpp["ca_6fix_call_atm"]["expected"]["npv"])
    assert folded < unfolded

    inputs = cpp["ca_today_fixing_folded_call"]["inputs"]
    today = _set_today(inputs)
    assert _date(str(inputs["fixing_dates"][0])) == today
    assert int(inputs["past_fixings"]) == 0

    option = _option(inputs)
    option.set_pricing_engine(_engine(inputs, _process(inputs, today)))
    tolerance.tight(option.npv(), folded, reason="today's fixing folded")


def test_fixing_dates_are_sorted_before_use(cpp: dict[str, Any]) -> None:
    """An out-of-order fixing vector prices identically to the sorted one.

    C++ sorts a *copy*, so the instrument is untouched. A port that indexed the
    caller's order would build the wrong correlation matrix — whose ``i < j``
    structure encodes "the earlier fixing's variance", not "the i-th argument's".
    """
    sorted_case = cpp["ca_today_fixing_folded_call"]
    shuffled_case = cpp["ca_today_fixing_folded_call_unsorted"]
    assert [str(d) for d in shuffled_case["inputs"]["fixing_dates"]] != sorted(
        str(d) for d in shuffled_case["inputs"]["fixing_dates"]
    )
    assert float(sorted_case["expected"]["npv"]) == float(
        shuffled_case["expected"]["npv"]
    )

    inputs = shuffled_case["inputs"]
    today = _set_today(inputs)
    option = _option(inputs)
    option.set_pricing_engine(_engine(inputs, _process(inputs, today)))
    tolerance.tight(
        option.npv(), float(shuffled_case["expected"]["npv"]), reason="unsorted fixings"
    )


def test_single_future_fixing_uses_the_black_formula(cpp: dict[str, Any]) -> None:
    """One unknown fixing: no basket, a plain Black formula.

    The forward is ``x0 / (past + future) * qDF / rDF`` — divided by the
    *total* fixing count — and the discount comes from the exercise date, which
    need not be the fixing date. The two ``ca_1fix_call*`` cases differ only in
    the exercise date, which isolates that.
    """
    same_day = float(cpp["ca_1fix_call"]["expected"]["npv"])
    late = float(cpp["ca_1fix_call_late_exercise"]["expected"]["npv"])
    assert late < same_day  # discounted six months further

    for name in ("ca_1fix_call", "ca_1fix_put", "ca_1fix_call_with_past",
                 "ca_1fix_call_late_exercise"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        assert len(inputs["fixing_dates"]) == 1
        today = _set_today(inputs)
        option = _option(inputs)
        option.set_pricing_engine(_engine(inputs, _process(inputs, today)))
        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)


def test_zero_future_fixings_is_the_discounted_intrinsic(cpp: dict[str, Any]) -> None:
    """All fixings known: ``payoff(accumulator / past) * df(exercise)``.

    Reachable only through the today-fold — the instrument's ``validate()``
    requires at least one fixing date — so the case supplies exactly one fixing,
    dated today. With everything known, the call and the put at strikes 90 and
    110 around a spot of 100 must price identically, which is asserted here as
    an arithmetic fact rather than trusted from the reference.
    """
    assert bool(cpp["ca_zero_fixings_is_unreachable"]["expected"]["reachable_directly"]) is False

    for name in ("ca_only_today_fixing_call", "ca_only_today_fixing_put"):
        inputs, expected = cpp[name]["inputs"], cpp[name]["expected"]
        today = _set_today(inputs)
        assert _date(str(inputs["fixing_dates"][0])) == today

        process = _process(inputs, today)
        option = _option(inputs)
        option.set_pricing_engine(_engine(inputs, process))

        discount = process.risk_free_rate().discount(
            _date(str(inputs["exercise_date"]))
        )
        strike = float(inputs["strike"])
        spot = float(inputs["spot"])
        intrinsic = (
            max(0.0, spot - strike)
            if str(inputs["option_type"]) == "Call"
            else max(0.0, strike - spot)
        ) * discount
        tolerance.tight(option.npv(), intrinsic, reason=f"{name}: intrinsic")
        tolerance.tight(option.npv(), float(expected["npv"]), reason=name)


def test_past_fixings_divide_by_the_total_count(cpp: dict[str, Any]) -> None:
    """The effective strike subtracts ``accumulator / (past + future)``.

    Not ``accumulator / past``. With 6 past fixings averaging 102 and 6 future
    ones, the strike drops by 51, not by 102 — a port using ``past_fixings``
    would produce a *negative* effective strike here and throw. Checked as
    arithmetic, then against the reference.
    """
    inputs, expected = cpp["ca_past6_future6_call"]["inputs"], cpp[
        "ca_past6_future6_call"
    ]["expected"]
    past = int(inputs["past_fixings"])
    future = len(inputs["fixing_dates"])
    accumulator = float(inputs["running_accumulator"])
    assert past == 6
    assert future == 6
    assert accumulator / (past + future) == 51.0
    assert float(inputs["strike"]) - accumulator / past < 0.0

    today = _set_today(inputs)
    option = _option(inputs)
    option.set_pricing_engine(_engine(inputs, _process(inputs, today)))
    tolerance.tight(option.npv(), float(expected["npv"]), reason="past6_future6")


def test_lambda_and_max_steps_are_real_knobs(cpp: dict[str, Any]) -> None:
    """Both are forwarded to the underlying Choi basket and move the answer."""
    base = float(cpp["ca_6fix_call_atm"]["expected"]["npv"])
    lambda5 = float(cpp["ca_6fix_call_lambda5"]["expected"]["npv"])
    lambda20 = float(cpp["ca_6fix_call_lambda20"]["expected"]["npv"])
    assert len({base, lambda5, lambda20}) == 3


@pytest.mark.parametrize(
    "case_name",
    [
        "ca_rejects_geometric_average",
        "ca_rejects_american_exercise",
        "ca_rejects_non_plain_payoff",
        "ca_rejects_negative_effective_strike",
        "ca_rejects_fixing_after_exercise",
        "ca_rejects_duplicate_fixing_dates",
    ],
)
def test_choi_asian_guards(cpp: dict[str, Any], case_name: str) -> None:
    inputs, expected = cpp[case_name]["inputs"], cpp[case_name]["expected"]
    assert bool(expected["throws"])

    base = cpp["ca_6fix_call_atm"]["inputs"]
    today = _set_today(base)
    process = _process(base, today)
    fixings = [
        _date(str(base["fixing_dates"][k])) for k in range(4)
    ]
    call = PlainVanillaPayoff(OptionType.Call, 100.0)

    kwargs: dict[str, Any] = {
        "fixing_dates": fixings,
        "running_accumulator": 0.0,
        "past_fixings": 0,
        "payoff": call,
        "exercise": EuropeanExercise(fixings[-1]),
    }
    match case_name:
        case "ca_rejects_geometric_average":
            kwargs["average_type"] = AverageType.Geometric
            kwargs["running_accumulator"] = 1.0
        case "ca_rejects_american_exercise":
            kwargs["exercise"] = AmericanExercise(today, fixings[-1])
        case "ca_rejects_non_plain_payoff":
            kwargs["payoff"] = CashOrNothingPayoff(OptionType.Call, 100.0, 1.0)
        case "ca_rejects_negative_effective_strike":
            kwargs["running_accumulator"] = 4 * 300.0
            kwargs["past_fixings"] = 4
        case "ca_rejects_fixing_after_exercise":
            kwargs["exercise"] = EuropeanExercise(fixings[0])
        case _:
            kwargs["fixing_dates"] = [fixings[0], fixings[1], fixings[1], fixings[2]]

    option = _option(base, **kwargs)
    option.set_pricing_engine(ChoiAsianEngine(process, 10.0, 2 << 12))
    with pytest.raises(LibraryException, match=str(inputs["why"]).strip()):
        option.npv()
