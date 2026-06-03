"""FRA sample — set up a term structure and price forward-rate agreements.

Port of QuantLib's ``Examples/FRA/FRA.cpp`` (the Java ``FRA.java`` was an
incomplete stub that never built the curve; this follows the complete C++
original). Bootstraps a Euribor-3M discount curve from five FRA quotes
(1x4 ... 9x12) via :class:`FraRateHelper` + :class:`PiecewiseYieldCurve`, then
constructs the corresponding :class:`ForwardRateAgreement` instruments and
prints their forward rate, amount and NPV (all ~0 at par). It then shifts the
quotes up 100 bp and re-prices the same strikes, showing positive NPVs.

PQuantLib has no ``RelinkableHandle`` and its :class:`PiecewiseYieldCurve`
bootstraps once and caches, so the 100 bp shift is modelled by building a second
fresh curve off the bumped quotes (rather than mutating the live
:class:`SimpleQuote` objects in place as the C++ original did) — same result,
two curves instead of one relinked handle.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.forward_rate_agreement import ForwardRateAgreement
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.position import PositionType
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.fra_rate_helper import FraRateHelper
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_unit import TimeUnit
from pquantlib_samples.util.stop_clock import StopClock


@dataclass(frozen=True, slots=True)
class FraQuote:
    """One row of FRA output for a given monthsToStart."""

    months_to_start: int
    strike: float
    forward_rate: float
    amount: float
    npv: float


@dataclass(frozen=True, slots=True)
class FraResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    todays_date: Date
    settlement_date: Date
    par: tuple[FraQuote, ...]
    shifted: tuple[FraQuote, ...]


_MONTHS_TO_START = (1, 2, 3, 6, 9)
_BASE_QUOTES = {1: 0.030, 2: 0.031, 3: 0.032, 6: 0.033, 9: 0.034}


def _build_quote(
    index: Euribor,
    curve: object,
    settlement_date: Date,
    months: int,
    strike: float,
    notional: float,
) -> FraQuote:
    cal = index.fixing_calendar()
    bdc = index.business_day_convention()
    value_date = cal.advance(settlement_date, months, TimeUnit.Months, bdc)
    fra = ForwardRateAgreement(
        index,
        value_date,
        PositionType.Long,
        strike,
        notional,
        discount_curve=curve,  # type: ignore[arg-type]
    )
    return FraQuote(
        months_to_start=months,
        strike=strike,
        forward_rate=fra.forward_rate().rate(),
        amount=fra.amount(),
        npv=fra.npv(),
    )


def compute() -> FraResult:
    todays_date = Date.from_ymd(23, Month.May, 2006)
    ObservableSettings().evaluation_date = todays_date

    euribor3m = Euribor.three_months()
    calendar = euribor3m.fixing_calendar()
    fixing_days = euribor3m.fixing_days()
    settlement_date = calendar.advance(todays_date, fixing_days, TimeUnit.Days)

    fra_day_counter = euribor3m.day_counter()
    convention = euribor3m.business_day_convention()
    end_of_month = euribor3m.end_of_month()

    def build_curve(quote_levels: dict[int, float]) -> PiecewiseYieldCurve:
        helpers = [
            FraRateHelper(
                SimpleQuote(quote_levels[m]),
                months_to_start=m,
                months_to_end=m + 3,
                fixing_days=fixing_days,
                calendar=calendar,
                convention=convention,
                end_of_month=end_of_month,
                day_counter=fra_day_counter,
                evaluation_date=todays_date,
            )
            for m in _MONTHS_TO_START
        ]
        c = PiecewiseYieldCurve(
            traits=Discount,
            reference_date=settlement_date,
            instruments=helpers,
            day_counter=ActualActual(Convention.ISDA),
        )
        # The last 9x12 FRA's forecast horizon nudges a hair past the final
        # bootstrap pillar; allow flat extrapolation so the forecast resolves
        # (the C++ piecewise curve is likewise queried at/after its last node).
        c.enable_extrapolation()
        return c

    par_curve = build_curve(_BASE_QUOTES)
    par_index = Euribor.three_months(par_curve)
    par = tuple(
        _build_quote(par_index, par_curve, settlement_date, m, _BASE_QUOTES[m], 100.0)
        for m in _MONTHS_TO_START
    )

    # 100 bp upward shift in all FRA quotes; strikes stay at the original level
    # so the repriced FRAs now show positive value. (pquantlib's
    # PiecewiseYieldCurve bootstraps once and caches, so we rebuild a fresh
    # curve off the shifted quotes rather than relinking in place — the C++
    # original mutated the SimpleQuotes behind a RelinkableHandle.)
    bps_shift = 0.01
    shifted_quotes = {m: _BASE_QUOTES[m] + bps_shift for m in _MONTHS_TO_START}
    shifted_curve = build_curve(shifted_quotes)
    shifted_index = Euribor.three_months(shifted_curve)
    shifted = tuple(
        _build_quote(shifted_index, shifted_curve, settlement_date, m, _BASE_QUOTES[m], 100.0)
        for m in _MONTHS_TO_START
    )

    return FraResult(
        todays_date=todays_date,
        settlement_date=settlement_date,
        par=par,
        shifted=shifted,
    )


def run() -> None:
    print("::::: FRA :::::")

    clock = StopClock()
    clock.start_clock()

    r = compute()
    print(f"Today: {r.todays_date}")
    print(f"Settlement date: {r.settlement_date}")
    print()
    print("Test FRA construction, NPV calculation, and FRA purchase")
    print()
    for q in r.par:
        print(f"3m Term FRA, Months to Start: {q.months_to_start}")
        print(f"strike FRA rate:     {q.strike}")
        print(f"FRA 3m forward rate: {q.forward_rate}")
        print(f"FRA amount [~zero]:  {q.amount}")
        print(f"FRA NPV [~zero]:     {q.npv}")
        print()

    print()
    print("Now take a 100 basis-point upward shift in FRA quotes and examine NPV")
    print()
    for q in r.shifted:
        print(f"3m Term FRA, 100 notional, Months to Start = {q.months_to_start}")
        print(f"strike FRA rate:     {q.strike}")
        print(f"FRA 3m forward rate: {q.forward_rate}")
        print(f"FRA amount [positive]: {q.amount}")
        print(f"FRA NPV [positive]:    {q.npv}")
        print()

    clock.stop_clock()
    clock.log()


if __name__ == "__main__":
    run()
