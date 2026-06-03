"""YieldCurveTermStructures sample — explores yield term-structure functionality.

Port of ``org.jquantlib.samples.YieldCurveTermStructures``. Exercises three
concrete yield term structures that pquantlib ships:

* :class:`FlatForward`              — flat continuously-compounded curve;
* :class:`ForwardSpreadedTermStructure` — adds a flat spread to the forwards;
* :class:`ImpliedTermStructure`     — re-bases an existing curve to a future date.

For each it prints the 30-day discount factor, the 30d->50d forward rate, and
the 10-day zero rate.

Divergences from the Java original (documented inline):

* The Java sample called ``curve.parRate(dates, freq, isBond)``. In C++/v1.42.1
  ``parRate`` is a free function on ``CashFlows`` (it requires a coupon leg), not
  a method on ``YieldTermStructure``; pquantlib follows the C++ layout, so the
  par-rate print line is omitted.
* The remaining Java sections (InterpolatedZero/Forward/Discount curves,
  PiecewiseYieldCurve, ZeroSpreadedTermStructure) were ``//TODO`` stubs in the
  original and produced no output; they are not reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.forward_spreaded_term_structure import (
    ForwardSpreadedTermStructure,
)
from pquantlib.termstructures.yield_.implied_term_structure import ImpliedTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib_samples.util.stop_clock import StopClock


@dataclass(frozen=True, slots=True)
class CurveProbe:
    """The three headline numbers each curve in the sample prints."""

    discount_30d: float
    forward_30d_50d: float
    zero_10d: float


@dataclass(frozen=True, slots=True)
class YieldCurveResult:
    """Computed quantities a :func:`run` would print — for cross-checking."""

    flat_forward: CurveProbe
    forward_spreaded: CurveProbe
    implied: CurveProbe


def _probe(curve: object, d10: Date, d30: Date, d50: Date) -> CurveProbe:
    dc = Actual365Fixed()
    return CurveProbe(
        discount_30d=curve.discount(d30),  # type: ignore[attr-defined]
        forward_30d_50d=curve.forward_rate(  # type: ignore[attr-defined]
            d30, d50, Compounding.Continuous, result_day_counter=dc
        ).rate(),
        zero_10d=curve.zero_rate(  # type: ignore[attr-defined]
            d10, Compounding.Continuous, result_day_counter=dc
        ).rate(),
    )


def compute() -> YieldCurveResult:
    dc = Actual365Fixed()

    # C++ parity: the Java sample used the settlement-days FlatForward ctor
    # (FlatForward(2, NYSE, handle, ...)); pquantlib's FlatForward is
    # date-anchored, so we reference it directly at today (a 0-settlement
    # equivalent), which is fine for the discount/forward/zero probes below.
    today = Date.todays_date()
    flat_forward = FlatForward(
        today,
        SimpleQuote(0.3),
        dc,
        Compounding.Continuous,
        Frequency.Daily,
    )

    d10 = today + 10
    d30 = today + 30
    d50 = today + 50

    forward_spreaded = ForwardSpreadedTermStructure(flat_forward, SimpleQuote(0.2))
    implied = ImpliedTermStructure(flat_forward, today)

    return YieldCurveResult(
        flat_forward=_probe(flat_forward, d10, d30, d50),
        forward_spreaded=_probe(forward_spreaded, d10, d30, d50),
        implied=_probe(implied, d10, d30, d50),
    )


def run() -> None:
    print("::::: YieldCurveTermStructures :::::")

    clock = StopClock()
    clock.start_clock()

    r = compute()

    print("//==========================================FlatForward termstructure===================")
    _print_probe(r.flat_forward)

    print("//==========================================ForwardSpreadedTermStructure==================")
    _print_probe(r.forward_spreaded)

    print("//==========================================ImpliedTermStructure==========================")
    _print_probe(r.implied)

    clock.stop_clock()
    clock.log()


def _print_probe(p: CurveProbe) -> None:
    print(f"The discount factor for the date 30 days from today is = {p.discount_30d}")
    print(
        f"The forward rate between the date 30 days from today to 50 days from today is = {p.forward_30d_50d}"
    )
    print(f"The zero rate for the date 10 days from today = {p.zero_10d}")


if __name__ == "__main__":
    run()
