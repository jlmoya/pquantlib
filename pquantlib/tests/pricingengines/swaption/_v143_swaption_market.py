"""Shared market builder for the v1.43 ``pe/swaption`` cluster tests.

Reconstructs the market described by
``migration-harness/cpp/probes/v143_pe_swaption/probe.cpp`` so the test
files can restate the *case*, not the constants. Leading underscore keeps
pytest from collecting this module.

Every constant here is mirrored in the probe's ``inputs`` blocks and is
asserted against them by ``test_black_swaption_engine_v143.py::
test_market_setup_matches_probe``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.models.shortrate.onefactor.gsr import Gsr
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.time.calendar import Calendar

# probe.cpp:293 — Settings::instance().evaluationDate() = Date(3, March, 2025).
TODAY: Date = Date.from_ymd(3, Month.March, 2025)
# probe.cpp:294-295 — the forwarding curve is 3%, the discount curve 2.5%.
FORWARDING_RATE: float = 0.03
DISCOUNT_RATE: float = 0.025
# probe.cpp:363-366 — Gsr(fwdCurve, [], [0.01], [0.01], 60.0).
GSR_SIGMA: float = 0.01
GSR_REVERSION: float = 0.01
GSR_T: float = 60.0


def calendar() -> Calendar:
    return TARGET()


def flat_curve(rate: float) -> FlatForward:
    # probe.cpp:305-308 — FlatForward(kToday, r, Actual365Fixed(), Continuous, Annual).
    return FlatForward.from_rate(
        TODAY, rate, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )


def forwarding_curve() -> FlatForward:
    return flat_curve(FORWARDING_RATE)


def discount_curve() -> FlatForward:
    return flat_curve(DISCOUNT_RATE)


def euribor6m(curve: FlatForward) -> Euribor:
    # probe.cpp:320-323.
    return Euribor(Period(6, TimeUnit.Months), curve)


def exercise_date() -> Date:
    # probe.cpp:330 — TARGET().advance(kToday, 5, Years).
    return calendar().advance(TODAY, 5, TimeUnit.Years)


def swap_start() -> Date:
    # probe.cpp:331 — TARGET().advance(swaptionExpiry(), 2, Days).
    return calendar().advance(exercise_date(), 2, TimeUnit.Days)


def swap_end() -> Date:
    # probe.cpp:332 — TARGET().advance(swapStart(), 5, Years).
    return calendar().advance(swap_start(), 5, TimeUnit.Years)


def bermudan_dates() -> list[Date]:
    # probe.cpp:520-522 — 5y, 6y, 7y out of today.
    cal = calendar()
    return [
        cal.advance(TODAY, 5, TimeUnit.Years),
        cal.advance(TODAY, 6, TimeUnit.Years),
        cal.advance(TODAY, 7, TimeUnit.Years),
    ]


def schedule(tenor: Period) -> Schedule:
    # probe.cpp:334-342 — TARGET, ModifiedFollowing both ends, Backward, not EOM.
    return Schedule.from_rule(
        swap_start(),
        swap_end(),
        tenor,
        calendar(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )


def fixed_schedule() -> Schedule:
    return schedule(Period(1, TimeUnit.Years))


def float_schedule() -> Schedule:
    return schedule(Period(6, TimeUnit.Months))


def make_swap(
    curve: FlatForward,
    swap_type: SwapType,
    fixed_rate: float,
    spread: float = 0.0,
) -> VanillaSwap:
    # probe.cpp:344-351 — notional 1.0, annual Thirty360(BondBasis) fixed leg,
    # semi-annual Actual360 Euribor6M float leg.
    return VanillaSwap(
        swap_type,
        1.0,
        fixed_schedule(),
        fixed_rate,
        Thirty360(Convention.BondBasis),
        float_schedule(),
        euribor6m(curve),
        spread,
        Actual360(),
    )


def make_gsr(curve: FlatForward) -> Gsr:
    # probe.cpp:363-366.
    return Gsr(
        curve,
        volstepdates=[],
        volatilities=[GSR_SIGMA],
        reversion=GSR_REVERSION,
        T=GSR_T,
    )
