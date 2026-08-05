"""Cross-validate CPIBond against the ``v143/inst/bondsamort`` probe.

The CPI block of the probe runs on its own evaluation date (25-Nov-2009) with
the canonical UKRPI fixing table from ``test-suite/inflationcpibond.cpp`` and a
``ZeroInflationCurve`` built from literal (date, zero-rate) nodes — no
bootstrap, so the curve is reproducible node-for-node on both sides.

Every optional constructor argument gets a NON-default test asserting the whole
bond, including the full cashflow listing, against C++.

Tolerance: TIGHT. A CPI coupon's rate is ``fixed_rate * I(t-lag) / baseCPI``:
one linear interpolation on the zero-inflation curve plus a division, with no
solver in the path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as AAConvention
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.uk_rpi import UKRPI
from pquantlib.instruments.bonds.cpi_bond import CPIBond
from pquantlib.pricingengines.bond.discounting_bond_engine import DiscountingBondEngine
from pquantlib.termstructures.inflation.interpolated_zero_inflation_curve import (
    InterpolatedZeroInflationCurve,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule, Schedule
from pquantlib.time.time_unit import TimeUnit

from ._bondsamort import compare_bond, load_reference, pinned_evaluation_date

CPI_TODAY: Date = Date.from_ymd(25, Month.November, 2009)
ISSUE: Date = Date.from_ymd(2, Month.October, 2007)

#: Canonical UKRPI monthly fixings Jul-2007 .. Sep-2009
#: (test-suite/inflationcpibond.cpp CommonVars).
RPI_FIXINGS: list[float] = [
    206.1, 207.3, 208.0, 208.9, 209.7, 210.9, 209.8, 211.4, 212.1,
    214.0, 215.1, 216.8, 216.5, 217.2, 218.4, 217.7, 216.0, 212.9,
    210.1, 211.4, 211.3, 211.5, 212.8, 213.4, 213.4, 213.4, 214.4,
]

#: Probe ``zDates`` / ``zRates`` — literal zero-inflation nodes; ``dates[0]``
#: is the curve's base date.
ZERO_NODES: list[tuple[Date, float]] = [
    (Date.from_ymd(1, Month.September, 2009), 0.0305),
    (Date.from_ymd(1, Month.September, 2012), 0.0293),
    (Date.from_ymd(1, Month.September, 2016), 0.0315),
    (Date.from_ymd(1, Month.September, 2021), 0.0348),
    (Date.from_ymd(1, Month.September, 2035), 0.0377),
    (Date.from_ymd(1, Month.September, 2060), 0.0371),
]

#: Probe ``gNotionalSamples`` for the CPI block.
CPI_NOTIONAL_SAMPLES: list[Date] = [
    Date.from_ymd(1, Month.January, 2008),
    Date.from_ymd(2, Month.April, 2010),
    Date.from_ymd(2, Month.October, 2013),
    Date.from_ymd(2, Month.October, 2017),
    Date.from_ymd(1, Month.January, 2020),
]


class CpiMarket:
    """Probe ``makeCpiMarket()`` — UKRPI history + literal zero-inflation curve."""

    def __init__(self) -> None:
        day_counter = ActualActual(AAConvention.ISDA)
        self.index: UKRPI = UKRPI()
        self.index.clear_fixings()
        rpi_schedule = (
            MakeSchedule()
            .from_date(Date.from_ymd(1, Month.July, 2007))
            .to(Date.from_ymd(1, Month.September, 2009))
            .with_frequency(Frequency.Monthly)
            .build()
        )
        for i, fixing in enumerate(RPI_FIXINGS):
            self.index.add_fixing(rpi_schedule.date(i), fixing, True)

        self.nominal: FlatForward = FlatForward.from_rate(CPI_TODAY, 0.05, day_counter)
        curve = InterpolatedZeroInflationCurve(
            reference_date=CPI_TODAY,
            dates=[d for d, _ in ZERO_NODES],
            rates=[r for _, r in ZERO_NODES],
            frequency=Frequency.Monthly,
            day_counter=day_counter,
        )
        self.index.set_zero_inflation_term_structure(curve)
        self.engine: DiscountingBondEngine = DiscountingBondEngine(self.nominal)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return load_reference()


@pytest.fixture
def pinned_cpi_today() -> Iterator[None]:
    with pinned_evaluation_date(CPI_TODAY):
        yield


@pytest.fixture(scope="module")
def cpi_market() -> CpiMarket:
    with pinned_evaluation_date(CPI_TODAY):
        return CpiMarket()


def _schedule() -> Schedule:
    """Probe ``cpiSchedule()`` — 20 semiannual periods, 2-Oct-2007 → 2-Oct-2017."""
    return (
        MakeSchedule()
        .from_date(ISSUE)
        .to(Date.from_ymd(2, Month.October, 2017))
        .with_tenor(Period(6, TimeUnit.Months))
        .with_calendar(UnitedKingdom())
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )


def _bond(market: CpiMarket, **overrides: Any) -> CPIBond:
    """Probe ``emitCpiBond`` — defaults, with one argument overridden."""
    args: dict[str, Any] = {
        "settlement_days": 3,
        "face_amount": 1000000.0,
        "base_cpi": 206.1,
        "observation_lag": Period(3, TimeUnit.Months),
        "cpi_index": market.index,
        "observation_interpolation": InterpolationType.Flat,
        "schedule": _schedule(),
        "coupons": [0.1],
        "accrual_day_counter": Actual365Fixed(),
    }
    args.update(overrides)
    bond = CPIBond(**args)
    bond.set_pricing_engine(market.engine)
    return bond


def _compare(bond: CPIBond, ref: dict[str, Any]) -> None:
    compare_bond(bond, ref, tight, CPI_NOTIONAL_SAMPLES)
    assert int(bond.frequency()) == ref["frequency"]
    assert bond.growth_only() is ref["growth_only"]
    tight(bond.base_cpi(), ref["base_cpi"])
    assert bond.observation_lag().length == ref["observation_lag_length"]
    assert int(bond.observation_lag().units) == ref["observation_lag_units"]
    assert int(bond.observation_interpolation()) == ref["observation_interpolation"]
    assert (bond.calendar() == UnitedKingdom()) is ref["calendar_is_uk"]
    assert (bond.calendar() == NullCalendar()) is ref["calendar_is_null"]


def test_base(pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]) -> None:
    """All-defaults bond: 20 CPI coupons plus the inflated notional flow.

    The redemption is the leg's own final ``CPICashFlow`` — C++ does not call
    ``addRedemptionsToCashflows`` here (cpibond.cpp:97-101).
    """
    bond = _bond(cpi_market)
    _compare(bond, cpp["cpi_base"])
    assert len(bond.redemptions()) == 1
    assert bond.redemptions()[0] is bond.cashflows()[-1]
    assert bond.cpi_index() is cpi_market.index


def test_observation_interpolation(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``observation_interpolation=Linear`` (default Flat here) changes every rate."""
    ref = cpp["cpi_interp_linear"]
    _compare(
        _bond(cpi_market, observation_interpolation=InterpolationType.Linear), ref
    )
    base_rates = [c.get("rate") for c in cpp["cpi_base"]["cashflows"]]
    assert [c.get("rate") for c in ref["cashflows"]] != base_rates


def test_payment_convention(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``payment_convention=Preceding`` (default ModifiedFollowing)."""
    ref = cpp["cpi_payment_convention"]
    _compare(_bond(cpi_market, payment_convention=BusinessDayConvention.Preceding), ref)
    base_dates = [c["date_serial"] for c in cpp["cpi_base"]["cashflows"]]
    assert [c["date_serial"] for c in ref["cashflows"]] != base_dates


def test_payment_calendar(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``payment_calendar=NullCalendar`` (default: the schedule's UK calendar).

    C++ passes the *bond's* calendar to the CPI leg, so the payment calendar
    also becomes the settlement calendar (cpibond.cpp:78-80, 93).
    """
    ref = cpp["cpi_payment_calendar"]
    bond = _bond(cpi_market, payment_calendar=NullCalendar())
    _compare(bond, ref)
    assert bond.calendar() == NullCalendar()
    assert ref["settlement_serial"] != cpp["cpi_base"]["settlement_serial"]


def test_issue_date(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``issue_date`` (default: null) clamps settlement dates before issue."""
    ref = cpp["cpi_issue_date"]
    bond = _bond(cpi_market, issue_date=ISSUE)
    _compare(bond, ref)
    assert bond.issue_date() == ISSUE
    assert (
        ref["settlement_before_issue_serial"]
        != cpp["cpi_base"]["settlement_before_issue_serial"]
    )


def test_growth_only(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``growth_only=True`` (default False) — the final flow pays growth only.

    That is the ``subtractInflationNominal`` payoff of the deprecated C++
    overload; the coupons are untouched, only the notional exchange shrinks.
    """
    ref = cpp["cpi_growth_only"]
    bond = _bond(cpi_market, growth_only=True)
    _compare(bond, ref)
    assert bond.growth_only() is True
    base = cpp["cpi_base"]
    assert ref["cashflows"][-1]["amount"] < base["cashflows"][-1]["amount"]
    assert [c.get("rate") for c in ref["cashflows"]] == [
        c.get("rate") for c in base["cashflows"]
    ]


def test_base_cpi(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``base_cpi=210.0`` (probe default 206.1) rescales every index ratio."""
    _compare(_bond(cpi_market, base_cpi=210.0), cpp["cpi_base_cpi"])


def test_observation_lag(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``observation_lag=8M`` (probe default 3M) moves every coupon fixing date."""
    ref = cpp["cpi_observation_lag"]
    _compare(_bond(cpi_market, observation_lag=Period(8, TimeUnit.Months)), ref)
    base_fixings = [c.get("fixing_serial") for c in cpp["cpi_base"]["cashflows"]]
    assert [c.get("fixing_serial") for c in ref["cashflows"]] != base_fixings


def test_coupon_vector(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """A per-period fixed-rate vector shorter than the leg: the last repeats."""
    _compare(_bond(cpi_market, coupons=[0.10, 0.08, 0.06]), cpp["cpi_coupon_vector"])


def test_settlement_days(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``settlement_days=7`` (probe default 3)."""
    ref = cpp["cpi_settlement_days"]
    _compare(_bond(cpi_market, settlement_days=7), ref)
    assert ref["settlement_serial"] != cpp["cpi_base"]["settlement_serial"]


def test_face_amount(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """``face_amount=2_500_000`` scales the notional schedule and every flow."""
    _compare(_bond(cpi_market, face_amount=2500000.0), cpp["cpi_face_amount"])


def test_ex_coupon(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """A 10-day ex-coupon period on UnitedKingdom, Preceding (default: none)."""
    ref = cpp["cpi_ex_coupon"]
    _compare(
        _bond(
            cpi_market,
            ex_coupon_period=Period(10, TimeUnit.Days),
            ex_coupon_calendar=UnitedKingdom(),
            ex_coupon_convention=BusinessDayConvention.Preceding,
        ),
        ref,
    )
    assert all(c["ex_coupon_serial"] == 0 for c in cpp["cpi_base"]["cashflows"])
    assert any(c["ex_coupon_serial"] != 0 for c in ref["cashflows"])


def test_ex_coupon_end_of_month(
    pinned_cpi_today: None, cpi_market: CpiMarket, cpp: dict[str, Any]
) -> None:
    """A 1-month ex-coupon period on NullCalendar, Following, end_of_month=True.

    All four ex-coupon arguments differ from :func:`test_ex_coupon`.
    """
    ref = cpp["cpi_ex_coupon_eom"]
    _compare(
        _bond(
            cpi_market,
            ex_coupon_period=Period(1, TimeUnit.Months),
            ex_coupon_calendar=NullCalendar(),
            ex_coupon_convention=BusinessDayConvention.Following,
            ex_coupon_end_of_month=True,
        ),
        ref,
    )
    ten_day = [c["ex_coupon_serial"] for c in cpp["cpi_ex_coupon"]["cashflows"]]
    assert [c["ex_coupon_serial"] for c in ref["cashflows"]] != ten_day
