"""CPIBondHelper dates + implied quote, cross-validated against C++ v1.43.

Reference: ``migration-harness/references/v143/ts/ratehelpers.json``
Probe:     ``migration-harness/cpp/probes/v143_ts_ratehelpers/probe.cpp``

14 configurations pin the helper's dates and its implied quote. The market is
the canonical UKRPI one from ``test-suite/inflationcpibond.cpp`` — literal
monthly fixings plus a zero-inflation curve built from literal (date, rate)
nodes, no bootstrap — extended by twelve months so the two 2010 evaluation
dates still have every fixing they need. **A missing inflation fixing does not
degrade the answer, it raises**, so the table is transcribed exactly and
re-seeded per test, truncated per evaluation date the way
:class:`_CpiMarket` documents.

What these cases are here to hold down:

* the dates are the BOND's, not the schedule's (bondhelpers.cpp:35-38).
  ``latest_date`` is the last CASHFLOW date: the schedule ends on Saturday
  2 October 2027, so the ModifiedFollowing payment convention pushes the final
  coupon and the redemption to Monday 4 October and ``latest_date`` lands two
  days after ``bond.maturity_date()``. ``cpi_payment_preceding`` is the mirror,
  where it lands one day *before*.
* ``earliest_date`` is ``bond.next_cash_flow_date()``, which moves with the
  evaluation date. ``cpi_eval_2010_03_31`` settles exactly ON a payment date
  (Good Friday 2 April 2010 rolls the coupon to Tuesday the 6th, which is also
  where three settlement days from 31 March land), so it pins the
  "has this flow occurred" boundary rather than an interior date.

Tolerance: EXACT for every date and count; TIGHT for the prices. A CPI coupon
is ``fixed_rate * I(t - lag) / base_cpi`` — one linear interpolation on the
zero-inflation curve and a division — and the helper adds a discounted sum on
top. No solver anywhere in the path.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as AAConvention
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.indexes.inflation.uk_rpi import UKRPI
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.inflation.interpolated_zero_inflation_curve import (
    InterpolatedZeroInflationCurve,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.bond_helper import BondPriceType
from pquantlib.termstructures.yield_.cpi_bond_helper import CPIBondHelper
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.united_kingdom import UnitedKingdom
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule, Schedule
from pquantlib.time.time_unit import TimeUnit

_MONTHS = TimeUnit.Months

#: probe.cpp — the CPI block's base evaluation date.
_CPI_TODAY = Date.from_ymd(25, Month.November, 2009)
_ISSUE = Date.from_ymd(2, Month.October, 2007)

#: probe.cpp ``kRpiFixings`` — the 27 canonical values from
#: ``test-suite/inflationcpibond.cpp`` (Jul-2007 .. Sep-2009) continued through
#: Sep-2010 so the 2010 evaluation dates have their history. How much of the
#: table each case actually loads is decided by :class:`_CpiMarket`.
_RPI_FIXINGS: list[float] = [
    206.1, 207.3, 208.0, 208.9, 209.7, 210.9, 209.8, 211.4, 212.1,
    214.0, 215.1, 216.8, 216.5, 217.2, 218.4, 217.7, 216.0, 212.9,
    210.1, 211.4, 211.3, 211.5, 212.8, 213.4, 213.4, 213.4, 214.4,
    # Oct-2009 .. Sep-2010
    215.3, 216.0, 218.0, 217.9, 219.2, 220.7, 222.8, 223.6,
    224.1, 223.6, 224.5, 225.3,
]

#: probe.cpp ``zDates`` / ``zRates`` — literal zero-inflation nodes.
#: ``dates[0]`` is the curve's base date.
_ZERO_NODES: list[tuple[Date, float]] = [
    (Date.from_ymd(1, Month.September, 2009), 0.0305),
    (Date.from_ymd(1, Month.September, 2012), 0.0293),
    (Date.from_ymd(1, Month.September, 2016), 0.0315),
    (Date.from_ymd(1, Month.September, 2021), 0.0348),
    (Date.from_ymd(1, Month.September, 2035), 0.0377),
    (Date.from_ymd(1, Month.September, 2060), 0.0371),
]

_AA_ISDA = ActualActual(AAConvention.ISDA)
_MF = BusinessDayConvention.ModifiedFollowing

#: C++ ``Bond::Price::Type`` is ``enum Type { Dirty, Clean }`` (bond.hpp:70).
#: The probe emits ``int(helper->priceType())`` in that encoding.
_CPP_PRICE_TYPE_NAMES: dict[int, str] = {0: "Dirty", 1: "Clean"}


@dataclass(frozen=True)
class _Case:
    """One ``CpiCase`` from the probe (probe.cpp ``struct CpiCase``)."""

    eval_date: Date = _CPI_TODAY
    price: float = 101.5
    settlement_days: int = 3
    face_amount: float = 1000000.0
    base_cpi: float = 206.1
    observation_lag: Period = field(default_factory=lambda: Period(3, _MONTHS))
    interpolation: InterpolationType = InterpolationType.Flat
    coupons: Sequence[float] = (0.02,)
    accrual_day_counter: DayCounter = field(default_factory=lambda: ActualActual(AAConvention.ISDA))
    payment_convention: BusinessDayConvention = _MF
    price_type: BondPriceType = BondPriceType.Clean


CASES: dict[str, _Case] = {
    "cpi_clean": _Case(),
    "cpi_dirty": _Case(price_type=BondPriceType.Dirty),
    # earliest_date is bond.next_cash_flow_date(), so it steps as the
    # evaluation date crosses an accrual boundary.
    "cpi_eval_2010_03_31": _Case(eval_date=Date.from_ymd(31, Month.March, 2010)),
    "cpi_eval_2010_04_06": _Case(eval_date=Date.from_ymd(6, Month.April, 2010)),
    "cpi_settle0": _Case(settlement_days=0),
    "cpi_settle7": _Case(settlement_days=7),
    "cpi_interp_linear": _Case(interpolation=InterpolationType.Linear),
    "cpi_lag_8m": _Case(observation_lag=Period(8, _MONTHS)),
    "cpi_base_cpi_210": _Case(base_cpi=210.0),
    "cpi_face_2_5m": _Case(face_amount=2500000.0),
    "cpi_coupon_vector": _Case(coupons=(0.03, 0.025, 0.02)),
    "cpi_daycount_a365f": _Case(accrual_day_counter=Actual365Fixed()),
    "cpi_payment_preceding": _Case(payment_convention=BusinessDayConvention.Preceding),
    "cpi_quote_98": _Case(price=98.0),
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/ts/ratehelpers")


@pytest.fixture(autouse=True)
def _global_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin the evaluation date and restore it, and clear the fixing histories.

    The probe sets ``Settings::instance().evaluationDate()`` per case
    (probe.cpp ``emitCpi``) and the UKRPI fixing table lives in the global
    IndexManager, so both are restored here. Each case re-seeds the table
    itself, which keeps the module independent of test ordering.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _CPI_TODAY  # probe.cpp ``kCpiToday``
    try:
        yield
    finally:
        settings.evaluation_date = previous
        IndexManager().clear_histories()


class _CpiMarket:
    """probe.cpp ``makeCpiMarket(eval)`` — UKRPI history + zero-inflation curve.

    Built per case, at that case's evaluation date, and seeded only with the
    fixings that date makes unambiguously historical.

    C++ ``ZeroInflationIndex::needsForecast`` (inflationindex.cpp:197-220) tests
    the requested period against ``inflationPeriod(today - availabilityLag,
    frequency)``: earlier than the window ⇒ the fixing must be provided; later
    ⇒ forecast **even if one is stored**; inside ⇒ the stored table decides.
    Cutting the table at the month before the window keeps every case on the
    unambiguous side of that test, so it pins ``CPIBondHelper`` rather than the
    index's availability logic.
    """

    def __init__(self, eval_date: Date = _CPI_TODAY) -> None:
        ObservableSettings().evaluation_date = eval_date
        self.index = UKRPI()
        self.index.clear_fixings()
        rpi_schedule = (
            MakeSchedule()
            .from_date(Date.from_ymd(1, Month.July, 2007))
            .to(Date.from_ymd(1, Month.September, 2010))
            .with_frequency(Frequency.Monthly)
            .build()
        )
        # Last day of the month before the availability window.
        last_stored = (
            inflation_period(
                eval_date - self.index.availability_lag(), self.index.frequency()
            )[0]
            - 1
        )
        for i, fixing in enumerate(_RPI_FIXINGS):
            if rpi_schedule.date(i) <= last_stored:
                self.index.add_fixing(rpi_schedule.date(i), fixing, True)

        # Both curves keep _CPI_TODAY as their reference date whatever the
        # case's evaluation date is, so the evaluation date moves only the
        # BOND's dates.
        self.nominal: YieldTermStructureProtocol = cast(
            YieldTermStructureProtocol,
            FlatForward.from_rate(_CPI_TODAY, 0.05, _AA_ISDA),
        )
        self.index.set_zero_inflation_term_structure(
            InterpolatedZeroInflationCurve(
                reference_date=_CPI_TODAY,
                dates=[d for d, _ in _ZERO_NODES],
                rates=[r for _, r in _ZERO_NODES],
                frequency=Frequency.Monthly,
                day_counter=_AA_ISDA,
            )
        )


def _schedule() -> Schedule:
    """probe.cpp ``cpiSchedule()`` — 40 semiannual periods, 2-Oct-2007 → 2-Oct-2027.

    2 October 2027 is a Saturday, which is what makes the last CASHFLOW date
    land after the bond's own maturity date under a Following-family payment
    convention.
    """
    return (
        MakeSchedule()
        .from_date(_ISSUE)
        .to(Date.from_ymd(2, Month.October, 2027))
        .with_tenor(Period(6, _MONTHS))
        .with_calendar(UnitedKingdom())
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )


def _build(case: _Case) -> CPIBondHelper:
    """Mirror ``emitCpi`` (probe.cpp) including the per-case evaluation date."""
    market = _CpiMarket(case.eval_date)

    helper = CPIBondHelper(
        case.price,
        case.settlement_days,
        case.face_amount,
        case.base_cpi,
        case.observation_lag,
        market.index,
        case.interpolation,
        _schedule(),
        case.coupons,
        case.accrual_day_counter,
        case.payment_convention,
        _ISSUE,
        UnitedKingdom(),
        price_type=case.price_type,
    )
    helper.set_term_structure(market.nominal)
    return helper


def _iso(d: Date) -> str:
    return f"{d.year()}-{int(d.month()):02d}-{d.day_of_month():02d}"


def _assert_date(actual: Date, ref: dict[str, Any], name: str) -> None:
    assert actual.serial_number() == ref[f"{name}_serial"], f"{name}: serial"
    assert _iso(actual) == ref[f"{name}_iso"], f"{name}: ISO"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(CASES))
def test_helper_dates(key: str, cpp: dict[str, Any]) -> None:
    """The five date accessors plus the bond dates they are read from. EXACT tier.

    ``BondHelper`` assigns only ``earliestDate_`` and ``latestDate_``, so
    ``maturity_date()``, ``latest_relevant_date()`` and ``pillar_date()`` all
    fall through (bootstraphelper.hpp:179-203).
    """
    ref = cpp[key]
    helper = _build(CASES[key])
    _assert_date(helper.earliest_date(), ref, "earliest")
    _assert_date(helper.latest_date(), ref, "latest")
    _assert_date(helper.maturity_date(), ref, "maturity")
    _assert_date(helper.latest_relevant_date(), ref, "latest_relevant")
    _assert_date(helper.pillar_date(), ref, "pillar")

    bond = helper.bond()
    _assert_date(bond.maturity_date(), ref, "bond_maturity")
    _assert_date(bond.settlement_date(), ref, "bond_settlement")
    _assert_date(bond.next_cash_flow_date(), ref, "bond_next_cashflow")
    _assert_date(bond.cashflows()[-1].date(), ref, "bond_last_cashflow")
    assert len(bond.cashflows()) == ref["n_cashflows"]

    # bondhelpers.cpp:35-38 — restated as identities on the built objects.
    assert helper.latest_date() == bond.cashflows()[-1].date()
    assert helper.earliest_date() == bond.next_cash_flow_date()


def test_latest_date_is_past_the_bond_maturity(cpp: dict[str, Any]) -> None:
    """A Following-family payment convention rolls the redemption past maturity.

    The schedule's terminal date, Saturday 2 October 2027, becomes Monday the
    4th under ModifiedFollowing; ``latest_date`` therefore sits two days after
    ``bond.maturity_date()``. Under Preceding it sits one day before. A port
    that used the maturity date as ``latest_date`` matches neither.
    """
    mf = cpp["cpi_clean"]
    preceding = cpp["cpi_payment_preceding"]
    assert mf["latest_serial"] - mf["bond_maturity_serial"] == 2
    assert preceding["latest_serial"] - preceding["bond_maturity_serial"] == -1

    helper = _build(CASES["cpi_clean"])
    assert helper.latest_date() - helper.bond().maturity_date() == 2
    helper_prec = _build(CASES["cpi_payment_preceding"])
    assert helper_prec.latest_date() - helper_prec.bond().maturity_date() == -1


def test_earliest_date_moves_with_the_evaluation_date(cpp: dict[str, Any]) -> None:
    """``earliest_date`` is the next cashflow, so it steps across a payment date.

    At 25 November 2009 the next flow is the April-2010 coupon, which the
    schedule puts on Friday 2 April — Good Friday — and ModifiedFollowing rolls
    to Tuesday 6 April (Easter Monday takes the 5th). At 31 March 2010 the
    three-day settlement roll skips those same four days and lands on Wednesday
    7 April, one day PAST that payment, so the flow counts as occurred and the
    next one is the October-2010 coupon. This is the "has this flow occurred"
    boundary, not an interior date.
    """
    base = cpp["cpi_clean"]
    later = cpp["cpi_eval_2010_03_31"]
    assert later["earliest_serial"] > base["earliest_serial"]
    # Settlement is one business day past the coupon the base case is waiting on.
    assert later["bond_settlement_serial"] == base["earliest_serial"] + 1
    # ...and the flow it steps to is two coupon periods on, not one.
    assert later["earliest_iso"] == "2010-10-04"

    assert (
        _build(CASES["cpi_eval_2010_03_31"]).earliest_date()
        > _build(CASES["cpi_clean"]).earliest_date()
    )
    assert (
        _build(CASES["cpi_eval_2010_03_31"]).bond().settlement_date()
        > _build(CASES["cpi_clean"]).earliest_date()
    )


# ---------------------------------------------------------------------------
# Implied quote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(CASES))
def test_implied_quote(key: str, cpp: dict[str, Any]) -> None:
    """Clean or dirty price off the curve being bootstrapped. TIGHT tier."""
    ref = cpp[key]
    helper = _build(CASES[key])
    tight(helper.implied_quote(), ref["implied_quote"], reason=f"{key} implied quote")
    tight(helper.quote_error(), ref["quote_error"], reason=f"{key} quote error")

    bond = helper.bond()
    tight(bond.clean_price(), ref["clean_price"], reason=f"{key} clean price")
    tight(bond.dirty_price(), ref["dirty_price"], reason=f"{key} dirty price")
    tight(bond.accrued_amount(), ref["accrued_amount"], reason=f"{key} accrued")
    # Compared by NAME, not by integer. C++ is ``enum Type { Dirty, Clean }``
    # (bond.hpp:70), i.e. Dirty == 0 and Clean == 1, while this port's
    # ``BondPriceType`` (termstructures/yield_/bond_helper.py:43-47) numbers
    # them the other way round despite claiming parity. That is a pre-existing
    # divergence in a file this test does not own; the SEMANTICS — which price
    # the helper reports — are what is pinned here, and they are correct, as
    # the clean/dirty comparison above shows.
    assert helper.price_type().name == _CPP_PRICE_TYPE_NAMES[ref["price_type"]]


def test_clean_and_dirty_select_different_prices(cpp: dict[str, Any]) -> None:
    """``price_type`` is the only difference between the two base cases.

    They differ by exactly the accrued amount — which is non-zero here, so a
    helper that reported the wrong one would not pass by coincidence.
    """
    clean = cpp["cpi_clean"]
    dirty = cpp["cpi_dirty"]
    assert clean["accrued_amount"] > 0.0
    tight(clean["implied_quote"], clean["clean_price"])
    tight(dirty["implied_quote"], dirty["dirty_price"])
    tight(dirty["implied_quote"] - clean["implied_quote"], clean["accrued_amount"])

    tight(
        _build(CASES["cpi_dirty"]).implied_quote()
        - _build(CASES["cpi_clean"]).implied_quote(),
        clean["accrued_amount"],
    )


def test_face_amount_does_not_move_the_price(cpp: dict[str, Any]) -> None:
    """A price is per 100 of face, so 2.5x the notional leaves it unchanged.

    Included because it is the one non-default argument whose *absence* from
    the answer is the correct behaviour — a port that forgot to normalise by
    the notional would show up here and nowhere else.
    """
    tight(cpp["cpi_face_2_5m"]["implied_quote"], cpp["cpi_clean"]["implied_quote"])
    tight(
        _build(CASES["cpi_face_2_5m"]).implied_quote(),
        _build(CASES["cpi_clean"]).implied_quote(),
    )


def test_coupon_vector_beyond_the_first_period_is_ignored_at_this_date(
    cpp: dict[str, Any],
) -> None:
    """``cpi_coupon_vector`` prices identically to the flat 2% case.

    The vector ``{0.03, 0.025, 0.02}`` applies to the first three periods, all
    of which have already paid by the November-2009 settlement, so the priced
    flows are the same. That is the C++ answer, pinned so a port cannot "fix"
    it into a difference.
    """
    tight(cpp["cpi_coupon_vector"]["implied_quote"], cpp["cpi_clean"]["implied_quote"])
    tight(
        _build(CASES["cpi_coupon_vector"]).implied_quote(),
        _build(CASES["cpi_clean"]).implied_quote(),
    )


@pytest.mark.parametrize(
    "key",
    [
        "cpi_settle0",
        "cpi_settle7",
        "cpi_interp_linear",
        "cpi_lag_8m",
        "cpi_base_cpi_210",
        "cpi_daycount_a365f",
        "cpi_payment_preceding",
    ],
)
def test_non_default_argument_changes_the_answer(key: str, cpp: dict[str, Any]) -> None:
    """Every one of these arguments must reach the bond and move the price."""
    assert cpp[key]["implied_quote"] != cpp["cpi_clean"]["implied_quote"], key


def test_default_payment_convention_is_following() -> None:
    """CPIBondHelper defaults to Following, unlike CPIBond's ModifiedFollowing.

    bondhelpers.hpp:113 vs cpibond.hpp:54 — the helper does NOT inherit the
    bond's default. On the probe's 2-April / 2-October schedule the two roll
    identically (no adjustment ever crosses a month end), so this test uses a
    31-May / 30-November schedule instead, where 31 May 2008 goes to Monday
    2 June under Following and back to Friday 30 May under ModifiedFollowing.
    No C++ number is needed: the claim is which of two conventions the omitted
    argument picks, and both are exercised here.
    """
    month_end_schedule = (
        MakeSchedule()
        .from_date(Date.from_ymd(30, Month.November, 2007))
        .to(Date.from_ymd(31, Month.May, 2018))
        .with_tenor(Period(6, _MONTHS))
        .with_calendar(UnitedKingdom())
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )

    def _dates(convention: BusinessDayConvention | None) -> list[Date]:
        market = _CpiMarket()
        kwargs: dict[str, Any] = {} if convention is None else {
            "payment_convention": convention
        }
        helper = CPIBondHelper(
            101.5,
            3,
            1000000.0,
            206.1,
            Period(3, _MONTHS),
            market.index,
            InterpolationType.Flat,
            month_end_schedule,
            (0.02,),
            _AA_ISDA,
            issue_date=Date.from_ymd(30, Month.November, 2007),
            payment_calendar=UnitedKingdom(),
            **kwargs,
        )
        return [cf.date() for cf in helper.bond().cashflows()]

    implicit = _dates(None)
    assert implicit == _dates(BusinessDayConvention.Following)
    assert implicit != _dates(_MF)


def test_implied_quote_requires_a_term_structure() -> None:
    """C++ ``QL_REQUIRE(termStructure_ != nullptr, ...)`` (bondhelpers.cpp:54)."""
    market = _CpiMarket()
    helper = CPIBondHelper(
        101.5,
        3,
        1000000.0,
        206.1,
        Period(3, _MONTHS),
        market.index,
        InterpolationType.Flat,
        _schedule(),
        (0.02,),
        _AA_ISDA,
        _MF,
        _ISSUE,
        UnitedKingdom(),
    )
    with pytest.raises(LibraryException, match="term structure not set"):
        helper.implied_quote()
