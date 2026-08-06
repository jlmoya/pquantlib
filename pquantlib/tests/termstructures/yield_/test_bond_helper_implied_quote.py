"""BondHelper / FixedRateBondHelper, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/bondhelper.json
Probe:     migration-harness/cpp/probes/v143_ts_bondhelper/probe.cpp

``latest_date`` is the last CASHFLOW date, not the bond's maturity date: the
schedule in the probe matures on a Saturday, so the Following payment
convention rolls the redemption two days past maturity and the two come apart.
``earliest_date`` is the next cashflow after settlement, so it moves with the
evaluation date.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.instruments.bonds.zero_coupon_bond import ZeroCouponBond
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.bond_helper import (
    BondHelper,
    BondPriceType,
    FixedRateBondHelper,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule

_REF_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration-harness/references/v143/ts/bondhelper.json"
)

_EVAL = Date.from_ymd(17, Month.January, 2024)
_THIRTY360 = Thirty360(Thirty360Convention.BondBasis)


@pytest.fixture(autouse=True)
def _evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """The probe pins Settings::evaluationDate; Bond.settlement_date reads it."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = _EVAL
    try:
        yield
    finally:
        settings.evaluation_date = previous


def _curve() -> YieldTermStructureProtocol:
    curve = FlatForward.from_rate(
        _EVAL, 0.035, Actual365Fixed(), Compounding.Continuous, Frequency.Annual
    )
    curve.enable_extrapolation()
    return cast("YieldTermStructureProtocol", curve)


def _annual_schedule() -> Schedule:
    return Schedule.from_rule(
        Date.from_ymd(15, Month.June, 2020),
        Date.from_ymd(15, Month.June, 2030),
        Period.from_frequency(Frequency.Annual),
        TARGET(),
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Backward,
        False,
    )


def _semi_schedule() -> Schedule:
    return Schedule.from_rule(
        Date.from_ymd(1, Month.March, 2021),
        Date.from_ymd(1, Month.March, 2029),
        Period.from_frequency(Frequency.Semiannual),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )


def _builders() -> dict[str, Callable[[], BondHelper]]:
    return {
        "fixed_rate_clean": lambda: FixedRateBondHelper(
            SimpleQuote(101.5), 3, 100.0, _annual_schedule(), [0.04], _THIRTY360,
            BusinessDayConvention.Following, 100.0,
            Date.from_ymd(15, Month.June, 2020), TARGET(), BondPriceType.Clean,
        ),
        "fixed_rate_dirty": lambda: FixedRateBondHelper(
            SimpleQuote(101.5), 3, 100.0, _annual_schedule(), [0.04], _THIRTY360,
            BusinessDayConvention.Following, 100.0,
            Date.from_ymd(15, Month.June, 2020), TARGET(), BondPriceType.Dirty,
        ),
        "fixed_rate_settle0": lambda: FixedRateBondHelper(
            SimpleQuote(101.5), 0, 100.0, _annual_schedule(), [0.04], _THIRTY360,
            BusinessDayConvention.Following, 100.0,
            Date.from_ymd(15, Month.June, 2020), TARGET(), BondPriceType.Clean,
        ),
        "fixed_rate_semi_act365": lambda: FixedRateBondHelper(
            SimpleQuote(98.25), 2, 1000.0, _semi_schedule(), [0.025], Actual365Fixed(),
            BusinessDayConvention.ModifiedFollowing, 100.0,
            Date.from_ymd(1, Month.March, 2021), TARGET(), BondPriceType.Clean,
        ),
        "zero_coupon_clean": lambda: BondHelper(
            SimpleQuote(82.0),
            ZeroCouponBond(
                2, TARGET(), 100.0, Date.from_ymd(15, Month.June, 2030),
                BusinessDayConvention.Following, 100.0,
                Date.from_ymd(15, Month.June, 2020),
            ),
            BondPriceType.Clean,
        ),
    }


@pytest.fixture(scope="module")
def refs() -> dict[str, dict[str, float]]:
    raw = cast("dict[str, object]", json.loads(_REF_PATH.read_text()))
    return {k: cast("dict[str, float]", v) for k, v in raw.items()}


def test_every_probe_case_is_covered(refs: dict[str, dict[str, float]]) -> None:
    assert set(_builders()) == set(refs)


@pytest.mark.parametrize("key", sorted(_builders()))
def test_dates_match_cpp(refs: dict[str, dict[str, float]], key: str) -> None:
    helper = _builders()[key]()
    helper.set_term_structure(_curve())
    block = refs[key]
    assert helper.earliest_date().serial == block["earliest_date"], key
    assert helper.latest_date().serial == block["latest_date"], key
    assert helper.pillar_date().serial == block["pillar_date"], key
    assert helper.bond().maturity_date().serial == block["bond_maturity_date"], key
    assert helper.bond().settlement_date().serial == block["bond_settlement_date"], key
    assert len(helper.bond().cashflows()) == block["n_cashflows"], key


def test_latest_date_outruns_maturity_where_cpp_says_so(
    refs: dict[str, dict[str, float]],
) -> None:
    """The discriminating case: the redemption payment rolls past maturity.

    A helper that used the bond's maturity date instead of its last cashflow
    date would agree everywhere except here.
    """
    rolled = {
        k for k, v in refs.items() if v["latest_date"] > v["bond_maturity_date"]
    }
    assert rolled == {
        "fixed_rate_clean",
        "fixed_rate_dirty",
        "fixed_rate_settle0",
        "zero_coupon_clean",
    }
    for key in rolled:
        helper = _builders()[key]()
        assert helper.latest_date() > helper.bond().maturity_date()


@pytest.mark.parametrize("key", sorted(_builders()))
def test_implied_quote_matches_cpp(refs: dict[str, dict[str, float]], key: str) -> None:
    helper = _builders()[key]()
    helper.set_term_structure(_curve())
    tolerance.tight(helper.implied_quote(), refs[key]["implied_quote"], reason=key)
    tolerance.tight(helper.quote_error(), refs[key]["quote_error"], reason=key)
