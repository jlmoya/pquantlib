"""Tests for the 8 concrete IBOR / overnight indexes.

Cross-validates default-market parameters against the probe JSON at
``migration-harness/references/cluster/l2c.json``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.europe import EURCurrency, GBPCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.eonia import Eonia
from pquantlib.indexes.ibor.estr import Estr
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.indexes.ibor.fed_funds import FedFunds
from pquantlib.indexes.ibor.gbp_libor import GBPLibor
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.ibor.sonia import Sonia
from pquantlib.indexes.ibor.usd_libor import USDLibor
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from tests.indexes._mock_curves import FlatForwardMock


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l2c")


_FactoryFn = Callable[[], IborIndex]

def _usd_libor_3m() -> USDLibor:
    return USDLibor(Period(3, TimeUnit.Months))


def _gbp_libor_6m() -> GBPLibor:
    return GBPLibor(Period(6, TimeUnit.Months))


# Map probe key → factory.
_CONCRETES: list[tuple[str, _FactoryFn]] = [
    ("euribor_3m", Euribor.three_months),
    ("euribor_6m", Euribor.six_months),
    ("euribor_1y", Euribor.one_year),
    ("euribor_1w", Euribor.one_week),
    ("usd_libor_3m", _usd_libor_3m),
    ("gbp_libor_6m", _gbp_libor_6m),
    ("eonia", Eonia),
    ("sofr", Sofr),
    ("sonia", Sonia),
    ("fed_funds", FedFunds),
    ("estr", Estr),
]


@pytest.mark.parametrize(("key", "factory"), _CONCRETES)
def test_concrete_default_market_params(
    ref: dict[str, Any],
    key: str,
    factory: _FactoryFn,
) -> None:
    idx = factory()
    expected = ref[key]
    assert idx.name() == expected["name"], f"{key} name"
    assert idx.family_name() == expected["familyName"], f"{key} familyName"
    assert idx.tenor() == Period(
        int(expected["tenor_length"]), TimeUnit(int(expected["tenor_units"])),
    ), f"{key} tenor"
    assert idx.fixing_days() == expected["fixingDays"], f"{key} fixingDays"
    assert idx.currency().code == expected["currencyCode"], f"{key} currency"
    assert idx.day_counter().name() == expected["dayCounterName"], f"{key} dayCounter"
    expected_bdc = BusinessDayConvention[str(expected["businessDayConvention"])]
    assert idx.business_day_convention() == expected_bdc, f"{key} bdc"
    assert idx.end_of_month() == expected["endOfMonth"], f"{key} eom"


def test_euribor_3m_classmethod_matches_period_constructor() -> None:
    a = Euribor.three_months()
    b = Euribor(Period(3, TimeUnit.Months))
    assert a.name() == b.name()
    assert a.tenor() == b.tenor()


def test_euribor_rejects_daily_tenor() -> None:
    with pytest.raises(LibraryException):
        Euribor(Period(1, TimeUnit.Days))


def test_euribor_3m_fixing_against_probe(ref: dict[str, Any]) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = Euribor.three_months(ts)
    expected = ref["euribor_3m_fixing"]
    fix = eval_date
    value = idx.value_date(fix)
    maturity = idx.maturity_date(value)
    # Date serial numbers ↔ excel-day-count.
    assert value.serial == expected["value_serial"], "value date"
    assert maturity.serial == expected["maturity_serial"], "maturity date"
    fixing = idx.fixing(fix, forecast_todays_fixing=True)
    tight(fixing, float(expected["fixing"]))


def test_usd_libor_3m_fixing_against_probe(ref: dict[str, Any]) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = USDLibor(Period(3, TimeUnit.Months), ts)
    expected = ref["usd_libor_3m_fixing"]
    fix = eval_date
    value = idx.value_date(fix)
    maturity = idx.maturity_date(value)
    assert value.serial == expected["value_serial"]
    assert maturity.serial == expected["maturity_serial"]
    fixing = idx.fixing(fix, forecast_todays_fixing=True)
    tight(fixing, float(expected["fixing"]))


def test_sofr_fixing_against_probe(ref: dict[str, Any]) -> None:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    ts = FlatForwardMock(eval_date, 0.05, Actual360())
    idx = Sofr(ts)
    expected = ref["sofr_fixing"]
    fix = eval_date
    value = idx.value_date(fix)
    maturity = idx.maturity_date(value)
    assert value.serial == expected["value_serial"]
    assert maturity.serial == expected["maturity_serial"]
    fixing = idx.fixing(fix, forecast_todays_fixing=True)
    tight(fixing, float(expected["fixing"]))


def test_currency_inspectors() -> None:
    assert isinstance(Euribor.three_months().currency(), EURCurrency)
    assert isinstance(USDLibor(Period(3, TimeUnit.Months)).currency(), USDCurrency)
    assert isinstance(GBPLibor(Period(6, TimeUnit.Months)).currency(), GBPCurrency)
    assert isinstance(Sofr().currency(), USDCurrency)
    assert isinstance(Sonia().currency(), GBPCurrency)
    assert isinstance(FedFunds().currency(), USDCurrency)
    assert isinstance(Eonia().currency(), EURCurrency)
    assert isinstance(Estr().currency(), EURCurrency)
