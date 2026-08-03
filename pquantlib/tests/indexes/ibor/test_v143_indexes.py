"""Cross-validate the three indexes new in C++ QuantLib v1.43.

Probe: ``v143/indexes`` (NIBOR / SHIR / ZARONIA).

These indexes are thin constructor wrappers, so the meaningful content is the
wiring — name, fixing days, currency, fixing calendar, day counter,
business-day convention, end-of-month — plus the value/maturity roll, which is
where a wrong calendar or convention actually shows up.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.indexes.ibor.nibor import Nibor
from pquantlib.indexes.ibor.shir import Shir
from pquantlib.indexes.ibor.zaronia import Zaronia
from pquantlib.testing import reference_reader
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/indexes")


def _check_wiring(idx: Any, ref: dict[str, Any]) -> None:
    assert idx.name() == ref["name"]
    assert idx.fixing_days() == ref["fixing_days"]
    assert idx.currency().code == ref["currency_code"]
    assert idx.fixing_calendar().name() == ref["fixing_calendar"]
    assert idx.day_counter().name() == ref["day_counter"]
    assert int(idx.business_day_convention()) == ref["business_day_convention"]
    assert idx.end_of_month() == ref["end_of_month"]


def _check_dates(idx: Any, ref: dict[str, Any]) -> None:
    fixing = Date(int(ref["fixing_serial"]))
    value = idx.value_date(fixing)
    assert value.serial_number() == ref["value_date_serial"]
    assert idx.maturity_date(value).serial_number() == ref["maturity_date_serial"]


def test_nibor_3m(cpp: dict[str, Any]) -> None:
    idx = Nibor(Period(3, TimeUnit.Months))
    _check_wiring(idx, cpp["nibor_3m"])
    _check_dates(idx, cpp["nibor_3m"])


def test_nibor_6m(cpp: dict[str, Any]) -> None:
    _check_wiring(Nibor(Period(6, TimeUnit.Months)), cpp["nibor_6m"])


def test_shir(cpp: dict[str, Any]) -> None:
    idx = Shir()
    _check_wiring(idx, cpp["shir"])
    _check_dates(idx, cpp["shir"])


def test_zaronia(cpp: dict[str, Any]) -> None:
    idx = Zaronia()
    _check_wiring(idx, cpp["zaronia"])
    _check_dates(idx, cpp["zaronia"])
