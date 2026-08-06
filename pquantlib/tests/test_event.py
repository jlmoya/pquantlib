"""Tests for ql/event.{hpp,cpp} and the ql/cashflow.cpp ``hasOccurred`` override.

Every expected value comes from ``migration-harness/cpp/probes/v143_root_tail/probe.cpp``
run against C++ QuantLib v1.43; the reference lives at
``migration-harness/references/v143/root/tail.json``.

The probe sets ``Settings::instance().evaluationDate()`` (probe.cpp, ``emitHasOccurred``,
`Settings::instance().evaluationDate() = EVAL_DATE;`) plus
``includeReferenceDateEvents`` and ``includeTodaysCashFlows``. Every one of those
globals is pinned here by the ``_pinned_settings`` autouse fixture and restored in
teardown via :class:`SavedSettings`, so the module is NOT wall-clock dependent.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.event import Event
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings, SavedSettings
from pquantlib.testing import reference_reader
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/root/tail")


@pytest.fixture(autouse=True)
def _pinned_settings(cpp: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Pin every global the probe touches; SavedSettings puts them all back."""
    guard = SavedSettings()
    settings = ObservableSettings()
    settings.evaluation_date = Date(int(cpp["has_occurred"]["eval_date_serial"]))
    try:
        yield
    finally:
        guard.restore()


class _ExCouponFlow(SimpleCashFlow):
    """Mirror of the probe's ``ExCouponFlow`` (probe.cpp, ``emitTradingExCoupon``)."""

    def __init__(self, amount: float, date: Date, ex_coupon: Date) -> None:
        super().__init__(amount, date)
        self._ex_coupon = ex_coupon

    def ex_coupon_date(self) -> Date:
        return self._ex_coupon


def _ref_date(name: str, cpp: dict[str, Any]) -> Date | None:
    cf_serial = int(cpp["has_occurred"]["cf_date_serial"])
    eval_serial = int(cpp["has_occurred"]["eval_date_serial"])
    match name:
        case "null":
            return None
        case "eval_date":
            return Date(eval_serial)
        case "before":
            return Date(cf_serial - 1)
        case "equal":
            return Date(cf_serial)
        case "after":
            return Date(cf_serial + 1)
        case _:
            raise AssertionError(f"unknown ref name {name}")


def test_has_occurred_truth_table(cpp: dict[str, Any]) -> None:
    """All 90 (settings x refDate x includeRefDate) rows, for Event and CashFlow.

    ``Event.has_occurred`` is invoked unbound so the ``CashFlow`` override is
    bypassed — the C++ probe does the same with ``ev.Event::hasOccurred(...)``.
    """
    section = cpp["has_occurred"]
    settings = ObservableSettings()
    cf = SimpleCashFlow(1000.0, Date(int(section["cf_date_serial"])))

    assert len(section["rows"]) == 90
    for row in section["rows"]:
        settings.include_reference_date_events = row["settings_include_ref_date_events"]
        settings.include_todays_cash_flows = row["settings_include_todays_cash_flows"]
        ref = _ref_date(row["ref"], cpp)
        arg = row["arg_include_ref_date"]

        assert Event.has_occurred(cf, ref, arg) is row["event"], f"Event row {row}"
        assert cf.has_occurred(ref, arg) is row["cashflow"], f"CashFlow row {row}"


def test_null_date_is_the_same_sentinel_as_none(cpp: dict[str, Any]) -> None:
    """C++ spells "no reference date" as ``Date()``; Python also accepts ``None``."""
    section = cpp["has_occurred"]
    settings = ObservableSettings()
    settings.include_reference_date_events = False
    settings.include_todays_cash_flows = None
    cf = SimpleCashFlow(1000.0, Date(int(section["cf_date_serial"])))
    assert cf.has_occurred(Date()) == cf.has_occurred(None)
    assert Event.has_occurred(cf, Date()) == Event.has_occurred(cf, None)


def test_trading_ex_coupon(cpp: dict[str, Any]) -> None:
    section = cpp["trading_ex_coupon"]
    ex_date = Date(int(section["ex_date_serial"]))
    cf_date = Date(int(cpp["has_occurred"]["cf_date_serial"]))

    with_ex = _ExCouponFlow(1000.0, cf_date, ex_date)
    without_ex = SimpleCashFlow(1000.0, cf_date)

    assert without_ex.trading_ex_coupon(None) is section["no_ex_coupon_date_null_ref"]
    assert with_ex.trading_ex_coupon(None) is section["null_ref"]
    assert with_ex.trading_ex_coupon(Date(ex_date.serial_number() - 1)) is section["before_ex"]
    assert with_ex.trading_ex_coupon(ex_date) is section["on_ex"]
    assert with_ex.trading_ex_coupon(Date(ex_date.serial_number() + 2)) is section["after_ex"]


def test_cash_flow_is_an_event() -> None:
    """C++: ``class CashFlow : public Event`` (ql/cashflow.hpp:41)."""
    assert issubclass(SimpleCashFlow, Event)


def test_event_accept_requires_a_visitor() -> None:
    """C++ ``Event::accept`` QL_FAILs when the visitor is not a ``Visitor<Event>``."""
    seen: list[object] = []

    class Recorder:
        def visit(self, target: object) -> None:
            seen.append(target)

    cf = SimpleCashFlow(1.0, Date.from_ymd(1, Month.March, 2025))
    cf.accept(Recorder())
    assert seen == [cf]

    with pytest.raises(LibraryException):
        cf.accept(object())  # pyright: ignore[reportArgumentType]
