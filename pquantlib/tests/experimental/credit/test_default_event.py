"""Cross-validate DefaultEvent against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.default_event import (
    BankruptcyEvent,
    DefaultEvent,
    DefaultSettlement,
    FailureToPayEvent,
    make_isda_conv_map,
)
from pquantlib.experimental.credit.default_probability_key import (
    make_north_america_corp_default_key,
)
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    FailureToPay,
    Restructuring,
    Seniority,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def _make_probe_event() -> DefaultEvent:
    usd = USDCurrency()
    credit_date = Date.from_ymd(15, Month.January, 2024)
    settle_date = Date.from_ymd(20, Month.January, 2024)
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    return DefaultEvent.from_rate(
        credit_date, dt, usd, Seniority.SnrFor, settle_date, 0.4
    )


def test_default_event_accessors_match_cpp(cpp_ref: dict[str, Any]) -> None:
    ev = _make_probe_event()
    ref = cpp_ref["default_event"]

    # TIGHT (structural): serial-date / currency code / seniority / type fields.
    assert ev.date().serial_number() == ref["date_serial"]
    assert ev.currency().code == ref["currency_code"]
    assert int(ev.event_seniority()) == ref["seniority_idx"]
    assert int(ev.default_type().default_type) == ref["default_type_idx"]
    assert (
        int(ev.default_type().restructuring_type) == ref["restructuring_type_idx"]
    )
    assert ev.is_restructuring() == ref["is_restructuring"]
    assert ev.is_default() == ref["is_default"]
    assert ev.has_settled() == ref["has_settled"]
    assert ev.settlement().date.serial_number() == ref["settlement_date_serial"]
    # TIGHT: closed-form recovery looked up by seniority key.
    rate = ev.recovery_rate(Seniority.SnrFor)
    assert rate is not None
    tolerance.tight(rate, ref["recovery_rate_snrfor"])


def test_default_event_matches_default_key(cpp_ref: dict[str, Any]) -> None:
    usd = USDCurrency()
    key = make_north_america_corp_default_key(
        usd, Seniority.SnrFor, Period(30, TimeUnit.Days), 1.0e6, Restructuring.CR  # type: ignore[attr-defined]
    )
    credit_date = Date.from_ymd(15, Month.January, 2024)
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    ev = DefaultEvent(credit_date, dt, usd, Seniority.SnrFor)
    assert ev.matches_default_key(key) == cpp_ref["default_event_matches_key"]


def test_default_event_eur_vs_usd_key_mismatch(cpp_ref: dict[str, Any]) -> None:
    usd = USDCurrency()
    eur = EURCurrency()
    key = make_north_america_corp_default_key(
        usd, Seniority.SnrFor, Period(30, TimeUnit.Days), 1.0e6, Restructuring.CR  # type: ignore[attr-defined]
    )
    credit_date = Date.from_ymd(15, Month.January, 2024)
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    ev = DefaultEvent(credit_date, dt, eur, Seniority.SnrFor)
    assert (
        ev.matches_default_key(key)
        == cpp_ref["default_event_eur_matches_usd_key"]
    )


def test_default_event_equality_ignores_settlement() -> None:
    usd = USDCurrency()
    credit_date = Date.from_ymd(15, Month.January, 2024)
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    settle1 = Date.from_ymd(20, Month.January, 2024)
    settle2 = Date.from_ymd(25, Month.January, 2024)
    a = DefaultEvent.from_rate(credit_date, dt, usd, Seniority.SnrFor, settle1, 0.4)
    b = DefaultEvent.from_rate(credit_date, dt, usd, Seniority.SnrFor, settle2, 0.5)
    # # C++ parity: operator== compares (currency, type, date, seniority) only.
    assert a == b
    assert hash(a) == hash(b)


def test_default_event_lt_orders_by_date() -> None:
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    a = DefaultEvent(Date.from_ymd(1, Month.January, 2024), dt, usd, Seniority.SnrFor)
    b = DefaultEvent(Date.from_ymd(2, Month.January, 2024), dt, usd, Seniority.SnrFor)
    assert a < b


def test_unsettled_event_recovery_rate_is_none() -> None:
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    ev = DefaultEvent(Date.from_ymd(15, Month.January, 2024), dt, usd, Seniority.SnrFor)
    assert ev.recovery_rate(Seniority.SnrFor) is None
    assert not ev.has_settled()


def test_settlement_no_seniority_overrides_all_isda_recoveries() -> None:
    settle = DefaultSettlement.from_seniority(
        Date.from_ymd(20, Month.January, 2024), Seniority.NoSeniority, 0.5
    )
    # Every seniority in ISDA map should now be 0.5.
    for sen in (
        Seniority.SecDom,
        Seniority.SnrFor,
        Seniority.SubLT2,
        Seniority.JrSubT2,
        Seniority.PrefT1,
    ):
        rate = settle.recovery_rate(sen)
        assert rate is not None
        tolerance.tight(rate, 0.5)


def test_settlement_specific_seniority_uses_isda_base_map() -> None:
    settle = DefaultSettlement.from_seniority(
        Date.from_ymd(20, Month.January, 2024), Seniority.SnrFor, 0.55
    )
    # SnrFor is overridden, others remain at the ISDA conventional value.
    rate = settle.recovery_rate(Seniority.SnrFor)
    assert rate is not None
    tolerance.tight(rate, 0.55)
    rate2 = settle.recovery_rate(Seniority.SecDom)
    assert rate2 is not None
    tolerance.tight(rate2, 0.65)


def test_settlement_rejects_no_seniority_request() -> None:
    settle = DefaultSettlement.from_seniority(
        Date.from_ymd(20, Month.January, 2024), Seniority.SnrFor, 0.55
    )
    with pytest.raises(LibraryException, match="NoSeniority is not valid"):
        settle.recovery_rate(Seniority.NoSeniority)


def test_settlement_from_map_rejects_no_seniority_in_map() -> None:
    bad_map = {Seniority.NoSeniority: 0.3}
    with pytest.raises(LibraryException, match="NoSeniority"):
        DefaultSettlement.from_map(Date.from_ymd(20, Month.January, 2024), bad_map)


def test_isda_conv_map_returns_fresh_copy() -> None:
    m1 = make_isda_conv_map()
    m2 = make_isda_conv_map()
    assert m1 == m2
    m1[Seniority.SnrFor] = 0.99
    assert m2[Seniority.SnrFor] == 0.4


def test_settlement_date_after_credit_date_required() -> None:
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    credit_date = Date.from_ymd(15, Month.January, 2024)
    earlier_settle = Date.from_ymd(10, Month.January, 2024)
    with pytest.raises(LibraryException, match="Settlement date"):
        DefaultEvent.from_rate(
            credit_date, dt, usd, Seniority.SnrFor, earlier_settle, 0.4
        )


def test_failure_to_pay_event_amount_threshold() -> None:
    usd = USDCurrency()
    ftp_contract = FailureToPay(
        grace_period=Period(30, TimeUnit.Days), amount_required=1.0e6
    )
    bigger = FailureToPayEvent(
        Date.from_ymd(15, Month.January, 2024), usd, Seniority.SnrFor, 2.0e6
    )
    smaller = FailureToPayEvent(
        Date.from_ymd(15, Month.January, 2024), usd, Seniority.SnrFor, 5.0e5
    )
    assert bigger.matches_event_type(ftp_contract) is True
    assert smaller.matches_event_type(ftp_contract) is False


def test_failure_to_pay_event_only_matches_failure_to_pay() -> None:
    usd = USDCurrency()
    ev = FailureToPayEvent(
        Date.from_ymd(15, Month.January, 2024), usd, Seniority.SnrFor, 2.0e6
    )
    bk = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    assert ev.matches_event_type(bk) is False


def test_bankruptcy_event_matches_anything() -> None:
    usd = USDCurrency()
    bk_ev = BankruptcyEvent(
        Date.from_ymd(15, Month.January, 2024), usd, Seniority.SnrFor
    )
    for et in (
        DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
        DefaultType(AtomicDefault.FailureToPay, Restructuring.NoRestructuring),
        DefaultType(AtomicDefault.Restructuring, Restructuring.FullRestructuring),
    ):
        assert bk_ev.matches_event_type(et) is True


def test_has_occurred_semantics() -> None:
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    credit_date = Date.from_ymd(15, Month.January, 2024)
    ev = DefaultEvent(credit_date, dt, usd, Seniority.SnrFor)
    # No ref_date → False.
    assert ev.has_occurred(None) is False
    # Later ref_date → True.
    assert ev.has_occurred(Date.from_ymd(20, Month.January, 2024)) is True
    # Earlier ref_date → False.
    assert ev.has_occurred(Date.from_ymd(10, Month.January, 2024)) is False
    # Equal ref_date: include_ref_date controls. Default False → not include → True.
    assert ev.has_occurred(credit_date) is True
    assert ev.has_occurred(credit_date, include_ref_date=True) is False
