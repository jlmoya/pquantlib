"""Cross-validate DefaultProbKey against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.currencies.europe import EURCurrency
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.default_probability_key import (
    DefaultProbKey,
    make_north_america_corp_default_key,
)
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    FailureToPay,
    Restructuring,
    Seniority,
)
from pquantlib.testing import reference_reader
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def test_north_america_corp_default_key_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    usd = USDCurrency()
    key = make_north_america_corp_default_key(
        usd, Seniority.SnrFor, Period(30, TimeUnit.Days), 1.0e6, Restructuring.CR  # type: ignore[attr-defined]
    )
    # TIGHT (structural): currency code + seniority idx + size of event_types.
    assert key.currency.code == cpp_ref["default_prob_key"]["currency_code"]
    assert int(key.seniority) == cpp_ref["default_prob_key"]["seniority_idx"]
    assert key.size() == cpp_ref["default_prob_key"]["size"]


def test_default_prob_key_no_restructuring_skips_restructuring_entry() -> None:
    usd = USDCurrency()
    key = make_north_america_corp_default_key(
        usd, Seniority.SnrFor, restructuring_type=Restructuring.NoRestructuring
    )
    # Without Restructuring, only FailureToPay + Bankruptcy → 2 entries.
    assert key.size() == 2


def test_default_prob_key_equality() -> None:
    usd = USDCurrency()
    a = DefaultProbKey(
        event_types=(
            DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
        ),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    b = DefaultProbKey(
        event_types=(
            DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring),
        ),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    assert a == b
    assert hash(a) == hash(b)
    # Different seniority -> not equal.
    c = DefaultProbKey(
        event_types=a.event_types,
        currency=usd,
        seniority=Seniority.PrefT1,
    )
    assert a != c
    # Different currency -> not equal.
    d = DefaultProbKey(
        event_types=a.event_types,
        currency=EURCurrency(),
        seniority=Seniority.SnrFor,
    )
    assert a != d


def test_default_prob_key_set_equality_event_order_invariant() -> None:
    usd = USDCurrency()
    e1 = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    e2 = DefaultType(AtomicDefault.FailureToPay, Restructuring.NoRestructuring)
    a = DefaultProbKey(event_types=(e1, e2), currency=usd, seniority=Seniority.SnrFor)
    b = DefaultProbKey(event_types=(e2, e1), currency=usd, seniority=Seniority.SnrFor)
    # # C++ parity: set-equality on event_types (defaultprobabilitykey.cpp:44-57).
    assert a == b


def test_default_prob_key_rejects_duplicate_atomic_types() -> None:
    usd = USDCurrency()
    e1 = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    e2 = DefaultType(AtomicDefault.Bankruptcy, Restructuring.FullRestructuring)
    with pytest.raises(LibraryException, match="Duplicated event type"):
        DefaultProbKey(
            event_types=(e1, e2),
            currency=usd,
            seniority=Seniority.SnrFor,
        )


def test_make_north_america_corp_default_key_contents() -> None:
    usd = USDCurrency()
    key = make_north_america_corp_default_key(
        usd, Seniority.SnrFor, Period(30, TimeUnit.Days), 1.0e6, Restructuring.CR  # type: ignore[attr-defined]
    )
    # First entry is FailureToPay with the configured grace/amount.
    ftp = key.event_types[0]
    assert isinstance(ftp, FailureToPay)
    assert ftp.grace_period == Period(30, TimeUnit.Days)
    assert ftp.amount_required == 1.0e6
    # Second is Bankruptcy/XR.
    bk = key.event_types[1]
    assert bk.default_type == AtomicDefault.Bankruptcy
    assert bk.restructuring_type == Restructuring.NoRestructuring
    # Third is Restructuring/CR.
    rs = key.event_types[2]
    assert rs.default_type == AtomicDefault.Restructuring
    assert rs.restructuring_type == Restructuring.FullRestructuring
