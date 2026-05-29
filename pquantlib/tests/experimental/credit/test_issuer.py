"""Cross-validate Issuer against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.default_event import (
    BankruptcyEvent,
    DefaultEvent,
)
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    Restructuring,
    Seniority,
)
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def _make_issuer_and_key() -> tuple[Issuer, DefaultProbKey, Date]:
    usd = USDCurrency()
    today = Date.from_ymd(15, Month.January, 2024)
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    et = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(et,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    issuer = Issuer(probabilities=[(key, curve)])
    return issuer, key, today


def test_issuer_default_probability_round_trips(cpp_ref: dict[str, Any]) -> None:
    issuer, key, _today = _make_issuer_and_key()
    curve = issuer.default_probability(key)
    # TIGHT: closed-form exp(-h*t) on FlatHazardRate with h=0.02.
    tolerance.tight(curve.survival_probability(1.0), cpp_ref["issuer"]["curve_survival_t1"])
    tolerance.tight(curve.survival_probability(5.0), cpp_ref["issuer"]["curve_survival_t5"])


def test_issuer_unknown_key_raises() -> None:
    issuer, _key, _today = _make_issuer_and_key()
    other_key = DefaultProbKey(
        event_types=(
            DefaultType(AtomicDefault.FailureToPay, Restructuring.NoRestructuring),
        ),
        currency=USDCurrency(),
        seniority=Seniority.SnrFor,
    )
    with pytest.raises(LibraryException, match="not available"):
        issuer.default_probability(other_key)


def test_issuer_events_are_sorted_by_date() -> None:
    usd = USDCurrency()
    e_late = BankruptcyEvent(
        Date.from_ymd(15, Month.June, 2024), usd, Seniority.SnrFor
    )
    e_early = BankruptcyEvent(
        Date.from_ymd(15, Month.January, 2024), usd, Seniority.SnrFor
    )
    issuer = Issuer(events=[e_late, e_early])
    events = issuer.events()
    assert events[0].date() == e_early.date()
    assert events[1].date() == e_late.date()


def test_issuer_defaulted_between_returns_first_match() -> None:
    usd = USDCurrency()
    today = Date.from_ymd(15, Month.January, 2024)
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    et = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(et,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    e1 = BankruptcyEvent(
        Date.from_ymd(15, Month.February, 2024), usd, Seniority.SnrFor
    )
    e2 = BankruptcyEvent(
        Date.from_ymd(15, Month.March, 2024), usd, Seniority.SnrFor
    )
    issuer = Issuer(probabilities=[(key, curve)], events=[e1, e2])

    found = issuer.defaulted_between(
        Date.from_ymd(1, Month.January, 2024),
        Date.from_ymd(31, Month.March, 2024),
        key,
    )
    assert found is not None
    assert found.date() == e1.date()  # first match in date order


def test_issuer_defaulted_between_no_match_returns_none() -> None:
    usd = USDCurrency()
    today = Date.from_ymd(15, Month.January, 2024)
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    et = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(et,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    e1 = BankruptcyEvent(
        Date.from_ymd(15, Month.April, 2024), usd, Seniority.SnrFor
    )
    issuer = Issuer(probabilities=[(key, curve)], events=[e1])
    # window before the event
    assert (
        issuer.defaulted_between(
            Date.from_ymd(1, Month.January, 2024),
            Date.from_ymd(31, Month.March, 2024),
            key,
        )
        is None
    )


def test_issuer_defaults_between_returns_all_matches() -> None:
    usd = USDCurrency()
    today = Date.from_ymd(15, Month.January, 2024)
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    et = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(et,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    e1 = BankruptcyEvent(
        Date.from_ymd(15, Month.February, 2024), usd, Seniority.SnrFor
    )
    e2 = BankruptcyEvent(
        Date.from_ymd(15, Month.March, 2024), usd, Seniority.SnrFor
    )
    # non-matching seniority (mismatch the key)
    e3 = BankruptcyEvent(
        Date.from_ymd(15, Month.April, 2024), usd, Seniority.PrefT1
    )
    issuer = Issuer(probabilities=[(key, curve)], events=[e1, e2, e3])

    defaults = issuer.defaults_between(
        Date.from_ymd(1, Month.January, 2024),
        Date.from_ymd(30, Month.April, 2024),
        key,
    )
    # e3 should be excluded since seniority mismatches the key.
    assert len(defaults) == 2
    assert defaults[0].date() == e1.date()
    assert defaults[1].date() == e2.date()


def test_issuer_from_components_round_trips() -> None:
    usd = USDCurrency()
    today = Date.from_ymd(15, Month.January, 2024)
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    bankruptcy_type = DefaultType(
        AtomicDefault.Bankruptcy, Restructuring.NoRestructuring
    )
    issuer = Issuer.from_components(
        event_types=[[bankruptcy_type]],
        currencies=[usd],
        seniorities=[Seniority.SnrFor],
        curves=[curve],
    )
    # constructed key should match a DefaultProbKey built from the same components
    expected_key = DefaultProbKey(
        event_types=(bankruptcy_type,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    retrieved = issuer.default_probability(expected_key)
    assert retrieved is curve


def test_issuer_from_components_size_mismatch_raises() -> None:
    usd = USDCurrency()
    today = Date.from_ymd(15, Month.January, 2024)
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    bankruptcy_type = DefaultType(
        AtomicDefault.Bankruptcy, Restructuring.NoRestructuring
    )
    with pytest.raises(LibraryException, match="Incompatible size"):
        Issuer.from_components(
            event_types=[[bankruptcy_type], [bankruptcy_type]],
            currencies=[usd],
            seniorities=[Seniority.SnrFor],
            curves=[curve],
        )


def test_default_event_set_alias_is_list() -> None:
    # Just a sanity check that we can use the alias in callers' type hints.
    usd = USDCurrency()
    events: list[DefaultEvent] = [
        BankruptcyEvent(Date.from_ymd(1, Month.January, 2024), usd, Seniority.SnrFor)
    ]
    issuer = Issuer(events=events)
    assert len(issuer.events()) == 1
