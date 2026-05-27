"""Tests for Callability + CallabilitySchedule (data-carriers only)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.callability import (
    Callability,
    CallabilitySchedule,
    CallabilityType,
)
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def test_callability_accessors() -> None:
    price = BondPrice(101.5, BondPriceType.Clean)
    date = Date.from_ymd(15, Month.June, 2027)
    call = Callability(price, CallabilityType.Call, date)
    assert call.price() is price
    assert call.type() == CallabilityType.Call
    assert call.date() == date


def test_put_type() -> None:
    price = BondPrice(99.0, BondPriceType.Clean)
    put = Callability(price, CallabilityType.Put, Date.from_ymd(15, Month.June, 2027))
    assert put.type() == CallabilityType.Put


def test_callability_schedule_is_a_list() -> None:
    """CallabilitySchedule is ``list[Callability]`` for Python idiomatic use."""
    p1 = BondPrice(101.0, BondPriceType.Clean)
    p2 = BondPrice(99.0, BondPriceType.Clean)
    sched: CallabilitySchedule = [
        Callability(p1, CallabilityType.Call, Date.from_ymd(15, Month.June, 2026)),
        Callability(p2, CallabilityType.Put, Date.from_ymd(15, Month.June, 2027)),
    ]
    assert len(sched) == 2
    assert sched[0].type() == CallabilityType.Call
    assert sched[1].type() == CallabilityType.Put


def test_callability_with_invalid_price_raises_on_amount() -> None:
    """A BondPrice without an amount raises when used."""
    bad = BondPrice()  # no amount
    call = Callability(bad, CallabilityType.Call, Date.from_ymd(1, Month.January, 2027))
    with pytest.raises(LibraryException, match="no amount given"):
        call.price().amount()
