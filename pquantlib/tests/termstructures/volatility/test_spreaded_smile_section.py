"""Tests for SpreadedSmileSection (base smile + additive vol spread)."""

from __future__ import annotations

from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.spreaded_smile_section import (
    SpreadedSmileSection,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance

_REF = reference_reader.load("cluster/l9c")
_SPR = _REF["spreaded_smile_section"]


# --- construction ------------------------------------------------------


def test_spreaded_smile_section_construction() -> None:
    base = FlatSmileSection(
        exercise_time=_SPR["exercise_time"],
        volatility=_SPR["base_vol"],
        atm_level=_SPR["atm_level"],
    )
    spread = SimpleQuote(_SPR["spread"])
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    tolerance.exact(section.exercise_time(), _SPR["exercise_time"])
    tolerance.exact(section.atm_level(), _SPR["atm_level"])


def test_spreaded_smile_section_vol_at_atm_matches_probe() -> None:
    base = FlatSmileSection(
        exercise_time=_SPR["exercise_time"],
        volatility=_SPR["base_vol"],
        atm_level=_SPR["atm_level"],
    )
    spread = SimpleQuote(_SPR["spread"])
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    tolerance.tight(section.volatility(0.05), _SPR["spreaded_vol_at_strike_5pct"])


def test_spreaded_smile_section_vol_at_strike_3pct_matches_probe() -> None:
    base = FlatSmileSection(
        exercise_time=_SPR["exercise_time"],
        volatility=_SPR["base_vol"],
        atm_level=_SPR["atm_level"],
    )
    spread = SimpleQuote(_SPR["spread"])
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    tolerance.tight(section.volatility(0.03), _SPR["spreaded_vol_at_strike_3pct"])


# --- delegation --------------------------------------------------------


def test_spreaded_smile_section_delegates_volatility_type() -> None:
    base = FlatSmileSection(
        exercise_time=1.0, volatility=0.20, atm_level=0.05,
        volatility_type=VolatilityType.Normal,
    )
    spread = SimpleQuote(0.005)
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    assert section.volatility_type() == VolatilityType.Normal


def test_spreaded_smile_section_delegates_shift() -> None:
    base = FlatSmileSection(
        exercise_time=1.0, volatility=0.20, atm_level=0.05, shift=0.01,
    )
    spread = SimpleQuote(0.005)
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    assert section.shift() == 0.01


def test_spreaded_smile_section_delegates_min_max_strike() -> None:
    base = FlatSmileSection(
        exercise_time=1.0, volatility=0.20, atm_level=0.05,
    )
    spread = SimpleQuote(0.005)
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    assert section.min_strike() == base.min_strike()
    assert section.max_strike() == base.max_strike()


# --- observable behaviour ---------------------------------------------


def test_spreaded_smile_section_propagates_spread_quote_updates() -> None:
    base = FlatSmileSection(
        exercise_time=1.0, volatility=0.20, atm_level=0.05,
    )
    spread = SimpleQuote(0.005)
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    counts = [0]

    class _Counter:
        def update(self) -> None:
            counts[0] += 1

    # Pin the listener strongly to defeat the WeakSet observer storage.
    listener = _Counter()
    section.register_with(listener)
    spread.set_value(0.010)
    assert counts[0] >= 1


def test_spreaded_smile_section_updates_vol_after_spread_change() -> None:
    base = FlatSmileSection(
        exercise_time=1.0, volatility=0.20, atm_level=0.05,
    )
    spread = SimpleQuote(0.005)
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    tolerance.tight(section.volatility(0.05), 0.205)
    spread.set_value(0.010)
    tolerance.tight(section.volatility(0.05), 0.210)


# --- variance ---------------------------------------------------------


def test_spreaded_smile_section_variance_uses_spreaded_vol() -> None:
    base = FlatSmileSection(
        exercise_time=2.0, volatility=0.20, atm_level=0.05,
    )
    spread = SimpleQuote(0.005)
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    v = 0.205
    expected = v * v * 2.0
    tolerance.tight(section.variance(0.05), expected)


# --- zero spread -------------------------------------------------------


def test_spreaded_smile_section_with_zero_spread_matches_base() -> None:
    base = FlatSmileSection(
        exercise_time=1.0, volatility=0.20, atm_level=0.05,
    )
    spread = SimpleQuote(0.0)
    section = SpreadedSmileSection(base=base, vol_spread=spread)
    tolerance.exact(section.volatility(0.05), base.volatility(0.05))
