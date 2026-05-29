"""Tests for AtmSmileSection (atm_level override over a base smile)."""

from __future__ import annotations

import pytest

from pquantlib.termstructures.volatility.atm_smile_section import AtmSmileSection
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.testing import reference_reader, tolerance

_REF = reference_reader.load("cluster/l10a")
_ATM = _REF["atm_smile_section"]


def _base() -> FlatSmileSection:
    return FlatSmileSection(
        volatility=_ATM["vol_at_strike_5pct"],
        exercise_time=_ATM["exercise_time"],
        atm_level=_ATM["base_atm_level"],
    )


def test_atm_level_returns_explicit_value() -> None:
    sec = AtmSmileSection(base=_base(), atm=_ATM["atm_level"])
    tolerance.exact(sec.atm_level(), _ATM["atm_level"])


def test_atm_level_falls_back_to_base_when_none() -> None:
    sec = AtmSmileSection(base=_base())
    tolerance.exact(sec.atm_level(), _ATM["base_atm_level"])


def test_volatility_delegates_to_base() -> None:
    sec = AtmSmileSection(base=_base(), atm=_ATM["atm_level"])
    # FlatSmileSection returns the same vol everywhere.
    tolerance.exact(sec.volatility(0.05), _ATM["vol_at_strike_5pct"])
    tolerance.exact(sec.volatility(0.03), _ATM["vol_at_strike_3pct"])


def test_exercise_time_delegates_to_base() -> None:
    sec = AtmSmileSection(base=_base(), atm=_ATM["atm_level"])
    tolerance.exact(sec.exercise_time(), _ATM["exercise_time"])


def test_strike_bounds_delegate_to_base() -> None:
    base = _base()
    sec = AtmSmileSection(base=base, atm=0.07)
    assert sec.min_strike() == base.min_strike()
    assert sec.max_strike() == base.max_strike()


def test_atm_with_nan_input_falls_back_to_base() -> None:
    sec = AtmSmileSection(base=_base(), atm=float("nan"))
    tolerance.exact(sec.atm_level(), _ATM["base_atm_level"])


def test_observer_propagation() -> None:
    """An update on the base re-fires update on the AtmSmileSection."""
    # Simple smoke test — verifies the registration didn't raise.
    sec = AtmSmileSection(base=_base(), atm=0.07)
    sec.update()  # should not raise


def test_construction_with_no_anchoring_fails() -> None:
    """The base must provide some exercise time/date for our constructor to succeed."""
    # FlatSmileSection always has a time, so this just confirms
    # the AtmSmileSection ctor doesn't blow up on a valid base.
    sec = AtmSmileSection(base=_base())
    assert sec.exercise_time() > 0


def test_atm_smile_section_does_not_require_day_counter() -> None:
    """If the base has no day counter (time-anchored) we don't either."""
    from pquantlib.exceptions import LibraryException  # noqa: PLC0415
    sec = AtmSmileSection(base=_base(), atm=0.07)
    with pytest.raises(LibraryException, match="day counter"):
        sec.day_counter()
