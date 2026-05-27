"""Tests for the Position enum."""

from __future__ import annotations

from pquantlib.position import PositionType


def test_position_long_is_zero() -> None:
    assert int(PositionType.Long) == 0


def test_position_short_is_one() -> None:
    assert int(PositionType.Short) == 1


def test_position_long_str_matches_cpp() -> None:
    assert str(PositionType.Long) == "Long"


def test_position_short_str_matches_cpp() -> None:
    assert str(PositionType.Short) == "Short"
