"""Tests for pquantlib.termstructures.extrapolator (Extrapolator mixin)."""

from __future__ import annotations

from pquantlib.termstructures.extrapolator import Extrapolator


def test_extrapolator_disabled_by_default() -> None:
    e = Extrapolator()
    assert e.allows_extrapolation() is False


def test_enable_extrapolation_toggles_on() -> None:
    e = Extrapolator()
    e.enable_extrapolation()
    assert e.allows_extrapolation() is True


def test_enable_extrapolation_with_false_turns_off() -> None:
    e = Extrapolator()
    e.enable_extrapolation()
    e.enable_extrapolation(False)
    assert e.allows_extrapolation() is False


def test_disable_extrapolation_toggles_off() -> None:
    e = Extrapolator()
    e.enable_extrapolation()
    e.disable_extrapolation()
    assert e.allows_extrapolation() is False
