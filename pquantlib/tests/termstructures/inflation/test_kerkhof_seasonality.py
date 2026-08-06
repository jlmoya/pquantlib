"""KerkhofSeasonality, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/kerkhof.json
Probe:     migration-harness/cpp/probes/v143_ts_kerkhof/probe.cpp

Kerkhof inherits from ``MultiplicativePriceSeasonality`` and replaces BOTH the
factor lookup and the correction kernel, so the probe emits both classes over
the same dates: wherever the two columns differ, a port that subclassed and
forgot one of the overrides fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.inflation.seasonality import (
    KerkhofSeasonality,
    MultiplicativePriceSeasonality,
)
from pquantlib.testing import tolerance
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

_REF_PATH = (
    Path(__file__).resolve().parents[4]
    / "migration-harness/references/v143/ts/kerkhof.json"
)

_BASE = Date.from_ymd(1, Month.June, 2023)
_FACTORS = [
    1.0100, 1.0021, 0.9985, 1.0043, 0.9971, 1.0008,
    1.0032, 0.9994, 1.0017, 0.9962, 1.0055, 0.9979,
]


@pytest.fixture(scope="module")
def refs() -> dict[str, list[float]]:
    raw = cast("dict[str, object]", json.loads(_REF_PATH.read_text()))
    return {k: cast("list[float]", v) for k, v in raw.items()}


def test_kerkhof_factor_matches_cpp(refs: dict[str, list[float]]) -> None:
    seasonality = KerkhofSeasonality(_BASE, _FACTORS)
    for serial, expected in zip(refs["dates"], refs["kerkhof_factor"], strict=True):
        tolerance.tight(
            seasonality.seasonality_factor(Date(int(serial))),
            expected,
            reason=str(int(serial)),
        )


def test_multiplicative_factor_still_matches_cpp(refs: dict[str, list[float]]) -> None:
    """Guard: the base class must not have been changed by the subclass."""
    seasonality = MultiplicativePriceSeasonality(_BASE, Frequency.Monthly, _FACTORS)
    for serial, expected in zip(
        refs["dates"], refs["multiplicative_factor"], strict=True
    ):
        tolerance.tight(
            seasonality.seasonality_factor(Date(int(serial))),
            expected,
            reason=str(int(serial)),
        )


def test_the_two_kernels_actually_differ(refs: dict[str, list[float]]) -> None:
    """If they agreed everywhere the test above would prove nothing."""
    differing = sum(
        1
        for a, b in zip(refs["kerkhof_factor"], refs["multiplicative_factor"], strict=True)
        if a != b
    )
    assert differing >= 8


def test_kerkhof_requires_exactly_twelve_factors() -> None:
    """24 factors is a legal multi-year cycle for the base class, not for Kerkhof.

    The base class only requires a multiple of the frequency, so this reaches
    Kerkhof's own check rather than being rejected at construction.
    """
    seasonality = KerkhofSeasonality(_BASE, _FACTORS + _FACTORS)
    with pytest.raises(LibraryException, match="12 monthly seasonal factors"):
        seasonality.seasonality_factor(Date.from_ymd(1, Month.December, 2023))
