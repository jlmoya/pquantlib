"""Shared fixtures for the ql/models/volatility/ cross-validation tests.

# C++ parity: reference values come from
# ``migration-harness/cpp/probes/v143_models_volatility/probe.cpp``, emitted to
# ``migration-harness/references/v143/models/volatility.json``.

NO EVALUATION-DATE FIXTURE, deliberately. The probe sets no
``Settings::instance().evaluationDate()`` — see the probe's header comment
("NO EVALUATION DATE IS SET", probe.cpp:18-24) — because nothing in this family
reads it: the estimators are pure functions of a ``TimeSeries`` and ``Garch11``
only does arithmetic on the series' own date keys. Pinning one here would be
noise pretending to be rigour.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.prices import IntervalPrice
from pquantlib.testing import reference_reader
from pquantlib.time.date import Date
from pquantlib.time.time_series import TimeSeries


@pytest.fixture(scope="session")
def cpp() -> dict[str, Any]:
    """The whole probe output. # C++ parity: probe.cpp ``main``."""
    return reference_reader.load("v143/models/volatility")


def _ohlc(block: dict[str, Any]) -> TimeSeries[IntervalPrice]:
    dates = [Date(serial) for serial in block["date_serials"]]
    return IntervalPrice.make_series(
        dates, block["open"], block["close"], block["high"], block["low"]
    )


@pytest.fixture(scope="session")
def ohlc(cpp: dict[str, Any]) -> TimeSeries[IntervalPrice]:
    """The consistent OHLC bars — four distinct fields on every bar."""
    return _ohlc(cpp["ohlc"])


@pytest.fixture(scope="session")
def ohlc_degenerate(cpp: dict[str, Any]) -> TimeSeries[IntervalPrice]:
    """Bars with ``high == low == open != close`` — drives Sigma4/Sigma5 negative."""
    return _ohlc(cpp["ohlc_degenerate"])


@pytest.fixture(scope="session")
def closes(cpp: dict[str, Any]) -> TimeSeries[float]:
    """The close component of :func:`ohlc`, as a plain ``TimeSeries[float]``."""
    dates = [Date(serial) for serial in cpp["ohlc"]["date_serials"]]
    return TimeSeries[float].from_pairs(dates, cpp["ohlc"]["close"])


@pytest.fixture(scope="session")
def year_fraction(cpp: dict[str, Any]) -> float:
    return float(cpp["year_fraction"])


@pytest.fixture(scope="session")
def market_open_fraction(cpp: dict[str, Any]) -> float:
    return float(cpp["market_open_fraction"])


@pytest.fixture(scope="session")
def vol_series(cpp: dict[str, Any]) -> TimeSeries[float]:
    """The hand-written volatility series ``ConstantEstimator`` is fed."""
    dates = [Date(serial) for serial in cpp["vol_series"]["date_serials"]]
    return TimeSeries[float].from_pairs(dates, cpp["vol_series"]["values"])


def assert_series_matches(
    actual: TimeSeries[float],
    expected: dict[str, Any],
    tier: Any,
) -> None:
    """Compare the FULL series — every date AND every value — against the probe.

    Comparing only the values would let an off-by-one in which date an
    estimate is filed under pass unnoticed, which is precisely the defect
    ``ConstantEstimator`` and ``GarmanKlassOpenClose`` are shaped to produce.
    """
    assert [d.serial for d in actual.dates()] == expected["date_serials"]
    values = list(actual.values())
    assert len(values) == len(expected["values"])
    for got, want in zip(values, expected["values"], strict=True):
        tier(got, want)
