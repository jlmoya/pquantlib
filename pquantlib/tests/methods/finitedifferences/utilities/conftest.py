"""Shared fixtures for the FD ``utilities`` cross-validation tests.

Every expected value comes from ``migration-harness/references/v143/methods/utilities.json``,
produced by ``migration-harness/cpp/probes/v143_methods_utilities/probe.cpp`` run
against C++ QuantLib v1.43 (submodule commit ``6b57206e0``). The fixtures below
rebuild exactly the inputs that probe uses, so the tests compare like with like.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from pquantlib.cashflows.dividend import Dividend, dividend_vector
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.utilities.escrowed_dividend_adjustment import (
    EscrowedDividendAdjustment,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    """The C++ v1.43 probe output for this cluster."""
    return load_reference("v143/methods/utilities")


@pytest.fixture
def today() -> Date:
    """Probe evaluation date: 15 January 2024."""
    return Date.from_ymd(15, Month.January, 2024)


@pytest.fixture
def day_counter() -> DayCounter:
    """Probe day counter: Actual/365 (Fixed)."""
    return Actual365Fixed()


@pytest.fixture
def calendar() -> Calendar:
    """Probe calendar: NullCalendar."""
    return NullCalendar()


@pytest.fixture
def small_mesher() -> FdmMesherComposite:
    """The 5x4 mesh the boundary-condition probe blocks use."""
    return FdmMesherComposite(Uniform1dMesher(-2.0, 2.0, 5), Uniform1dMesher(-1.0, 1.0, 4))


@pytest.fixture
def log_spot_mesher() -> FdmMesherComposite:
    """11-node log-spot mesh spanning ``[log 50, log 150]``."""
    return FdmMesherComposite(Uniform1dMesher(math.log(50.0), math.log(150.0), 11))


@pytest.fixture
def dividends(today: Date) -> list[Dividend]:
    """The three-dividend schedule the probe uses."""
    return dividend_vector([today + 90, today + 250, today + 500], [2.5, 3.0, 4.0])


@pytest.fixture
def escrowed_adjustment(
    today: Date, day_counter: DayCounter, dividends: list[Dividend]
) -> EscrowedDividendAdjustment:
    """``EscrowedDividendAdjustment`` with the probe's 5%/2% curves, T = 1."""
    r_ts = FlatForward.from_rate(today, 0.05, day_counter)
    q_ts = FlatForward.from_rate(today, 0.02, day_counter)
    return EscrowedDividendAdjustment(
        dividends, r_ts, q_ts, lambda d: day_counter.year_fraction(today, d), 1.0
    )


def ramp(n: int) -> Array:
    """``[1, 2, ..., n]`` — the probe's stand-in payload array."""
    return np.arange(1, n + 1, dtype=np.float64)


def as_floats(values: Sequence[Any]) -> list[float]:
    """Coerce a reference JSON array to ``list[float]``."""
    return [float(v) for v in values]
