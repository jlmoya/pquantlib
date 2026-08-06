"""Cross-validation of ``FdmQuantoHelper`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmquantohelper.{hpp,cpp}
# @ v1.43 (6b57206e0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import (
    FdmQuantoHelper,
)
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

from .conftest import as_floats


@pytest.fixture
def helper(today: Date, day_counter: DayCounter, calendar: Calendar) -> FdmQuantoHelper:
    """Probe setup: r = 5%, f = 3%, fx vol = 15%, rho = -0.75, ATM = 1.25."""
    r_ts = FlatForward.from_rate(today, 0.05, day_counter)
    f_ts = FlatForward.from_rate(today, 0.03, day_counter)
    fx_vol_ts = BlackConstantVol(
        reference_date=today, calendar=calendar, volatility=0.15, day_counter=day_counter
    )
    return FdmQuantoHelper(r_ts, f_ts, fx_vol_ts, -0.75, 1.25)


@pytest.mark.parametrize(
    ("equity_vol", "t1", "t2", "key"),
    [
        (0.25, 0.5, 1.5, "quanto_adjustment_scalar_v025"),
        (0.40, 0.0, 1.0, "quanto_adjustment_scalar_v040"),
    ],
)
def test_scalar_adjustment(
    reference_data: dict[str, Any],
    helper: FdmQuantoHelper,
    equity_vol: float,
    t1: float,
    t2: float,
    key: str,
) -> None:
    tight(helper.quanto_adjustment(equity_vol, t1, t2), float(reference_data[key]))


def test_array_adjustment(reference_data: dict[str, Any], helper: FdmQuantoHelper) -> None:
    vols = np.array([0.10, 0.20, 0.30, 0.45], dtype=np.float64)
    actual = helper.quanto_adjustment_array(vols, 0.5, 1.5)
    expected = as_floats(reference_data["quanto_adjustment_array"])
    for a, e in zip(actual, expected, strict=True):
        tight(float(a), e)


def test_array_adjustment_matches_scalar(helper: FdmQuantoHelper) -> None:
    """# C++ parity: the Array overload is the scalar one under ``std::transform``."""
    vols = np.array([0.10, 0.20, 0.30, 0.45], dtype=np.float64)
    vector = helper.quanto_adjustment_array(vols, 0.5, 1.5)
    for i, vol in enumerate(vols):
        tight(float(vector[i]), helper.quanto_adjustment(float(vol), 0.5, 1.5))
