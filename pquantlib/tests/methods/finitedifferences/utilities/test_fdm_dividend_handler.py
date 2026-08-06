"""Cross-validation of ``FdmDividendHandler`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/fdmdividendhandler.{hpp,cpp}
# @ v1.43 (6b57206e0).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pquantlib.cashflows.dividend import Dividend
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.utilities.fdm_dividend_handler import (
    FdmDividendHandler,
)
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date

from .conftest import as_floats


def _spot_grid(mesher: FdmMesherComposite) -> Array:
    """Grid values in physical units — ``exp(log-spot)`` at every node."""
    a = np.empty(mesher.layout().size(), dtype=np.float64)
    for it in mesher.layout().iter():
        a[it.index] = math.exp(mesher.location(it, 0))
    return a


def test_schedule_accessors(
    reference_data: dict[str, Any],
    dividends: list[Dividend],
    log_spot_mesher: FdmMesherComposite,
    today: Date,
    day_counter: DayCounter,
) -> None:
    handler = FdmDividendHandler(dividends, log_spot_mesher, today, day_counter, 0)
    for a, e in zip(
        handler.dividend_times(),
        as_floats(reference_data["dividend_handler_times"]),
        strict=True,
    ):
        tight(a, e)
    for a, e in zip(
        handler.dividends(),
        as_floats(reference_data["dividend_handler_amounts"]),
        strict=True,
    ):
        tight(a, e)
    serials = [d.serial_number() for d in handler.dividend_dates()]
    assert serials == [int(v) for v in reference_data["dividend_handler_date_serials"]]


def test_apply_first_dividend_1d(
    reference_data: dict[str, Any],
    dividends: list[Dividend],
    log_spot_mesher: FdmMesherComposite,
    today: Date,
    day_counter: DayCounter,
) -> None:
    handler = FdmDividendHandler(dividends, log_spot_mesher, today, day_counter, 0)
    a = _spot_grid(log_spot_mesher)
    handler.apply_to(a, handler.dividend_times()[0])
    expected = as_floats(reference_data["dividend_handler_1d_applied_div0"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_apply_third_dividend_1d(
    reference_data: dict[str, Any],
    dividends: list[Dividend],
    log_spot_mesher: FdmMesherComposite,
    today: Date,
    day_counter: DayCounter,
) -> None:
    handler = FdmDividendHandler(dividends, log_spot_mesher, today, day_counter, 0)
    a = _spot_grid(log_spot_mesher)
    handler.apply_to(a, handler.dividend_times()[2])
    expected = as_floats(reference_data["dividend_handler_1d_applied_div2"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_no_dividend_at_time_leaves_grid_untouched(
    reference_data: dict[str, Any],
    dividends: list[Dividend],
    log_spot_mesher: FdmMesherComposite,
    today: Date,
    day_counter: DayCounter,
) -> None:
    """The C++ lookup is an exact ``std::find`` on the dividend times."""
    handler = FdmDividendHandler(dividends, log_spot_mesher, today, day_counter, 0)
    a = _spot_grid(log_spot_mesher)
    handler.apply_to(a, 0.123456)
    expected = as_floats(reference_data["dividend_handler_1d_no_dividend"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)


def test_apply_first_dividend_2d(
    reference_data: dict[str, Any],
    dividends: list[Dividend],
    today: Date,
    day_counter: DayCounter,
) -> None:
    """The 2-D branch re-interpolates every slice along the non-equity axis."""
    mesher = FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(150.0), 11),
        Uniform1dMesher(0.0, 1.0, 4),
    )
    handler = FdmDividendHandler(dividends, mesher, today, day_counter, 0)
    a = np.empty(mesher.layout().size(), dtype=np.float64)
    for it in mesher.layout().iter():
        a[it.index] = math.exp(mesher.location(it, 0)) + 10.0 * mesher.location(it, 1)
    handler.apply_to(a, handler.dividend_times()[0])
    expected = as_floats(reference_data["dividend_handler_2d_applied_div0"])
    for actual_v, expected_v in zip(a, expected, strict=True):
        tight(float(actual_v), expected_v)
