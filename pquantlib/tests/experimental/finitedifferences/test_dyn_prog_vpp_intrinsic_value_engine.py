"""Cross-validate :class:`DynProgVPPIntrinsicValueEngine` against C++.

# C++ parity reference:
# ql/experimental/finitedifferences/dynprogvppintrinsicvalueengine.{hpp,cpp}
# v1.42.1 + test-suite/vpp.cpp `testVPPIntrinsicValue`.

The test case is identical to the C++ ``testVPPIntrinsicValue``:

* VPP parameters: pMin=8, pMax=40, tMinUp=tMinDown=2, startUpFuel=20,
  startUpFixCost=100, fuelCostAddon=3.
* Swing exercise: hourly, 7 days (168 instants).
* Efficiency sweep: ``[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.90]``.
* Each efficiency yields heat_rate = 1/efficiency.

Expected values are the C++ reference NPVs (which themselves match the
mixed-integer LP reference from the Spanderen blog post). LOOSE tier
because the closed-form DP NPV uses 168 hourly steps with floating-point
maxima accumulated through each step — the C++ vs Python numbers
diverge only in the ULP-tail (small numerical noise) but easily within
the LOOSE tolerance for power-market dollar-magnitude values.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.finitedifferences.dyn_prog_vpp_intrinsic_value_engine import (
    DynProgVPPIntrinsicValueEngine,
)
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.experimental.finitedifferences.vanilla_vpp_option import (
    VanillaVPPOption,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# Canonical fuel/power price arrays from C++ test-suite/vpp.cpp.
_FUEL_PRICES: list[float] = [
    20.74, 21.65, 20.78, 21.58, 21.43, 20.82, 22.02, 21.52, 21.02, 21.46,
    21.75, 20.69, 22.16, 20.38, 20.82, 20.68, 20.57, 21.92, 22.04, 20.45,
    20.75, 21.92, 20.53, 20.67, 20.88, 21.02, 20.82, 21.67, 21.82, 22.12,
    20.45, 20.74, 22.39, 20.95, 21.71, 20.70, 20.94, 21.59, 22.33, 21.13,
    21.50, 21.42, 20.56, 21.23, 21.37, 21.90, 20.62, 21.17, 21.86, 22.04,
    22.05, 21.00, 20.70, 21.12, 21.26, 22.40, 21.31, 22.24, 21.96, 21.02,
    21.71, 20.48, 21.36, 21.75, 21.90, 20.44, 21.26, 22.29, 20.34, 21.79,
    21.66, 21.50, 20.76, 20.27, 20.84, 20.24, 21.97, 20.52, 20.98, 21.40,
    20.39, 20.71, 20.78, 20.30, 21.56, 21.72, 20.27, 21.57, 21.82, 20.57,
    21.33, 20.51, 22.32, 21.99, 20.57, 22.11, 21.56, 22.24, 20.62, 21.70,
    21.11, 21.19, 21.79, 20.46, 22.21, 20.82, 20.52, 22.29, 20.71, 21.45,
    22.40, 20.63, 20.95, 21.97, 22.20, 20.67, 21.01, 22.25, 20.76, 21.33,
    20.49, 20.33, 21.94, 20.64, 20.99, 21.09, 20.97, 22.17, 20.72, 22.06,
    20.86, 21.40, 21.75, 20.78, 21.79, 20.47, 21.19, 21.60, 20.75, 21.36,
    21.61, 20.37, 21.67, 20.28, 22.33, 21.37, 21.33, 20.87, 21.25, 22.01,
    22.08, 20.81, 20.70, 21.84, 21.82, 21.68, 21.24, 22.36, 20.83, 20.64,
    21.03, 20.57, 22.34, 20.96, 21.54, 21.26, 21.43, 22.39,
]

_POWER_PRICES: list[float] = [
    40.40, 36.71, 31.87, 25.81, 31.61, 35.00, 46.22, 60.68, 42.45, 38.01,
    33.84, 29.79, 31.84, 38.53, 49.23, 59.92, 43.85, 37.47, 34.89, 29.99,
    30.85, 29.19, 29.25, 38.67, 36.90, 25.93, 22.12, 20.19, 17.19, 19.29,
    13.51, 18.14, 33.76, 30.48, 25.63, 18.01, 23.86, 32.41, 48.56, 64.69,
    38.42, 39.31, 32.73, 29.97, 31.41, 35.02, 46.85, 58.12, 39.14, 35.42,
    32.61, 28.76, 29.41, 35.83, 46.73, 61.41, 61.01, 59.43, 60.43, 66.29,
    62.79, 62.66, 57.66, 51.63, 62.18, 60.53, 61.94, 64.86, 59.57, 58.15,
    53.74, 48.36, 45.64, 51.21, 51.54, 50.79, 54.50, 49.92, 41.58, 39.81,
    28.86, 37.42, 39.78, 42.36, 45.67, 36.84, 33.91, 28.75, 62.97, 63.84,
    62.91, 68.77, 64.33, 61.95, 59.12, 54.89, 63.62, 60.90, 66.57, 69.51,
    64.71, 59.89, 57.28, 57.10, 65.09, 63.82, 67.52, 70.51, 65.59, 59.36,
    58.22, 54.64, 52.17, 53.02, 57.12, 53.50, 53.16, 49.21, 52.21, 40.96,
    49.01, 47.94, 49.89, 53.83, 52.96, 50.33, 51.72, 46.99, 39.06, 47.99,
    47.91, 52.35, 48.51, 47.39, 50.45, 43.66, 25.62, 35.76, 42.76, 46.51,
    45.62, 46.79, 48.76, 41.00, 52.65, 55.57, 57.67, 56.79, 55.15, 54.74,
    50.31, 47.49, 53.72, 55.62, 55.89, 58.11, 54.46, 52.92, 49.61, 44.68,
    51.59, 57.44, 56.50, 55.12, 57.22, 54.61, 49.92, 45.20,
]


@pytest.fixture(scope="module")
def today() -> Date:
    return Date.from_ymd(18, Month.December, 2011)


@pytest.fixture(scope="module")
def day_counter() -> ActualActual:
    return ActualActual(Convention.ISDA)


@pytest.fixture(scope="module")
def r_ts(today: Date, day_counter: ActualActual) -> FlatForward:
    """Zero-rate flat curve at 0% — matches the C++ test setup."""
    return FlatForward.from_rate(today, 0.0, day_counter)


@pytest.fixture(scope="module")
def exercise_168h(today: Date) -> SwingExercise:
    """168 hourly exercise instants (7 days, 1h step) — C++ test setup."""
    ex = SwingExercise.from_range(today, today + 6, 3600)
    # Sanity: matches C++ probe's exercise length.
    assert len(ex.dates()) == 168
    return ex


@pytest.mark.parametrize(
    "case_idx", [0, 1, 2, 3, 4, 5, 6], ids=[
        "eff_0.35", "eff_0.40", "eff_0.45", "eff_0.50",
        "eff_0.55", "eff_0.60", "eff_0.90",
    ],
)
def test_dynprog_npv_matches_cpp_per_efficiency(
    case_idx: int,
    cpp_ref: dict[str, Any],
    exercise_168h: SwingExercise,
    r_ts: FlatForward,
) -> None:
    """NPV for the i-th efficiency case matches the C++ ``testVPPIntrinsicValue``
    reference value at LOOSE tolerance. The first case (efficiency=0.35) is
    exactly zero (cold VPP regime); the rest are positive 4- and 5-digit
    dollar values.

    The C++ test uses ``1e-4`` absolute tolerance; the Python LOOSE tier
    (1e-8 abs/rel) is tighter — confirming exact ULP-tail matching of the
    DP algorithm.
    """
    ref = cpp_ref["dynprog_intrinsic"]
    efficiencies = ref["efficiencies"]
    expected_npvs = ref["npvs"]
    eff = efficiencies[case_idx]
    expected = expected_npvs[case_idx]
    heat_rate = 1.0 / eff

    option = VanillaVPPOption(
        heat_rate=heat_rate,
        p_min=ref["p_min"],
        p_max=ref["p_max"],
        t_min_up=ref["t_min_up"],
        t_min_down=ref["t_min_down"],
        start_up_fuel=ref["start_up_fuel"],
        start_up_fix_cost=ref["start_up_fix_cost"],
        exercise=exercise_168h,
    )
    engine = DynProgVPPIntrinsicValueEngine(
        fuel_prices=_FUEL_PRICES,
        power_prices=_POWER_PRICES,
        fuel_cost_addon=ref["fuel_cost_addon"],
        r_ts=r_ts,
    )
    option.set_pricing_engine(engine)
    npv = option.npv()
    # LOOSE tier (1e-8) is enough — the algorithm is closed-form DP so the
    # Python NPV matches the C++ NPV essentially bit-identically across
    # 168 backward steps.
    loose(npv, expected, reason="DP closed-form NPV across 168 hourly steps")


def test_dynprog_zero_efficiency_case_is_zero(
    cpp_ref: dict[str, Any],
    exercise_168h: SwingExercise,
    r_ts: FlatForward,
) -> None:
    """Efficiency=0.35 yields heat_rate ~ 2.857 — the plant is unprofitable
    across the entire price path and NPV = 0 (no exercise is taken).
    """
    ref = cpp_ref["dynprog_intrinsic"]
    option = VanillaVPPOption(
        heat_rate=1.0 / 0.35,
        p_min=ref["p_min"], p_max=ref["p_max"],
        t_min_up=ref["t_min_up"], t_min_down=ref["t_min_down"],
        start_up_fuel=ref["start_up_fuel"],
        start_up_fix_cost=ref["start_up_fix_cost"],
        exercise=exercise_168h,
    )
    engine = DynProgVPPIntrinsicValueEngine(
        fuel_prices=_FUEL_PRICES,
        power_prices=_POWER_PRICES,
        fuel_cost_addon=ref["fuel_cost_addon"],
        r_ts=r_ts,
    )
    option.set_pricing_engine(engine)
    tight(option.npv(), 0.0)


def test_dynprog_rejects_unequal_price_arrays(
    today: Date, day_counter: ActualActual, r_ts: FlatForward,
) -> None:
    """Constructor raises if fuel + power arrays differ in length."""
    with pytest.raises(LibraryException):
        DynProgVPPIntrinsicValueEngine(
            fuel_prices=_FUEL_PRICES,
            power_prices=_POWER_PRICES[:-1],
            fuel_cost_addon=3.0,
            r_ts=r_ts,
        )
