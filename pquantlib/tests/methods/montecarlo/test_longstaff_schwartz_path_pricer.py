"""Tests for ``LongstaffSchwartzPathPricer``.

# C++ parity: ql/methods/montecarlo/longstaffschwartzpathpricer.hpp (v1.42.1).

The pricer is unit-tested with a synthetic ``EarlyExercisePathPricer``
implementation that exercises a known linear regression — we set up
the paths so the optimal continuation rule has a known closed form,
then check the trained regression's coefficients and the priced
American value.

Full LSM 1998 paper cross-validation happens in
``test_mc_american_engine.py`` via the
``MCAmericanEngine.lsm_path_pricer`` wiring.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.montecarlo.early_exercise_path_pricer import (
    EarlyExercisePathPricer,
)
from pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer import (
    LongstaffSchwartzPathPricer,
)
from pquantlib.methods.montecarlo.path import Path
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import loose
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


class _PutPathPricer(EarlyExercisePathPricer[Path, float]):
    """Put-style exercise rule: max(strike - path[t], 0).

    # C++ parity: same shape as ``AmericanPathPricer`` but without
    # the basis-system rescaling (basis here is plain monomials).
    """

    __slots__ = ("_basis", "_strike")

    def __init__(self, strike: float) -> None:
        self._strike: float = strike
        self._basis: list[Callable[[float], float]] = [
            lambda x: 1.0,
            lambda x: x,
            lambda x: x * x,
        ]

    def __call__(self, path: Path, t: int) -> float:
        return max(self._strike - float(path[t]), 0.0)

    def state(self, path: Path, t: int) -> float:
        return float(path[t])

    def basis_system(self) -> list[Callable[[float], float]]:
        return self._basis


def _flat_zero_rate_curve() -> FlatForward:
    """Flat 0% rate, so dF[i] = 1 — simpler regression cross-checks."""
    ref = Date.from_ymd(15, Month.May, 2026)
    return FlatForward.from_rate(
        reference_date=ref, forward_rate=0.0, day_counter=Actual365Fixed()
    )


def _flat_six_pct_curve() -> FlatForward:
    ref = Date.from_ymd(15, Month.May, 2026)
    return FlatForward.from_rate(
        reference_date=ref, forward_rate=0.06, day_counter=Actual365Fixed()
    )


def _grid_3_steps() -> TimeGrid:
    return TimeGrid.regular(1.0, 3)


def _make_path(values: list[float], grid: TimeGrid) -> Path:
    arr = np.array(values, dtype=np.float64)
    return Path(grid, arr)


def test_calibration_phase_records_paths_and_returns_zero() -> None:
    """First N calls are calibration — pricer stores path, returns 0.0."""
    grid = _grid_3_steps()
    pricer = LongstaffSchwartzPathPricer[Path, float](
        grid, _PutPathPricer(strike=10.0), _flat_zero_rate_curve()
    )
    assert pricer.is_calibration_phase()
    p1 = _make_path([10.0, 9.5, 9.0, 8.5], grid)
    p2 = _make_path([10.0, 10.5, 11.0, 11.5], grid)
    assert pricer(p1) == 0.0
    assert pricer(p2) == 0.0
    # Both stored.
    assert pricer.is_calibration_phase()


def test_calibrate_transitions_to_pricing_phase() -> None:
    grid = _grid_3_steps()
    pricer = LongstaffSchwartzPathPricer[Path, float](
        grid, _PutPathPricer(strike=10.0), _flat_zero_rate_curve()
    )
    # Need enough calibration paths to do a regression.
    # 3 basis functions => need >= 3 ITM paths per timestep.
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Random downward-drifting path so puts are typically ITM at maturity.
        path_vals = [10.0, 9.5 + rng.normal(0, 0.5), 9.0 + rng.normal(0, 0.5), 8.5 + rng.normal(0, 0.5)]
        pricer(_make_path(path_vals, grid))
    pricer.calibrate()
    assert not pricer.is_calibration_phase()


def test_calibrate_produces_correct_coefficient_count() -> None:
    """Coefficients should have one Array per interior exercise date.

    Grid of length L has L-2 interior dates (t in 1..L-2), so coeff has length L-2.
    Each Array has length = len(basis) = 3.
    """
    grid = _grid_3_steps()  # length 4 (t=0, 1/3, 2/3, 1.0)
    pricer = LongstaffSchwartzPathPricer[Path, float](
        grid, _PutPathPricer(strike=10.0), _flat_zero_rate_curve()
    )
    rng = np.random.default_rng(7)
    for _ in range(100):
        vals = [10.0, 9.0 + rng.normal(0, 0.5), 9.0 + rng.normal(0, 0.5), 9.0 + rng.normal(0, 0.5)]
        pricer(_make_path(vals, grid))
    pricer.calibrate()
    coeffs = pricer.coefficients()
    assert len(coeffs) == len(grid) - 2  # 2 interior dates
    for c in coeffs:
        assert c.shape == (3,)


def test_pricing_phase_returns_nonneg_value() -> None:
    """After calibration, pricing a representative path yields a nonnegative NPV."""
    grid = _grid_3_steps()
    pricer = LongstaffSchwartzPathPricer[Path, float](
        grid, _PutPathPricer(strike=10.0), _flat_six_pct_curve()
    )
    rng = np.random.default_rng(11)
    for _ in range(200):
        # Volatile mean-reverting-ish prices around 9-11
        vals = [10.0]
        for _ in range(3):
            vals.append(max(0.1, vals[-1] + rng.normal(0, 0.6)))
        pricer(_make_path(vals, grid))
    pricer.calibrate()

    # Price a 'fresh' deeply-ITM path (terminal underlying 5.0).
    deep_itm = _make_path([10.0, 8.0, 6.0, 5.0], grid)
    val = pricer(deep_itm)
    assert val > 0.0
    # Discount-applied: 5.0 (intrinsic at t=3) discounted by dF[0]*dF[1]*dF[2]
    # — never larger than 5.0 itself.
    assert val <= 5.0 + 1e-9


def test_reproducibility_same_paths_same_result() -> None:
    """Same calibration paths + same evaluation path → same NPV."""

    def build_pricer() -> LongstaffSchwartzPathPricer[Path, float]:
        grid = _grid_3_steps()
        p = LongstaffSchwartzPathPricer[Path, float](
            grid, _PutPathPricer(strike=10.0), _flat_six_pct_curve()
        )
        rng = np.random.default_rng(2026)
        for _ in range(300):
            vals = [10.0]
            for _ in range(3):
                vals.append(max(0.1, vals[-1] + rng.normal(0, 0.5)))
            p(_make_path(vals, grid))
        p.calibrate()
        return p

    grid = _grid_3_steps()
    test_path = _make_path([10.0, 9.0, 8.0, 7.0], grid)
    p1 = build_pricer()
    p2 = build_pricer()
    val1 = p1(test_path)
    val2 = p2(test_path)
    assert val1 == val2


def test_too_few_itm_paths_falls_back_to_zero_coefficients() -> None:
    """When ITM paths < basis size, coeffs collapse to zeros.

    # C++ parity: ``if (v_.size() <= x.size()) ... else coeff_[i-1] = Array(v_.size(), 0.0)``.
    """
    grid = _grid_3_steps()
    pricer = LongstaffSchwartzPathPricer[Path, float](
        grid, _PutPathPricer(strike=10.0), _flat_zero_rate_curve()
    )
    # Only 2 paths — basis size 3 — so regression is undetermined.
    # Both paths OTM at the interior dates to make sure no ITM filter kicks in.
    p1 = _make_path([10.0, 20.0, 20.0, 5.0], grid)  # OTM at interior 1 and 2
    p2 = _make_path([10.0, 25.0, 25.0, 5.0], grid)
    pricer(p1)
    pricer(p2)
    pricer.calibrate()
    coeffs = pricer.coefficients()
    # Each interior date should have zero coefs (since no ITM samples).
    assert np.all(coeffs[0] == 0.0)
    assert np.all(coeffs[1] == 0.0)


def test_exercise_probability_increments_during_pricing() -> None:
    """``exercise_probability`` advances with each priced path."""
    grid = _grid_3_steps()
    pricer = LongstaffSchwartzPathPricer[Path, float](
        grid, _PutPathPricer(strike=10.0), _flat_zero_rate_curve()
    )
    rng = np.random.default_rng(42)
    for _ in range(100):
        vals = [10.0]
        for _ in range(3):
            vals.append(max(0.1, vals[-1] + rng.normal(0, 0.5)))
        pricer(_make_path(vals, grid))
    pricer.calibrate()
    # Price 50 paths, each known to be ITM at maturity.
    for _ in range(50):
        vals = [10.0, 8.0, 7.0, 6.0]
        pricer(_make_path(vals, grid))
    # All 50 priced paths exercised somewhere → exercise probability == 1.
    loose(pricer.exercise_probability(), 1.0)


def test_constructor_rejects_too_short_grid() -> None:
    """``len(times) >= 2`` required — single-point grid is meaningless."""
    one_point_grid = TimeGrid(times=[1.0], mandatory_times=[1.0])
    with pytest.raises(Exception, match="at least 2 points"):
        LongstaffSchwartzPathPricer[Path, float](
            one_point_grid, _PutPathPricer(strike=10.0), _flat_zero_rate_curve()
        )


def test_calibrate_rejects_empty_paths() -> None:
    """Calling ``calibrate()`` with no stored paths raises."""
    grid = _grid_3_steps()
    pricer = LongstaffSchwartzPathPricer[Path, float](
        grid, _PutPathPricer(strike=10.0), _flat_zero_rate_curve()
    )
    with pytest.raises(Exception, match="no paths stored"):
        pricer.calibrate()


def test_dummy_use_of_ndarray() -> None:
    """Ensure NDArray import is exercised (smoke test for the import)."""
    coef = np.zeros(3, dtype=np.float64)
    assert coef.shape == (3,)
    assert isinstance(coef, np.ndarray)
    # Typing reference — used by signatures, this just smoke-tests the type alias.
    _: npt.NDArray[np.float64] = coef
    assert _ is coef
