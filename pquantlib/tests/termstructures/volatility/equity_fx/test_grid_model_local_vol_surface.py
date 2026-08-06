"""FixedLocalVolSurface + GridModelLocalVolSurface, cross-validated vs v1.43.

Reference: migration-harness/references/v143/eqfx/gridlocalvol.json
Probe:     migration-harness/cpp/probes/v143_eqfx_gridlocalvol/probe.cpp

The fixture gives each time slice a DIFFERENT strike range — [80, 120],
[60, 140], [90, 110] — so that "last slice", "union of slices" and "first
slice" are three different answers to ``min_strike()`` / ``max_strike()``.
C++ reads the last slice.

Queries run at times ON a node, just off it, between nodes and outside the
grid, because the strike-extrapolation policy is bypassed on a node: there
C++ extrapolates linearly whatever the policy says.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.grid_model_local_vol_surface import (
    GridModelLocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_REF = reference_reader.load("v143/eqfx/gridlocalvol")

_DC = Actual365Fixed()
_REF_DATE = Date.from_ymd(15, Month.June, 2026)

_PER_SLICE_STRIKES: list[list[float]] = [
    [80.0, 100.0, 120.0],
    [60.0, 100.0, 140.0],
    [90.0, 100.0, 110.0],
]
_SHARED_STRIKES: list[float] = [80.0, 100.0, 120.0]
_TIMES: list[float] = [0.25, 0.75, 1.5]
_DATES = [
    Date.from_ymd(15, Month.September, 2026),
    Date.from_ymd(15, Month.March, 2027),
    Date.from_ymd(15, Month.December, 2027),
]

_VOL_MATRIX = np.asarray(
    [
        [0.25, 0.30, 0.34],
        [0.20, 0.22, 0.24],
        [0.22, 0.27, 0.29],
    ],
    dtype=np.float64,
)

# Mirrors the probe's kQueryTimes / kQueryStrikes label tables.
_QUERY_TIMES: list[tuple[str, float]] = [
    ("t0p1_before_grid", 0.1),
    ("t0p25_node0", 0.25),
    ("t0p2501_just_after_node0", 0.2501),
    ("t0p5_between", 0.5),
    ("t0p75_node1", 0.75),
    ("t1p0_between", 1.0),
    ("t1p5_node2", 1.5),
    ("t2p0_after_grid", 2.0),
]
_QUERY_STRIKES: list[tuple[str, float]] = [
    ("k50", 50.0),
    ("k70", 70.0),
    ("k85", 85.0),
    ("k100", 100.0),
    ("k105", 105.0),
    ("k130", 130.0),
    ("k150", 150.0),
]

_CONSTANT = FixedLocalVolSurface.Extrapolation.ConstantExtrapolation
_INTERPOLATOR = FixedLocalVolSurface.Extrapolation.InterpolatorDefaultExtrapolation
_POLICIES: dict[
    str, tuple[FixedLocalVolSurface.Extrapolation, FixedLocalVolSurface.Extrapolation]
] = {
    "constant_constant": (_CONSTANT, _CONSTANT),
    "interpolator_interpolator": (_INTERPOLATOR, _INTERPOLATOR),
    "constant_lower_only": (_CONSTANT, _INTERPOLATOR),
}


def _assert_matches(expected: Any, call: Callable[[], float], *, where: str) -> None:
    if isinstance(expected, dict):
        assert expected == {"raises": True}, f"unexpected sentinel at {where}"
        with pytest.raises(LibraryException):
            call()
        return
    tolerance.tight(call(), float(expected), reason=where)


def _check_grid(
    surface: LocalVolTermStructure, expected: dict[str, Any], prefix: str
) -> None:
    for t_label, t in _QUERY_TIMES:
        for k_label, k in _QUERY_STRIKES:
            key = f"{t_label}_{k_label}"
            _assert_matches(
                expected[key],
                lambda t=t, k=k: surface.local_vol_at_time(t, k, extrapolate=True),
                where=f"{prefix}/{key}",
            )


def _fixed(
    lower: FixedLocalVolSurface.Extrapolation,
    upper: FixedLocalVolSurface.Extrapolation,
) -> FixedLocalVolSurface:
    return FixedLocalVolSurface(
        reference_date=_REF_DATE,
        times=_TIMES,
        strikes=_PER_SLICE_STRIKES,
        local_vol_matrix=_VOL_MATRIX,
        day_counter=_DC,
        lower_extrapolation=lower,
        upper_extrapolation=upper,
    )


def _policy_ids() -> Sequence[str]:
    return list(_POLICIES)


@pytest.mark.parametrize("policy_key", _policy_ids())
def test_fixed_surface_accessors(policy_key: str) -> None:
    surface = _fixed(*_POLICIES[policy_key])
    expected = _REF["fixed"][policy_key]
    assert surface.max_date().serial_number() == expected["max_date"]
    tolerance.tight(surface.max_time(), expected["max_time"])
    # The LAST slice's range — 90 / 110, not the union 60 / 140.
    tolerance.exact(surface.min_strike(), expected["min_strike"])
    tolerance.exact(surface.max_strike(), expected["max_strike"])


@pytest.mark.parametrize("policy_key", _policy_ids())
def test_fixed_surface_grid(policy_key: str) -> None:
    surface = _fixed(*_POLICIES[policy_key])
    _check_grid(surface, _REF["fixed"][policy_key]["local_vol"], f"fixed/{policy_key}")


def test_extrapolation_policy_is_bypassed_on_a_time_node() -> None:
    """The quirk the reference pins, stated as a property.

    At t = 0.25 (a node) both policies must agree, and both must differ from
    the clamped value; a hair later they must diverge from each other.
    """
    node_constant = _REF["fixed"]["constant_constant"]["local_vol"]["t0p25_node0_k50"]
    node_interp = _REF["fixed"]["interpolator_interpolator"]["local_vol"][
        "t0p25_node0_k50"
    ]
    off_constant = _REF["fixed"]["constant_constant"]["local_vol"][
        "t0p2501_just_after_node0_k50"
    ]
    off_interp = _REF["fixed"]["interpolator_interpolator"]["local_vol"][
        "t0p2501_just_after_node0_k50"
    ]
    assert node_constant == node_interp
    # Clamping to the slice's lowest strike would give the matrix's [0][0].
    assert node_constant != _VOL_MATRIX[0, 0]
    assert off_constant != off_interp


def test_fixed_surface_from_dates() -> None:
    surface = FixedLocalVolSurface.from_dates(
        reference_date=_REF_DATE,
        dates=_DATES,
        strikes=_SHARED_STRIKES,
        local_vol_matrix=_VOL_MATRIX,
        day_counter=_DC,
    )
    expected = _REF["fixed_from_dates"]
    assert surface.max_date().serial_number() == expected["max_date"]
    tolerance.tight(surface.max_time(), expected["max_time"])
    tolerance.exact(surface.min_strike(), expected["min_strike"])
    tolerance.exact(surface.max_strike(), expected["max_strike"])
    _check_grid(surface, expected["local_vol"], "fixed_from_dates")


def _grid_model() -> GridModelLocalVolSurface:
    return GridModelLocalVolSurface(
        reference_date=_REF_DATE,
        dates=_DATES,
        strikes=_PER_SLICE_STRIKES,
        day_counter=_DC,
    )


def test_grid_model_initial_state() -> None:
    grid = _grid_model()
    expected = _REF["grid_model_initial"]
    params = grid.params()
    assert params.size == expected["n_params"]
    for actual, want in zip(params, expected["params"], strict=True):
        tolerance.exact(float(actual), float(want))
    assert grid.max_date().serial_number() == expected["max_date"]
    tolerance.tight(grid.max_time(), expected["max_time"])
    tolerance.exact(grid.min_strike(), expected["min_strike"])
    tolerance.exact(grid.max_strike(), expected["max_strike"])
    _check_grid(grid, expected["local_vol"], "grid_model_initial")


def test_grid_model_after_set_params() -> None:
    """Pins the ROW-major argument layout: a transposed port fails here."""
    grid = _grid_model()
    n = grid.params().size
    grid.set_params(np.asarray([(i + 1) / 10.0 for i in range(n)], dtype=np.float64))
    expected = _REF["grid_model_after_set_params"]
    for actual, want in zip(grid.params(), expected["params"], strict=True):
        tolerance.exact(float(actual), float(want))
    _check_grid(grid, expected["local_vol"], "grid_model_after_set_params")


def test_argument_layout_is_not_transposition_invariant() -> None:
    """Guard the discriminating power of the previous test.

    If the parameter grid happened to be symmetric, a transposed port would
    pass; assert the reference actually separates the two layouts.
    """
    grid = _grid_model()
    n = grid.params().size
    ascending = np.asarray([(i + 1) / 10.0 for i in range(n)], dtype=np.float64)
    grid.set_params(ascending)
    row_major = grid.local_vol_at_time(0.5, 100.0, extrapolate=True)
    n_strikes, n_times = 3, 3
    transposed = ascending.reshape(n_strikes, n_times).T.reshape(-1)
    grid.set_params(transposed)
    assert row_major != grid.local_vol_at_time(0.5, 100.0, extrapolate=True)


def test_parameters_are_not_aliased() -> None:
    """Each grid node must be its own Parameter object.

    C++ ``std::fill`` copies the seed ConstantParameter into every slot; a
    Python port that assigned the same object everywhere would move all nodes
    at once. Setting one parameter must move exactly one.
    """
    grid = _grid_model()
    grid.arguments[0].set_param(0, 5.0)
    params = grid.params()
    tolerance.exact(float(params[0]), 5.0)
    for value in params[1:]:
        tolerance.exact(float(value), 1.0)
