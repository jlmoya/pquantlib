"""Structural-only tests for the W5-B FD engine scaffolds.

# C++ parity reference:
# ql/experimental/finitedifferences/fdsimpleextoustorageengine.hpp
# ql/experimental/finitedifferences/fdsimpleklugeextouvppengine.hpp
# ql/experimental/finitedifferences/fdsimpleextoujumpswingengine.hpp
# (v1.42.1).

These engines depend on Extended-Ornstein-Uhlenbeck / Kluge process FD
operators that don't yet exist in pquantlib (deferred to W5-A or
later). At W5-B we cover only:

* Constructor wires the C++ signature 1:1 (process + curve + grid
  dimensions + shape arrays + scheme descriptor).
* ``calculate()`` raises :class:`NotImplementedError` with a message
  pointing at the W5-A carve-out.
* The engines correctly use the matching argument types via
  :class:`GenericEngine`.

Once W5-A lands and provides the FD operator + solver, ``calculate()``
will be wired and the structural tests here will be superseded by NPV
cross-validation tests against the C++ probe.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_actual import ActualActual, Convention
from pquantlib.experimental.finitedifferences.fd_simple_ext_ou_jump_swing_engine import (
    FdSimpleExtOUJumpSwingEngine,
)
from pquantlib.experimental.finitedifferences.fd_simple_ext_ou_storage_engine import (
    FdSimpleExtOUStorageEngine,
)
from pquantlib.experimental.finitedifferences.fd_simple_kluge_ext_ou_vpp_engine import (
    FdSimpleKlugeExtOUVPPEngine,
)
from pquantlib.experimental.finitedifferences.vanilla_storage_option import (
    VanillaStorageOptionArguments,
)
from pquantlib.experimental.finitedifferences.vanilla_swing_option import (
    VanillaSwingOptionArguments,
)
from pquantlib.experimental.finitedifferences.vanilla_vpp_option import (
    VanillaVPPOptionArguments,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def today() -> Date:
    return Date.from_ymd(18, Month.December, 2011)


@pytest.fixture(scope="module")
def r_ts(today: Date) -> FlatForward:
    return FlatForward.from_rate(today, 0.05, ActualActual(Convention.ISDA))


class _MockExtOU:
    """Minimal ExtendedOrnsteinUhlenbeck process stub for engine wiring."""

    def x0(self) -> float:
        return 3.0


class _MockExtOUWithJumps:
    """Minimal ExtOUWithJumps process stub."""

    def factors(self) -> int:
        return 2


class _MockKluge:
    """Minimal KlugeExtOU process stub."""

    def factors(self) -> int:
        return 3


# ---------- FdSimpleExtOUStorageEngine --------------------------------------


def test_storage_engine_constructor_wires_signature(r_ts: FlatForward) -> None:
    engine = FdSimpleExtOUStorageEngine(
        process=_MockExtOU(),
        r_ts=r_ts,
        t_grid=1,
        x_grid=25,
    )
    args = engine.get_arguments()
    assert isinstance(args, VanillaStorageOptionArguments)
    results = engine.get_results()
    assert results.value is None


def test_storage_engine_calculate_defers_to_w5a(r_ts: FlatForward) -> None:
    engine = FdSimpleExtOUStorageEngine(
        process=_MockExtOU(),
        r_ts=r_ts,
    )
    with pytest.raises(NotImplementedError):
        engine.calculate()


def test_storage_engine_accepts_optional_shape_and_y_grid(
    r_ts: FlatForward,
) -> None:
    shape: list[tuple[float, float]] = [(0.5, 21.0), (1.0, 21.5)]
    engine = FdSimpleExtOUStorageEngine(
        process=_MockExtOU(),
        r_ts=r_ts,
        t_grid=50,
        x_grid=100,
        y_grid=20,
        shape=shape,
        scheme_desc=None,
    )
    # No exception ⇒ constructor signature parity OK.
    assert engine.get_arguments() is not None


# ---------- FdSimpleKlugeExtOUVPPEngine ------------------------------------


def test_kluge_vpp_engine_constructor_wires_signature(r_ts: FlatForward) -> None:
    fuel_shape: list[tuple[float, float]] = [(0.5, 21.0), (1.0, 21.5)]
    power_shape: list[tuple[float, float]] = [(0.5, 35.0), (1.0, 36.0)]
    engine = FdSimpleKlugeExtOUVPPEngine(
        process=_MockKluge(),
        r_ts=r_ts,
        fuel_shape=fuel_shape,
        power_shape=power_shape,
        fuel_cost_addon=3.0,
        t_grid=1,
        x_grid=50,
        y_grid=10,
        g_grid=20,
    )
    args = engine.get_arguments()
    assert isinstance(args, VanillaVPPOptionArguments)


def test_kluge_vpp_engine_calculate_defers_to_w5a(r_ts: FlatForward) -> None:
    engine = FdSimpleKlugeExtOUVPPEngine(
        process=_MockKluge(),
        r_ts=r_ts,
        fuel_shape=None,
        power_shape=None,
        fuel_cost_addon=3.0,
    )
    with pytest.raises(NotImplementedError):
        engine.calculate()


# ---------- FdSimpleExtOUJumpSwingEngine -----------------------------------


def test_swing_engine_constructor_wires_signature(r_ts: FlatForward) -> None:
    engine = FdSimpleExtOUJumpSwingEngine(
        process=_MockExtOUWithJumps(),
        r_ts=r_ts,
        t_grid=25,
        x_grid=50,
        y_grid=25,
    )
    args = engine.get_arguments()
    assert isinstance(args, VanillaSwingOptionArguments)


def test_swing_engine_calculate_defers_to_w5a(r_ts: FlatForward) -> None:
    engine = FdSimpleExtOUJumpSwingEngine(
        process=_MockExtOUWithJumps(),
        r_ts=r_ts,
    )
    with pytest.raises(NotImplementedError):
        engine.calculate()


def test_swing_engine_accepts_optional_shape(r_ts: FlatForward) -> None:
    shape: list[tuple[float, float]] = [(0.0, 30.0), (1.0, 32.0)]
    engine = FdSimpleExtOUJumpSwingEngine(
        process=_MockExtOUWithJumps(),
        r_ts=r_ts,
        shape=shape,
    )
    assert engine.get_arguments() is not None
