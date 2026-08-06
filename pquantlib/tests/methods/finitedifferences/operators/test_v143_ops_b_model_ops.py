"""Cross-validation of FdmWienerOp / Fdm2dBlackScholesOp / FdmSabrOp / FdmCEVOp.

# C++ parity: ql/methods/finitedifferences/operators/fdmwienerop.{hpp,cpp},
# fdm2dblackscholesop.{hpp,cpp}, fdmsabrop.{hpp,cpp}, fdmcevop.{hpp,cpp}
# @ v1.43.

Reference values come from ``migration-harness/cpp/probes/v143_methods_operators2``
(→ ``migration-harness/references/v143/methods/operators2.json``).

Tolerance: TIGHT (abs 1e-14 / rel 1e-12) throughout.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_2d_black_scholes_op import (
    Fdm2dBlackScholesOp,
)
from pquantlib.methods.finitedifferences.operators.fdm_cev_op import FdmCEVOp
from pquantlib.methods.finitedifferences.operators.fdm_sabr_op import FdmSabrOp
from pquantlib.methods.finitedifferences.operators.fdm_wiener_op import FdmWienerOp
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

REF: dict[str, Any] = reference_reader.load("v143/methods/operators2")

DC = Actual365Fixed()
CAL = NullCalendar()
TODAY = Date.from_ymd(15, Month.January, 2024)
T1 = 0.1
T2 = 0.35


def _ref_list(key: str) -> list[float]:
    value: Any = REF[key]
    assert isinstance(value, list)
    return [float(v) for v in cast("list[Any]", value)]


def _assert_tight(key: str, got: Array) -> None:
    expected = _ref_list(key)
    got_list = [float(v) for v in np.asarray(got).ravel()]
    assert len(got_list) == len(expected), f"{key}: {len(got_list)} != {len(expected)}"
    for i, (g, e) in enumerate(zip(got_list, expected, strict=True)):
        tight(g, e, reason=f"{key}[{i}]")


def _dense(matrix: Any) -> Array:
    return np.asarray(matrix.todense(), dtype=np.float64).ravel()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def _const(mesher: FdmMesher, c: float = 1.0) -> Array:
    return np.full(mesher.layout().size(), c, dtype=np.float64)


def _lin(mesher: FdmMesher, direction: int) -> Array:
    return mesher.locations(direction).copy()


def _quad(mesher: FdmMesher, direction: int) -> Array:
    x = mesher.locations(direction)
    return x * x


def _prod(mesher: FdmMesher) -> Array:
    return mesher.locations(0) * mesher.locations(1)


def _ramp(mesher: FdmMesher) -> Array:
    n = mesher.layout().size()
    return np.array([1.0 + 0.25 * i - 0.03 * i * i for i in range(n)], dtype=np.float64)


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(
        reference_date=TODAY, forward_rate=rate, day_counter=DC
    )


# ===========================================================================
# FdmWienerOp
# ===========================================================================


def _wiener_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(-1.0, 1.0, 5), Uniform1dMesher(-0.5, 1.5, 4)
    )


def test_wiener_op_matches_cpp() -> None:
    mesher = _wiener_mesher()
    op = FdmWienerOp(mesher, _flat(0.04), np.array([0.3, 0.5], dtype=np.float64))
    assert op.size() == int(REF["wiener_size"])
    op.set_time(T1, T2)

    _assert_tight("wiener_apply_const", op.apply(_const(mesher)))
    _assert_tight("wiener_apply_lin0", op.apply(_lin(mesher, 0)))
    _assert_tight("wiener_apply_lin1", op.apply(_lin(mesher, 1)))
    _assert_tight("wiener_apply_quad0", op.apply(_quad(mesher, 0)))
    _assert_tight("wiener_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("wiener_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight("wiener_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("wiener_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("wiener_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.1))
    _assert_tight("wiener_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.1))
    _assert_tight("wiener_precond_ramp", op.preconditioner(_ramp(mesher), 0.1))

    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(REF["wiener_decomp_size"])
    _assert_tight("wiener_decomp0", _dense(decomp[0]))
    _assert_tight("wiener_decomp1", _dense(decomp[1]))


def test_wiener_op_without_yield_curve_matches_cpp() -> None:
    """``rTS`` may be null; ``setTime`` then leaves ``r`` at 0."""
    mesher = FdmMesherComposite(Uniform1dMesher(-2.0, 2.0, 6))
    op = FdmWienerOp(mesher, None, np.array([0.7], dtype=np.float64))
    assert op.size() == int(REF["wiener_null_size"])
    op.set_time(T1, T2)
    _assert_tight("wiener_null_apply_const", op.apply(_const(mesher)))
    _assert_tight("wiener_null_apply_quad", op.apply(_quad(mesher, 0)))
    _assert_tight("wiener_null_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("wiener_null_solve_dir0", op.solve_splitting(0, _ramp(mesher), 0.1))


def test_wiener_op_rejects_dimension_mismatch() -> None:
    mesher = _wiener_mesher()
    with pytest.raises(LibraryException):
        FdmWienerOp(mesher, _flat(0.04), np.array([0.3], dtype=np.float64))


# ===========================================================================
# Fdm2dBlackScholesOp
# ===========================================================================


def _bs2d_processes() -> tuple[
    GeneralizedBlackScholesProcess, GeneralizedBlackScholesProcess
]:
    p1 = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=_flat(0.02),
        risk_free_ts=_flat(0.05),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY, calendar=CAL, day_counter=DC, volatility=0.25
        ),
    )
    p2 = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(90.0),
        dividend_ts=_flat(0.01),
        risk_free_ts=_flat(0.05),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY, calendar=CAL, day_counter=DC, volatility=0.30
        ),
    )
    return p1, p2


def _bs2d_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(200.0), 5),
        Uniform1dMesher(math.log(45.0), math.log(180.0), 4),
    )


def test_fdm_2d_black_scholes_op_matches_cpp() -> None:
    mesher = _bs2d_mesher()
    p1, p2 = _bs2d_processes()
    op = Fdm2dBlackScholesOp(mesher, p1, p2, 0.4, 1.0)
    assert op.size() == int(REF["bs2d_size"])
    op.set_time(T1, T2)

    _assert_tight("bs2d_apply_const", op.apply(_const(mesher)))
    _assert_tight("bs2d_apply_lin0", op.apply(_lin(mesher, 0)))
    _assert_tight("bs2d_apply_lin1", op.apply(_lin(mesher, 1)))
    _assert_tight("bs2d_apply_prod", op.apply(_prod(mesher)))
    _assert_tight("bs2d_apply_ramp", op.apply(_ramp(mesher)))
    # apply_mixed carries the correlation stencil *plus* +r*x.
    _assert_tight("bs2d_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight("bs2d_apply_mixed_prod", op.apply_mixed(_prod(mesher)))
    _assert_tight("bs2d_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("bs2d_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("bs2d_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.1))
    _assert_tight("bs2d_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.1))
    _assert_tight("bs2d_precond_ramp", op.preconditioner(_ramp(mesher), 0.1))


def test_fdm_2d_black_scholes_op_rejects_third_direction() -> None:
    mesher = _bs2d_mesher()
    p1, p2 = _bs2d_processes()
    op = Fdm2dBlackScholesOp(mesher, p1, p2, 0.4, 1.0)
    op.set_time(T1, T2)
    with pytest.raises(LibraryException):
        op.apply_direction(2, _ramp(mesher))
    with pytest.raises(LibraryException):
        op.solve_splitting(2, _ramp(mesher), 0.1)


def test_fdm_2d_black_scholes_op_local_vol_is_carved_out() -> None:
    """``local_vol=True`` is blocked, not silently degraded to constant vol.

    See the class docstring: it needs the local-vol branch of the
    pre-existing ``FdmBlackScholesOp``, which is a documented carve-out.
    """
    mesher = _bs2d_mesher()
    p1, p2 = _bs2d_processes()
    with pytest.raises(LibraryException, match="local_vol"):
        Fdm2dBlackScholesOp(mesher, p1, p2, 0.4, 1.0, True)


# ===========================================================================
# FdmSabrOp
# ===========================================================================


def _sabr_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(0.01, 0.2, 6),
        Uniform1dMesher(math.log(0.1), math.log(0.6), 5),
    )


def test_sabr_op_matches_cpp() -> None:
    mesher = _sabr_mesher()
    _assert_tight("sabr_locations0", mesher.locations(0))
    _assert_tight("sabr_locations1", mesher.locations(1))

    op = FdmSabrOp(mesher, _flat(0.03), 0.05, 0.25, 0.6, 0.4, -0.3)
    assert op.size() == int(REF["sabr_size"])
    op.set_time(T1, T2)

    _assert_tight("sabr_apply_const", op.apply(_const(mesher)))
    _assert_tight("sabr_apply_lin0", op.apply(_lin(mesher, 0)))
    _assert_tight("sabr_apply_lin1", op.apply(_lin(mesher, 1)))
    _assert_tight("sabr_apply_prod", op.apply(_prod(mesher)))
    _assert_tight("sabr_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("sabr_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight("sabr_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("sabr_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("sabr_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.05))
    _assert_tight("sabr_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.05))
    # preconditioner chains both splitting solves.
    _assert_tight("sabr_precond_ramp", op.preconditioner(_ramp(mesher), 0.05))

    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(REF["sabr_decomp_size"])
    # Order is [mapA_, mapF_, correlationMap_] — alpha direction first.
    _assert_tight("sabr_decomp0", _dense(decomp[0]))
    _assert_tight("sabr_decomp1", _dense(decomp[1]))
    _assert_tight("sabr_decomp2", _dense(decomp[2]))


def test_sabr_op_rejects_third_direction() -> None:
    mesher = _sabr_mesher()
    op = FdmSabrOp(mesher, _flat(0.03), 0.05, 0.25, 0.6, 0.4, -0.3)
    op.set_time(T1, T2)
    with pytest.raises(LibraryException):
        op.apply_direction(2, _ramp(mesher))
    with pytest.raises(LibraryException):
        op.solve_splitting(2, _ramp(mesher), 0.05)


# ===========================================================================
# FdmCEVOp
# ===========================================================================


def _cev_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(0.01, 0.2, 7))


def test_cev_op_matches_cpp() -> None:
    mesher = _cev_mesher()
    _assert_tight("cev_locations", mesher.locations(0))

    op = FdmCEVOp(mesher, _flat(0.03), 0.05, 0.3, 0.6, 0)
    assert op.size() == int(REF["cev_size"])
    op.set_time(T1, T2)

    _assert_tight("cev_apply_const", op.apply(_const(mesher)))
    _assert_tight("cev_apply_lin", op.apply(_lin(mesher, 0)))
    _assert_tight("cev_apply_quad", op.apply(_quad(mesher, 0)))
    _assert_tight("cev_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("cev_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight("cev_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("cev_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("cev_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.05))
    # NOTE: unlike the other forward ops, CEV's off-direction solve
    # returns *zero*, not r (fdmcevop.cpp:69-76).
    _assert_tight("cev_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.05))
    _assert_tight("cev_precond_ramp", op.preconditioner(_ramp(mesher), 0.05))

    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(REF["cev_decomp_size"])
    _assert_tight("cev_decomp0", _dense(decomp[0]))


def test_cev_op_direction_one_matches_cpp() -> None:
    """2-D with ``direction = 1``.

    This pins the C++ quirk that ``dxxMap_`` is built from
    ``SecondDerivativeOp(0, mesher)`` — direction 0 — while the
    coefficient array and the splitting operator use ``direction``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(0.0, 1.0, 4), Uniform1dMesher(0.01, 0.2, 7)
    )
    _assert_tight("cev_2d_locations0", mesher.locations(0))
    _assert_tight("cev_2d_locations1", mesher.locations(1))

    op = FdmCEVOp(mesher, _flat(0.03), 0.05, 0.3, 0.6, 1)
    op.set_time(T1, T2)
    _assert_tight("cev_2d_apply_const", op.apply(_const(mesher)))
    _assert_tight("cev_2d_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("cev_2d_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("cev_2d_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("cev_2d_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.05))
    _assert_tight("cev_2d_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.05))
    _assert_tight("cev_2d_precond_ramp", op.preconditioner(_ramp(mesher), 0.05))
    _assert_tight("cev_2d_decomp0", _dense(op.to_matrix_decomp()[0]))
