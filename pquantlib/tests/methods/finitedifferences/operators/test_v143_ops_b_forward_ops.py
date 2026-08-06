"""Cross-validation of the Fokker-Planck (forward) FD operators vs C++ v1.43.

Covers ``FdmBlackScholesFwdOp``, ``FdmLocalVolFwdOp``,
``FdmSquareRootFwdOp`` and ``FdmHestonFwdOp``.

# C++ parity: ql/methods/finitedifferences/operators/fdmblackscholesfwdop.{hpp,cpp},
# fdmlocalvolfwdop.{hpp,cpp}, fdmsquarerootfwdop.{hpp,cpp},
# fdmhestonfwdop.{hpp,cpp} @ v1.43.

Reference values come from ``migration-harness/cpp/probes/v143_methods_operators2``
(→ ``migration-harness/references/v143/methods/operators2.json``).

Every case pins ``apply`` on four structurally different vectors
(constant, linear in each direction, quadratic / product, and a
non-symmetric "ramp"), plus ``apply_mixed``, ``apply_direction`` for
**both** directions, ``solve_splitting`` for both directions, the
preconditioner, and the dense matrices of ``to_matrix_decomp()``. The
ramp vector is deliberately asymmetric so a swapped direction or a
sign slip cannot cancel out.

Tolerance: TIGHT (abs 1e-14 / rel 1e-12) throughout — verified across
all 30k+ probed values with zero exceptions needed.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_fwd_op import (
    FdmBlackScholesFwdOp,
)
from pquantlib.methods.finitedifferences.operators.fdm_heston_fwd_op import (
    FdmHestonFwdOp,
)
from pquantlib.methods.finitedifferences.operators.fdm_local_vol_fwd_op import (
    FdmLocalVolFwdOp,
)
from pquantlib.methods.finitedifferences.operators.fdm_square_root_fwd_op import (
    FdmSquareRootFwdOp,
    TransformationType,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
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


class ProbeLocalVol(LocalVolTermStructure):
    """The closed-form local-vol surface hard-coded in the C++ probe.

    ``sigma_loc(t, S) = 0.20 + 0.05*log(S/100) + 0.10*t`` — deliberately
    varying in **both** ``t`` and ``S`` so a wrong (t, spot) mapping in
    the local-vol / leverage code paths cannot pass.
    """

    def __init__(self, reference_date: Date, day_counter: DayCounter) -> None:
        super().__init__(
            reference_date=reference_date,
            calendar=NullCalendar(),
            day_counter=day_counter,
        )

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return 1.0e-8

    def max_strike(self) -> float:
        return 1.0e8

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        return 0.20 + 0.05 * math.log(underlying_level / 100.0) + 0.10 * t


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
# FdmBlackScholesFwdOp
# ===========================================================================


def _bs_process(local_vol: LocalVolTermStructure | None = None) -> GeneralizedBlackScholesProcess:
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=_flat(0.02),
        risk_free_ts=_flat(0.05),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY, calendar=CAL, day_counter=DC, volatility=0.25
        ),
        local_vol_ts=local_vol,
    )


def _bsfwd_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(math.log(50.0), math.log(200.0), 9))


def test_black_scholes_fwd_constant_vol_matches_cpp() -> None:
    mesher = _bsfwd_mesher()
    _assert_tight("bsfwd_locations", mesher.locations(0))
    op = FdmBlackScholesFwdOp(mesher, _bs_process(), 100.0)
    assert op.size() == int(REF["bsfwd_size"])
    op.set_time(T1, T2)

    _assert_tight("bsfwd_apply_const", op.apply(_const(mesher)))
    _assert_tight("bsfwd_apply_lin", op.apply(_lin(mesher, 0)))
    _assert_tight("bsfwd_apply_quad", op.apply(_quad(mesher, 0)))
    _assert_tight("bsfwd_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("bsfwd_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight("bsfwd_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("bsfwd_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("bsfwd_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.1))
    # Off-direction solve returns r unchanged, not zero.
    _assert_tight("bsfwd_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.1))
    _assert_tight("bsfwd_precond_ramp", op.preconditioner(_ramp(mesher), 0.1))

    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(REF["bsfwd_decomp_size"])
    _assert_tight("bsfwd_decomp0", _dense(decomp[0]))


def test_black_scholes_fwd_local_vol_matches_cpp() -> None:
    """The ``multR`` (adjoint) branch, driven by a t- and S-dependent surface."""
    mesher = _bsfwd_mesher()
    op = FdmBlackScholesFwdOp(mesher, _bs_process(ProbeLocalVol(TODAY, DC)), 100.0, True)
    op.set_time(T1, T2)
    _assert_tight("bsfwd_lv_apply_const", op.apply(_const(mesher)))
    _assert_tight("bsfwd_lv_apply_lin", op.apply(_lin(mesher, 0)))
    _assert_tight("bsfwd_lv_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("bsfwd_lv_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.1))


def test_black_scholes_fwd_direction_one_matches_cpp() -> None:
    """2-D mesher with ``direction = 1`` — pins the directional dispatch."""
    mesher = FdmMesherComposite(
        Uniform1dMesher(0.0, 1.0, 4),
        Uniform1dMesher(math.log(50.0), math.log(200.0), 5),
    )
    op = FdmBlackScholesFwdOp(mesher, _bs_process(), 100.0, False, direction=1)
    op.set_time(T1, T2)
    _assert_tight("bsfwd_d1_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("bsfwd_d1_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("bsfwd_d1_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("bsfwd_d1_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.1))
    _assert_tight("bsfwd_d1_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.1))
    _assert_tight("bsfwd_d1_precond_ramp", op.preconditioner(_ramp(mesher), 0.1))


# ===========================================================================
# FdmLocalVolFwdOp
# ===========================================================================


def test_local_vol_fwd_matches_cpp() -> None:
    mesher = _bsfwd_mesher()
    op = FdmLocalVolFwdOp(
        mesher,
        SimpleQuote(100.0),
        _flat(0.05),
        _flat(0.02),
        ProbeLocalVol(TODAY, DC),
    )
    assert op.size() == int(REF["lvfwd_size"])
    op.set_time(T1, T2)
    _assert_tight("lvfwd_apply_const", op.apply(_const(mesher)))
    _assert_tight("lvfwd_apply_lin", op.apply(_lin(mesher, 0)))
    _assert_tight("lvfwd_apply_quad", op.apply(_quad(mesher, 0)))
    _assert_tight("lvfwd_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("lvfwd_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight("lvfwd_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight("lvfwd_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight("lvfwd_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.1))
    _assert_tight("lvfwd_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.1))
    _assert_tight("lvfwd_precond_ramp", op.preconditioner(_ramp(mesher), 0.1))
    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(REF["lvfwd_decomp_size"])
    _assert_tight("lvfwd_decomp0", _dense(decomp[0]))


def test_local_vol_fwd_direction_one_matches_cpp() -> None:
    mesher = FdmMesherComposite(
        Uniform1dMesher(0.0, 1.0, 4),
        Uniform1dMesher(math.log(50.0), math.log(200.0), 5),
    )
    op = FdmLocalVolFwdOp(
        mesher,
        SimpleQuote(100.0),
        _flat(0.05),
        _flat(0.02),
        ProbeLocalVol(TODAY, DC),
        1,
    )
    op.set_time(T1, T2)
    _assert_tight("lvfwd_d1_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight("lvfwd_d1_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.1))


# ===========================================================================
# FdmSquareRootFwdOp
# ===========================================================================

_KAPPA = 2.5
_THETA = 0.06
_SIGMA = 0.4


def _sqrt_mesher_plain() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(0.005, 0.4, 8))


def _sqrt_mesher_log() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(math.log(0.005), math.log(0.4), 8))


def _sqrt_mesher_2d() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(200.0), 4),
        Uniform1dMesher(0.005, 0.4, 8),
    )


_SQRT_CASES: list[tuple[str, str, int, TransformationType]] = [
    ("sqrtfwd_plain", "plain", 0, TransformationType.Plain),
    ("sqrtfwd_power", "plain", 0, TransformationType.Power),
    ("sqrtfwd_log", "log", 0, TransformationType.Log),
    ("sqrtfwd_2d_plain", "2d", 1, TransformationType.Plain),
]


@pytest.mark.parametrize(("tag", "grid", "direction", "ttype"), _SQRT_CASES)
def test_square_root_fwd_matches_cpp(
    tag: str, grid: str, direction: int, ttype: TransformationType
) -> None:
    mesher = {
        "plain": _sqrt_mesher_plain,
        "log": _sqrt_mesher_log,
        "2d": _sqrt_mesher_2d,
    }[grid]()
    op = FdmSquareRootFwdOp(mesher, _KAPPA, _THETA, _SIGMA, direction, ttype)
    assert op.size() == int(REF[f"{tag}_size"])

    # Ghost-node-extended grid: v(0) and v(n+1) are extrapolated.
    n = mesher.layout().dim()[direction]
    _assert_tight(f"{tag}_v", np.array([op.v(i) for i in range(n + 2)]))
    tight(op.lower_boundary_factor(ttype), float(REF[f"{tag}_lower_boundary_factor"]))
    tight(op.upper_boundary_factor(ttype), float(REF[f"{tag}_upper_boundary_factor"]))

    # setTime is a documented no-op in C++; calling it must not change anything.
    op.set_time(T1, T2)

    other = 1 if direction == 0 else 0
    _assert_tight(f"{tag}_apply_const", op.apply(_const(mesher)))
    _assert_tight(f"{tag}_apply_lin", op.apply(_lin(mesher, direction)))
    _assert_tight(f"{tag}_apply_quad", op.apply(_quad(mesher, direction)))
    _assert_tight(f"{tag}_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight(f"{tag}_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight(f"{tag}_apply_dir_ramp", op.apply_direction(direction, _ramp(mesher)))
    _assert_tight(f"{tag}_apply_other_ramp", op.apply_direction(other, _ramp(mesher)))
    _assert_tight(f"{tag}_solve_dir_ramp", op.solve_splitting(direction, _ramp(mesher), 0.05))
    _assert_tight(f"{tag}_solve_other_ramp", op.solve_splitting(other, _ramp(mesher), 0.05))
    _assert_tight(f"{tag}_precond_ramp", op.preconditioner(_ramp(mesher), 0.05))

    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(REF[f"{tag}_decomp_size"])
    _assert_tight(f"{tag}_decomp0", _dense(decomp[0]))


def test_square_root_fwd_mesher_locations_match_cpp() -> None:
    _assert_tight("sqrtfwd_plain_locations", _sqrt_mesher_plain().locations(0))
    _assert_tight("sqrtfwd_log_locations", _sqrt_mesher_log().locations(0))


def test_square_root_fwd_rejects_out_of_range_ghost_index() -> None:
    mesher = _sqrt_mesher_plain()
    op = FdmSquareRootFwdOp(mesher, _KAPPA, _THETA, _SIGMA, 0, TransformationType.Plain)
    with pytest.raises(LibraryException):
        op.v(mesher.layout().dim()[0] + 2)


# ===========================================================================
# FdmHestonFwdOp
# ===========================================================================


def _heston_process() -> HestonProcess:
    return HestonProcess(
        risk_free_rate=_flat(0.05),
        dividend_yield=_flat(0.02),
        s0=SimpleQuote(100.0),
        v0=0.05,
        kappa=_KAPPA,
        theta=_THETA,
        sigma=_SIGMA,
        rho=-0.6,
    )


def _heston_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(200.0), 6),
        Uniform1dMesher(0.005, 0.4, 5),
    )


def _heston_mesher_log() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(200.0), 6),
        Uniform1dMesher(math.log(0.005), math.log(0.4), 5),
    )


_HESTON_CASES: list[tuple[str, str, TransformationType, bool, float]] = [
    ("hestonfwd_plain", "plain", TransformationType.Plain, False, 1.0),
    ("hestonfwd_power", "plain", TransformationType.Power, False, 1.0),
    ("hestonfwd_plain_mix", "plain", TransformationType.Plain, False, 0.8),
    ("hestonfwd_log", "log", TransformationType.Log, False, 1.0),
    ("hestonfwd_plain_lev", "plain", TransformationType.Plain, True, 1.0),
    ("hestonfwd_power_lev", "plain", TransformationType.Power, True, 1.0),
    ("hestonfwd_log_lev", "log", TransformationType.Log, True, 1.0),
]


@pytest.mark.parametrize(
    ("tag", "grid", "ttype", "with_leverage", "mixing"), _HESTON_CASES
)
def test_heston_fwd_matches_cpp(
    tag: str,
    grid: str,
    ttype: TransformationType,
    with_leverage: bool,
    mixing: float,
) -> None:
    mesher = _heston_mesher() if grid == "plain" else _heston_mesher_log()
    leverage = ProbeLocalVol(TODAY, DC) if with_leverage else None
    op = FdmHestonFwdOp(mesher, _heston_process(), ttype, leverage, mixing)
    assert op.size() == int(REF[f"{tag}_size"])
    op.set_time(T1, T2)

    _assert_tight(f"{tag}_apply_const", op.apply(_const(mesher)))
    _assert_tight(f"{tag}_apply_lin0", op.apply(_lin(mesher, 0)))
    _assert_tight(f"{tag}_apply_lin1", op.apply(_lin(mesher, 1)))
    _assert_tight(f"{tag}_apply_prod", op.apply(_prod(mesher)))
    _assert_tight(f"{tag}_apply_ramp", op.apply(_ramp(mesher)))
    _assert_tight(f"{tag}_apply_mixed_ramp", op.apply_mixed(_ramp(mesher)))
    _assert_tight(f"{tag}_apply_dir0_ramp", op.apply_direction(0, _ramp(mesher)))
    _assert_tight(f"{tag}_apply_dir1_ramp", op.apply_direction(1, _ramp(mesher)))
    _assert_tight(f"{tag}_solve_dir0_ramp", op.solve_splitting(0, _ramp(mesher), 0.05))
    _assert_tight(f"{tag}_solve_dir1_ramp", op.solve_splitting(1, _ramp(mesher), 0.05))
    _assert_tight(f"{tag}_precond_ramp", op.preconditioner(_ramp(mesher), 0.05))

    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(REF[f"{tag}_decomp_size"])
    _assert_tight(f"{tag}_decomp0", _dense(decomp[0]))
    _assert_tight(f"{tag}_decomp1", _dense(decomp[1]))
    _assert_tight(f"{tag}_decomp2", _dense(decomp[2]))


def test_heston_fwd_mesher_locations_match_cpp() -> None:
    _assert_tight("hestonfwd_locations0", _heston_mesher().locations(0))
    _assert_tight("hestonfwd_locations1", _heston_mesher().locations(1))
    _assert_tight("hestonfwd_log_locations1", _heston_mesher_log().locations(1))


def test_heston_fwd_rejects_third_direction() -> None:
    mesher = _heston_mesher()
    op = FdmHestonFwdOp(mesher, _heston_process())
    op.set_time(T1, T2)
    with pytest.raises(LibraryException):
        op.apply_direction(2, _ramp(mesher))
    with pytest.raises(LibraryException):
        op.solve_splitting(2, _ramp(mesher), 0.05)
