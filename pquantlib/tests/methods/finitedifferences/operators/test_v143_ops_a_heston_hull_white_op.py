"""Cross-validation of the Heston / Hull-White FD operator against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/operators/fdmhestonhullwhiteop.{hpp,cpp}
# @ v1.43 (6b57206e0).

Covers ``FdmHestonHullWhiteEquityPart`` and ``FdmHestonHullWhiteOp`` on a
3-D (log-spot, variance, short-rate-state) grid, plus ``FdmHullWhiteOp``
acting on direction 2 of that same grid. Expected values come from
``migration-harness/cpp/probes/v143_methods_operators/probe.cpp`` (block 4).

``HullWhiteProcess`` is not part of ``pquantlib.processes``; the operator
takes the process structurally (it reads only ``a()`` and ``sigma()``), and
the test supplies the ported ``HullWhiteForwardProcess`` built with the same
two parameters, so both sides see identical numbers.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_heston_hull_white_op import (
    FdmHestonHullWhiteEquityPart,
    FdmHestonHullWhiteOp,
)
from pquantlib.methods.finitedifferences.operators.fdm_hull_white_op import FdmHullWhiteOp
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from tests.methods.finitedifferences.operators.test_v143_ops_a_fixtures import (
    REFERENCE_KEY,
    ToleranceFn,
    all_vectors,
    assert_array,
    assert_surface,
    diag_of,
    heston_process,
    q_ts,
    r_ts,
    vec_const,
    vec_idx,
)

_DT = 0.02
_T1 = 0.15
_T2 = 0.4
_HW_A = 0.05
_HW_SIGMA = 0.012
_CORR = 0.4

# ``apply`` on a vector that is linear in log-spot and constant along the
# other two axes is the one badly-conditioned case in this block. There the
# equity map contributes -0.0300 and the Hull-White map +0.0450, so the
# result (~0.0150) is a 3x cancellation of terms that each carry the frozen
# level phi = 0.5*(r(t1,0) + r(t2,0)) multiplied by log-spot (~4.4).
#
# phi itself is a finite difference: HullWhite's fitting parameter calls
# YieldTermStructure::forwardRate(t, t, ...), which C++ evaluates as
# log(D(t-dt/2)/D(t+dt/2))/dt with dt = 1e-4 — a 1e4 amplification of the
# last bits of discount(). C++ and Python therefore agree on phi only to
# ~4.4e-15 absolute (1.1e-13 relative, still inside TIGHT — asserted
# directly in test_frozen_short_rate_level below); times log-spot 4.4 and
# divided by the 0.0150 result that becomes ~1e-12 relative here.
# 1e-13 absolute is ~3 ulps of the largest cancelling band product (3.06).
_CANCELLATION_TOL: ToleranceFn = partial(
    custom,
    abs_tol=1e-13,
    rel_tol=1e-11,
    reason=(
        "phi = 0.5*(r(t1,0)+r(t2,0)) is a 1e-4-step finite difference of "
        "discount factors, so C++/Python agree on it only to ~4.4e-15 "
        "absolute; multiplied by log-spot (~4.4) and divided by the "
        "-0.0300 + 0.0450 = 0.0150 cancellation, that is ~1e-12 relative "
        "in this one apply()"
    ),
)


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load(REFERENCE_KEY)


def _mesher() -> FdmMesherComposite:
    """# C++ parity: probe.cpp ``block_heston_hull_white``."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(80.0), math.log(130.0), 5),
        Uniform1dMesher(0.01, 0.3, 4),
        Uniform1dMesher(-0.05, 0.07, 4),
    )


def _hw_model() -> HullWhite:
    return HullWhite(r_ts(), _HW_A, _HW_SIGMA)


def _build() -> tuple[FdmHestonHullWhiteOp, FdmMesherComposite]:
    mesher = _mesher()
    op = FdmHestonHullWhiteOp(
        mesher,
        heston_process(r_ts(), q_ts()),
        HullWhiteForwardProcess(r_ts(), _HW_A, _HW_SIGMA),
        _CORR,
    )
    op.set_time(_T1, _T2)
    return op, mesher


def test_mesher_locations(ref: dict[str, Any]) -> None:
    mesher = _mesher()
    for d in range(3):
        assert_array(mesher.locations(d), ref, f"hhw_locations{d}")


def test_frozen_short_rate_level(ref: dict[str, Any]) -> None:
    dynamics = _hw_model().dynamics()
    tight(dynamics.short_rate(_T1, 0.0), float(ref["hhw_short_rate_t1"]))
    tight(dynamics.short_rate(_T2, 0.0), float(ref["hhw_short_rate_t2"]))


def test_equity_part(ref: dict[str, Any]) -> None:
    mesher = _mesher()
    part = FdmHestonHullWhiteEquityPart(mesher, _hw_model(), q_ts())
    part.set_time(_T1, _T2)
    assert_array(part.get_map().apply(vec_const(mesher)), ref, "hhw_equity_part_apply_const")
    assert_array(part.get_map().apply(vec_idx(mesher)), ref, "hhw_equity_part_apply_idx")
    assert_array(diag_of(part.get_map().to_matrix()), ref, "hhw_equity_part_diag")


def test_hull_white_op_on_direction_2(ref: dict[str, Any]) -> None:
    """``FdmHullWhiteOp`` driving direction 2 of a 3-D grid."""
    mesher = _mesher()
    op = FdmHullWhiteOp(mesher, _hw_model(), 2)
    op.set_time(_T1, _T2)
    idx = vec_idx(mesher)
    assert_array(op.apply(idx), ref, "hhw_hwop_dir2_apply_idx")
    assert_array(op.solve_splitting(2, idx, _DT), ref, "hhw_hwop_dir2_solve_idx")
    assert_array(op.apply_direction(0, idx), ref, "hhw_hwop_dir2_apply_dir0")


def test_size(ref: dict[str, Any]) -> None:
    op, _ = _build()
    assert op.size() == int(ref["hhw_size"])


@pytest.mark.parametrize("tag", ["const", "lin0", "lin1", "lin2", "mix", "idx"])
def test_operator_surface(ref: dict[str, Any], tag: str) -> None:
    op, mesher = _build()
    vectors = dict(all_vectors(mesher, with_direction2=True))
    compare: ToleranceFn = _CANCELLATION_TOL if tag == "lin0" else tight
    assert_surface(ref, "hhw", op, vectors[tag], tag, 3, _DT, compare=compare)


def test_to_matrix_decomp(ref: dict[str, Any]) -> None:
    op, _ = _build()
    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(ref["hhw_decomp_size"])
    for i in range(4):
        assert_array(diag_of(decomp[i]), ref, f"hhw_decomp{i}_diag")


def test_direction_out_of_range_fails() -> None:
    """# C++ parity: ``QL_FAIL("direction too large")``."""
    op, mesher = _build()
    v = vec_const(mesher)
    with pytest.raises(LibraryException, match="direction too large"):
        op.apply_direction(3, v)
    with pytest.raises(LibraryException, match="direction too large"):
        op.solve_splitting(3, v, _DT)


def test_negative_eigenvalue_guard() -> None:
    """# C++ parity: ``QL_REQUIRE(corr^2 + rho^2 <= 1.0, ...)`` (rho = -0.6)."""
    mesher = _mesher()
    with pytest.raises(LibraryException, match="negative eigenvalues"):
        FdmHestonHullWhiteOp(
            mesher,
            heston_process(r_ts(), q_ts()),
            HullWhiteForwardProcess(r_ts(), _HW_A, _HW_SIGMA),
            0.9,
        )
