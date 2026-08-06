"""Cross-validation of ``FdmG2Op`` against C++ QuantLib v1.43.

# C++ parity: ql/methods/finitedifferences/operators/fdmg2op.{hpp,cpp}
# @ v1.43 (6b57206e0).

Expected values come from
``migration-harness/cpp/probes/v143_methods_operators/probe.cpp`` (block 2).
The ``g2rev_*`` cases build the operator with ``direction1``/``direction2``
swapped, so a port that hard-codes directions 0/1 cannot pass.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_g2_op import FdmG2Op
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from tests.methods.finitedifferences.operators.test_v143_ops_a_fixtures import (
    REFERENCE_KEY,
    all_vectors,
    assert_array,
    assert_surface,
    diag_of,
    r_ts,
    vec_const,
    vec_idx,
)

_DT = 0.02
_T1 = 0.25
_T2 = 0.75


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load(REFERENCE_KEY)


def _mesher() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(-0.07, 0.09, 6), Uniform1dMesher(-0.05, 0.06, 5))


def _model() -> G2:
    """# C++ parity: probe.cpp ``G2(rTS, 0.1, 0.011, 0.2, 0.021, -0.45)``."""
    return G2(r_ts(), 0.1, 0.011, 0.2, 0.021, -0.45)


def _build(direction1: int = 0, direction2: int = 1) -> tuple[FdmG2Op, FdmMesherComposite]:
    mesher = _mesher()
    op = FdmG2Op(mesher, _model(), direction1, direction2)
    op.set_time(_T1, _T2)
    return op, mesher


def test_size_and_locations(ref: dict[str, Any]) -> None:
    op, mesher = _build()
    assert op.size() == int(ref["g2_size"])
    assert_array(mesher.locations(0), ref, "g2_locations0")
    assert_array(mesher.locations(1), ref, "g2_locations1")


def test_frozen_short_rate_level(ref: dict[str, Any]) -> None:
    dynamics = _model().dynamics()
    tight(dynamics.short_rate(_T1, 0.0, 0.0), float(ref["g2_short_rate_t1"]))
    tight(dynamics.short_rate(_T2, 0.0, 0.0), float(ref["g2_short_rate_t2"]))


@pytest.mark.parametrize("tag", ["const", "lin0", "lin1", "mix", "idx"])
def test_operator_surface(ref: dict[str, Any], tag: str) -> None:
    op, mesher = _build()
    vectors = dict(all_vectors(mesher))
    assert_surface(ref, "g2", op, vectors[tag], tag, 2, _DT)


def test_foreign_direction_is_zero(ref: dict[str, Any]) -> None:
    op, mesher = _build()
    v = vec_const(mesher)
    assert_array(op.apply_direction(2, v), ref, "g2_const_apply_dir2")
    assert_array(op.solve_splitting(2, v, _DT), ref, "g2_const_solve_dir2")


def test_to_matrix_decomp(ref: dict[str, Any]) -> None:
    op, _ = _build()
    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(ref["g2_decomp_size"])
    for i in range(3):
        assert_array(diag_of(decomp[i]), ref, f"g2_decomp{i}_diag")


def test_swapped_directions(ref: dict[str, Any]) -> None:
    """``direction1 = 1``, ``direction2 = 0`` — the maps must swap with them."""
    op, mesher = _build(1, 0)
    const = vec_const(mesher)
    idx = vec_idx(mesher)
    assert_array(op.apply(const), ref, "g2rev_const_apply")
    assert_array(op.apply(idx), ref, "g2rev_idx_apply")
    assert_array(op.apply_direction(0, idx), ref, "g2rev_idx_apply_dir0")
    assert_array(op.apply_direction(1, idx), ref, "g2rev_idx_apply_dir1")
    assert_array(op.solve_splitting(0, idx, _DT), ref, "g2rev_idx_solve_dir0")
    assert_array(op.solve_splitting(1, idx, _DT), ref, "g2rev_idx_solve_dir1")
