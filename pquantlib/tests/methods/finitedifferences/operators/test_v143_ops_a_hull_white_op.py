"""Cross-validation of ``FdmHullWhiteOp`` against C++ QuantLib v1.43.

# C++ parity: ql/methods/finitedifferences/operators/fdmhullwhiteop.{hpp,cpp}
# @ v1.43 (6b57206e0).

Every expected value comes from
``migration-harness/cpp/probes/v143_methods_operators/probe.cpp`` (block 1),
loaded from ``migration-harness/references/v143/methods/operators.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_hull_white_op import FdmHullWhiteOp
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from tests.methods.finitedifferences.operators.test_v143_ops_a_fixtures import (
    REFERENCE_KEY,
    assert_array,
    assert_surface,
    diag_of,
    r_ts,
    vec_const,
    vec_idx,
    vec_lin,
)

_DT = 0.02
_T1 = 0.2
_T2 = 0.5


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load(REFERENCE_KEY)


def _build() -> tuple[FdmHullWhiteOp, FdmMesherComposite, HullWhite]:
    """# C++ parity: probe.cpp ``block_hull_white_1d``."""
    mesher = FdmMesherComposite(Uniform1dMesher(-0.06, 0.08, 9))
    model = HullWhite(r_ts(), 0.05, 0.012)
    op = FdmHullWhiteOp(mesher, model, 0)
    op.set_time(_T1, _T2)
    return op, mesher, model


def test_size_and_locations(ref: dict[str, Any]) -> None:
    op, mesher, _ = _build()
    assert op.size() == int(ref["hw1d_size"])
    assert_array(mesher.locations(0), ref, "hw1d_locations")


def test_frozen_short_rate_level(ref: dict[str, Any]) -> None:
    """``phi = 0.5 (r(t1,0) + r(t2,0))`` is the drift level ``setTime`` freezes."""
    _, _, model = _build()
    dynamics = model.dynamics()
    tight(dynamics.short_rate(_T1, 0.0), float(ref["hw1d_short_rate_t1"]))
    tight(dynamics.short_rate(_T2, 0.0), float(ref["hw1d_short_rate_t2"]))


@pytest.mark.parametrize("tag", ["const", "lin0", "idx"])
def test_operator_surface(ref: dict[str, Any], tag: str) -> None:
    op, mesher, _ = _build()
    vectors = {
        "const": vec_const(mesher),
        "lin0": vec_lin(mesher, 0),
        "idx": vec_idx(mesher),
    }
    assert_surface(ref, "hw1d", op, vectors[tag], tag, 1, _DT)


def test_other_direction_is_zero(ref: dict[str, Any]) -> None:
    """C++ returns an all-zero array (not ``r``) for a foreign direction."""
    op, mesher, _ = _build()
    v = vec_const(mesher)
    assert_array(op.apply_direction(1, v), ref, "hw1d_const_apply_dir1")
    assert_array(op.solve_splitting(1, v, _DT), ref, "hw1d_const_solve_dir1")


def test_to_matrix_decomp(ref: dict[str, Any]) -> None:
    op, _, _ = _build()
    decomp = op.to_matrix_decomp()
    assert len(decomp) == int(ref["hw1d_decomp_size"])
    assert_array(diag_of(decomp[0]), ref, "hw1d_decomp0_diag")
