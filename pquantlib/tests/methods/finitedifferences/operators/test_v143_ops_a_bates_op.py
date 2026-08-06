"""Cross-validation of ``FdmBatesOp`` / ``IntegroIntegrand`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/operators/fdmbatesop.{hpp,cpp}
# @ v1.43 (6b57206e0).

Expected values come from
``migration-harness/cpp/probes/v143_methods_operators/probe.cpp`` (block 6),
which covers both an empty boundary-condition set and a Lower+Upper
``FdmDirichletBoundary`` pair, and both Gauss-Hermite orders 8 and 16.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_bates_op import (
    FdmBatesOp,
    IntegroIntegrand,
)
from pquantlib.methods.finitedifferences.utilities.fdm_dirichlet_boundary import (
    FdmDirichletBoundary,
)
from pquantlib.processes.bates_process import BatesProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from tests.methods.finitedifferences.operators.test_v143_ops_a_fixtures import (
    REFERENCE_KEY,
    all_vectors,
    assert_array,
    assert_surface,
    q_ts,
    r_ts,
    vec_const,
    vec_idx,
    vec_mix,
)

_DT = 0.02
_T1 = 0.1
_T2 = 0.35
_ORDER = 8


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load(REFERENCE_KEY)


def _mesher() -> FdmMesherComposite:
    """# C++ parity: probe.cpp ``block_bates``."""
    return FdmMesherComposite(
        Uniform1dMesher(math.log(70.0), math.log(140.0), 7),
        Uniform1dMesher(0.005, 0.4, 4),
    )


def _bates_process() -> BatesProcess:
    return BatesProcess(
        risk_free_rate=r_ts(),
        dividend_yield=q_ts(),
        s0=SimpleQuote(100.0),
        v0=0.09,
        kappa=1.5,
        theta=0.04,
        sigma=0.4,
        rho=-0.6,
        lambda_=0.4,
        nu=-0.5,
        delta=0.3,
    )


def _build(
    bc_set: list[FdmDirichletBoundary] | None = None, order: int = _ORDER
) -> tuple[FdmBatesOp, FdmMesherComposite]:
    mesher = _mesher()
    op = FdmBatesOp(mesher, _bates_process(), bc_set or [], order)
    op.set_time(_T1, _T2)
    return op, mesher


def test_mesher_locations(ref: dict[str, Any]) -> None:
    mesher = _mesher()
    assert_array(mesher.locations(0), ref, "bates_locations0")
    assert_array(mesher.locations(1), ref, "bates_locations1")


def test_size(ref: dict[str, Any]) -> None:
    op, _ = _build()
    assert op.size() == int(ref["bates_size"])


@pytest.mark.parametrize("tag", ["const", "lin0", "lin1", "mix", "idx"])
def test_operator_surface(ref: dict[str, Any], tag: str) -> None:
    """Empty boundary set — ``apply`` = Heston + integro, ``apply_mixed`` too."""
    op, mesher = _build()
    vectors = dict(all_vectors(mesher))
    assert_surface(ref, "bates", op, vectors[tag], tag, 2, _DT)


def test_higher_integration_order(ref: dict[str, Any]) -> None:
    """Order 16 must move the integral — pins the quadrature-order plumbing."""
    op, mesher = _build(order=16)
    idx = vec_idx(mesher)
    assert_array(op.apply(idx), ref, "bates_ord16_idx_apply")
    assert_array(op.apply_mixed(idx), ref, "bates_ord16_idx_apply_mixed")


def _boundaries(mesher: FdmMesherComposite, ref: dict[str, Any]) -> list[FdmDirichletBoundary]:
    """# C++ parity: probe.cpp — Lower(2.5) + Upper(-1.5) on direction 0."""
    locations = mesher.locations(0)
    tight(float(locations[0]), float(ref["bates_bc_x_lower"]))
    tight(float(locations[mesher.layout().size() - 1]), float(ref["bates_bc_x_upper"]))
    return [
        FdmDirichletBoundary(mesher, 2.5, 0, BoundaryConditionSide.LOWER),
        FdmDirichletBoundary(mesher, -1.5, 0, BoundaryConditionSide.UPPER),
    ]


def test_dirichlet_boundary_set(ref: dict[str, Any]) -> None:
    """Lower (2.5) + Upper (-1.5) Dirichlet clamps inside the jump integral."""
    mesher = _mesher()
    op = FdmBatesOp(mesher, _bates_process(), _boundaries(mesher, ref), _ORDER)
    op.set_time(_T1, _T2)
    const = vec_const(mesher)
    idx = vec_idx(mesher)
    assert_array(op.apply(const), ref, "bates_bc_const_apply")
    assert_array(op.apply_mixed(const), ref, "bates_bc_const_apply_mixed")
    assert_array(op.apply(idx), ref, "bates_bc_idx_apply")
    assert_array(op.apply_mixed(idx), ref, "bates_bc_idx_apply_mixed")
    assert_array(op.apply_mixed(vec_mix(mesher)), ref, "bates_bc_mix_apply_mixed")


def test_boundary_set_changes_the_integral(ref: dict[str, Any]) -> None:
    """The Dirichlet set must actually bite — otherwise the test above is vacuous."""
    mesher = _mesher()
    plain, _ = _build()
    clamped = FdmBatesOp(mesher, _bates_process(), _boundaries(mesher, ref), _ORDER)
    clamped.set_time(_T1, _T2)
    idx = vec_idx(mesher)
    assert not np.allclose(plain.apply(idx), clamped.apply(idx))


def test_integro_integrand_matches_gauss_hermite_node() -> None:
    """``IntegroIntegrand(y)`` = ``exp(-y^2) * f(x + sqrt(2) delta y + nu)``.

    # C++ parity: fdmbatesop.cpp:68-85. Checked against the definition with
    # a hand-built linear interpolation (no C++ reference needed: this is a
    # closed-form identity, and the quadrature that consumes it *is*
    # cross-validated above).
    """
    xs: Array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys: Array = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float64)
    interpl = LinearInterpolation(xs, ys)
    delta, nu = 0.3, -0.5
    integrand = IntegroIntegrand(interpl, [], 1.5, delta, nu)
    for y in (-1.25, 0.0, 0.75):
        x = 1.5 + math.sqrt(2.0) * delta * y + nu
        tight(integrand(y), math.exp(-y * y) * interpl(x, allow_extrapolation=True))


def test_non_dirichlet_boundary_rejected() -> None:
    """# C++ parity: the ``dynamic_pointer_cast`` + ``QL_REQUIRE`` guard."""
    mesher = _mesher()
    op = FdmBatesOp(mesher, _bates_process(), [object()], _ORDER)
    op.set_time(_T1, _T2)
    with pytest.raises(LibraryException, match="Dirichlet boundary conditions"):
        op.apply(vec_const(mesher))


def test_to_matrix_decomp_not_implemented() -> None:
    """# C++ parity: ``QL_FAIL("not implemented")`` (fdmbatesop.cpp:120-122)."""
    op, _ = _build()
    with pytest.raises(LibraryException, match="not implemented"):
        op.to_matrix_decomp()
