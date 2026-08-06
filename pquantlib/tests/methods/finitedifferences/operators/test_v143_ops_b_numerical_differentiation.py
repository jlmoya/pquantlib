"""Cross-validation of NumericalDifferentiation + NthOrderDerivativeOp vs C++ v1.43.

# C++ parity: ql/methods/finitedifferences/operators/numericaldifferentiation.{hpp,cpp}
# + ql/methods/finitedifferences/operators/nthorderderivativeop.{hpp,cpp} @ v1.43.

Reference values come from ``migration-harness/cpp/probes/v143_methods_operators2``
(→ ``migration-harness/references/v143/methods/operators2.json``), built against
the pinned QuantLib v1.43 submodule.

Tolerance policy
----------------
* **Fornberg weights + offsets: EXACT.** The port reproduces the C++
  recurrence bit-for-bit, including the compiler's fused multiply-add
  contraction of the two ``a*b - c`` sub-expressions (see the module
  docstring of ``numerical_differentiation``). Anything weaker would
  hide a genuine algorithmic substitution — e.g. solving the Vandermonde
  system with ``numpy.linalg.solve``, which agrees only to a few ULP.
* **``operator()`` results and operator applications: TIGHT.** These add
  ``sum(w_i * f(x + x_i))`` over stencils whose weights scale like
  ``h^-M``, so catastrophic cancellation costs a handful of digits; the
  measured worst case here is 1.4e-13 relative, well inside TIGHT's
  1e-12 relative bound.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.methods.finitedifferences.operators.nth_order_derivative_op import (
    NthOrderDerivativeOp,
)
from pquantlib.methods.finitedifferences.operators.numerical_differentiation import (
    NumericalDifferentiation,
    Scheme,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight

REF: dict[str, Any] = reference_reader.load("v143/methods/operators2")


def _ref_list(key: str) -> list[float]:
    value: Any = REF[key]
    assert isinstance(value, list)
    return [float(v) for v in cast("list[Any]", value)]


def _assert_exact(key: str, got: Array) -> None:
    expected = _ref_list(key)
    got_list = [float(v) for v in np.asarray(got).ravel()]
    assert len(got_list) == len(expected), key
    for i, (g, e) in enumerate(zip(got_list, expected, strict=True)):
        exact(g, e, reason=f"{key}[{i}]")


def _assert_tight(key: str, got: Array) -> None:
    expected = _ref_list(key)
    got_list = [float(v) for v in np.asarray(got).ravel()]
    assert len(got_list) == len(expected), key
    for i, (g, e) in enumerate(zip(got_list, expected, strict=True)):
        tight(g, e, reason=f"{key}[{i}]")


def _const_vec(mesher: FdmMesher, c: float = 1.0) -> Array:
    return np.full(mesher.layout().size(), c, dtype=np.float64)


def _lin_vec(mesher: FdmMesher, direction: int) -> Array:
    return mesher.locations(direction).copy()


def _quad_vec(mesher: FdmMesher, direction: int) -> Array:
    x = mesher.locations(direction)
    return x * x


def _prod_vec(mesher: FdmMesher) -> Array:
    return mesher.locations(0) * mesher.locations(1)


def _ramp_vec(mesher: FdmMesher) -> Array:
    """Strictly non-symmetric probe vector — catches direction mix-ups."""
    n = mesher.layout().size()
    return np.array([1.0 + 0.25 * i - 0.03 * i * i for i in range(n)], dtype=np.float64)


def _dense(matrix: Any) -> Array:
    return np.asarray(matrix.todense(), dtype=np.float64).ravel()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


# ===========================================================================
# NumericalDifferentiation — scheme-based constructor
# ===========================================================================

_SCHEME_CASES: list[tuple[str, int, float, int, Scheme]] = [
    ("nd_central_o1_n3", 1, 0.1, 3, Scheme.Central),
    ("nd_central_o1_n5", 1, 0.05, 5, Scheme.Central),
    ("nd_central_o2_n3", 2, 0.1, 3, Scheme.Central),
    ("nd_central_o2_n5", 2, 0.1, 5, Scheme.Central),
    ("nd_central_o2_n7", 2, 0.25, 7, Scheme.Central),
    ("nd_central_o3_n7", 3, 0.1, 7, Scheme.Central),
    ("nd_central_o4_n9", 4, 0.2, 9, Scheme.Central),
    ("nd_backward_o1_n3", 1, 0.1, 3, Scheme.Backward),
    ("nd_backward_o1_n5", 1, 0.02, 5, Scheme.Backward),
    ("nd_backward_o2_n4", 2, 0.1, 4, Scheme.Backward),
    ("nd_forward_o1_n3", 1, 0.1, 3, Scheme.Forward),
    ("nd_forward_o2_n4", 2, 0.1, 4, Scheme.Forward),
    ("nd_forward_o3_n5", 3, 0.15, 5, Scheme.Forward),
]


@pytest.mark.parametrize(("tag", "order", "h", "steps", "scheme"), _SCHEME_CASES)
def test_scheme_offsets_and_weights_match_cpp(
    tag: str, order: int, h: float, steps: int, scheme: Scheme
) -> None:
    """Offsets and Fornberg weights are bit-identical to C++ v1.43."""
    nd = NumericalDifferentiation.from_scheme(None, order, h, steps, scheme)
    _assert_exact(f"{tag}_offsets", nd.offsets())
    _assert_exact(f"{tag}_weights", nd.weights())


_IRREGULAR_CASES: list[tuple[str, list[float], int]] = [
    ("nd_irregular_a_o1", [-0.3, -0.1, 0.0, 0.15, 0.4], 1),
    ("nd_irregular_a_o2", [-0.3, -0.1, 0.0, 0.15, 0.4], 2),
    ("nd_irregular_a_o3", [-0.3, -0.1, 0.0, 0.15, 0.4], 3),
    ("nd_irregular_b_o1", [0.0, 0.1, 0.3, 0.7], 1),
    # Fornberg does not require the offsets to be sorted.
    ("nd_unsorted_o2", [0.2, -0.4, 0.0, 0.05], 2),
]


@pytest.mark.parametrize(("tag", "offsets", "order"), _IRREGULAR_CASES)
def test_irregular_offsets_weights_match_cpp(
    tag: str, offsets: list[float], order: int
) -> None:
    """Weights on arbitrarily spaced (and unsorted) grids match C++ bit-for-bit."""
    nd = NumericalDifferentiation(None, order, np.array(offsets, dtype=np.float64))
    _assert_exact(f"{tag}_offsets", nd.offsets())
    _assert_exact(f"{tag}_weights", nd.weights())


def _probe_function(x: float) -> float:
    """The exact expression the C++ probe differentiates.

    # C++ probe: ``std::exp(0.7*x)*std::sin(1.3*x) + 0.25*x*x*x``.
    """
    return math.exp(0.7 * x) * math.sin(1.3 * x) + 0.25 * x * x * x


def test_probe_function_matches_cpp() -> None:
    """Sanity check: the Python transcription of the probe's ``f`` agrees."""
    tight(_probe_function(0.5), float(REF["nd_probe_function_at_0_5"]))
    tight(_probe_function(-1.25), float(REF["nd_probe_function_at_m1_25"]))


def test_call_operator_matches_cpp() -> None:
    """``operator()`` — weighted sum over the stencil.

    TIGHT rather than EXACT: the sum amplifies round-off by ``h^-M``
    (worst case measured here is 1.4e-13 relative, for the 4-point
    backward first derivative with ``h = 0.005``).
    """
    nd1 = NumericalDifferentiation.from_scheme(
        _probe_function, 1, 0.01, 5, Scheme.Central
    )
    tight(nd1(0.5), float(REF["nd_call_central_o1_n5_at_0_5"]))
    tight(nd1(-1.25), float(REF["nd_call_central_o1_n5_at_m1_25"]))

    nd2 = NumericalDifferentiation.from_scheme(
        _probe_function, 2, 0.02, 7, Scheme.Central
    )
    tight(nd2(0.5), float(REF["nd_call_central_o2_n7_at_0_5"]))

    nd3 = NumericalDifferentiation.from_scheme(
        _probe_function, 1, 0.005, 4, Scheme.Backward
    )
    tight(nd3(0.5), float(REF["nd_call_backward_o1_n4_at_0_5"]))

    nd4 = NumericalDifferentiation(
        _probe_function, 1, np.array([-0.3, -0.1, 0.0, 0.15, 0.4], dtype=np.float64)
    )
    tight(nd4(0.5), float(REF["nd_call_irregular_a_o1_at_0_5"]))


def test_call_operator_skips_negligible_weights() -> None:
    """Weights with ``|w| <= QL_EPSILON^2`` never trigger a call to ``f``.

    # C++ parity: the ``std::fabs(w_[i]) > QL_EPSILON*QL_EPSILON`` guard
    # in ``NumericalDifferentiation::operator()``.
    """
    calls: list[float] = []

    def recording(x: float) -> float:
        calls.append(x)
        return 1.0

    # h=0.1 central 3-point: the centre weight is ~5.6e-16 (round-off, but
    # far above eps^2 = 4.9e-32), so every offset is visited.
    nd = NumericalDifferentiation.from_scheme(recording, 1, 0.1, 3, Scheme.Central)
    nd(0.0)
    assert len(calls) == 3

    # Unit offsets make the centre weight come out as an exact 0.0, and
    # the guard then skips that evaluation entirely.
    calls.clear()
    nd_zero = NumericalDifferentiation(
        recording, 1, np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    )
    assert float(nd_zero.weights()[1]) == 0.0
    nd_zero(0.0)
    assert calls == [-1.0, 1.0]


def test_no_function_raises() -> None:
    """Calling a weights-only instance raises rather than silently returning 0."""
    nd = NumericalDifferentiation.from_scheme(None, 1, 0.1, 3, Scheme.Central)
    with pytest.raises(LibraryException):
        nd(0.0)


def test_constructor_requires_enough_points() -> None:
    """``N > M`` is required (C++ ``QL_REQUIRE`` in ``calcWeights``)."""
    with pytest.raises(LibraryException):
        NumericalDifferentiation(None, 3, np.array([0.0, 1.0, 2.0], dtype=np.float64))


def test_central_scheme_requires_odd_step_count() -> None:
    """Central needs an odd number of steps greater than two."""
    with pytest.raises(LibraryException):
        NumericalDifferentiation.from_scheme(None, 1, 0.1, 4, Scheme.Central)
    with pytest.raises(LibraryException):
        NumericalDifferentiation.from_scheme(None, 1, 0.1, 2, Scheme.Central)


# ===========================================================================
# NthOrderDerivativeOp
# ===========================================================================


def _mesher_1d() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(0.0, 1.2, 7))


def _mesher_1d_b() -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(-2.0, 3.0, 9))


def _mesher_2d() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(0.0, 1.0, 5), Uniform1dMesher(-1.0, 2.0, 4)
    )


def test_nth_order_mesher_locations_match_cpp() -> None:
    _assert_tight("nth_1d_locations", _mesher_1d().locations(0))
    _assert_tight("nth_1db_locations", _mesher_1d_b().locations(0))
    _assert_tight("nth_2d_locations0", _mesher_2d().locations(0))
    _assert_tight("nth_2d_locations1", _mesher_2d().locations(1))


_NTH_1D_CASES: list[tuple[str, int, int]] = [
    ("nth_1d_o1_p3", 1, 3),
    ("nth_1d_o1_p4", 1, 4),  # even stencil — exercises the isEven branch
    ("nth_1d_o2_p3", 2, 3),
    ("nth_1d_o2_p5", 2, 5),
    ("nth_1d_o3_p5", 3, 5),
    ("nth_1d_o4_p5", 4, 5),
]


@pytest.mark.parametrize(("tag", "order", "n_points"), _NTH_1D_CASES)
def test_nth_order_1d_apply_matches_cpp(tag: str, order: int, n_points: int) -> None:
    mesher = _mesher_1d()
    op = NthOrderDerivativeOp(0, order, n_points, mesher)
    _assert_tight(f"{tag}_apply_const", op.apply(_const_vec(mesher)))
    _assert_tight(f"{tag}_apply_lin", op.apply(_lin_vec(mesher, 0)))
    _assert_tight(f"{tag}_apply_quad", op.apply(_quad_vec(mesher, 0)))
    _assert_tight(f"{tag}_apply_ramp", op.apply(_ramp_vec(mesher)))


_NTH_MATRIX_CASES: list[tuple[str, int, int]] = [
    ("nth_1d_o1_p3_matrix", 1, 3),
    ("nth_1d_o1_p4_matrix", 1, 4),
    ("nth_1d_o2_p5_matrix", 2, 5),
]


@pytest.mark.parametrize(("key", "order", "n_points"), _NTH_MATRIX_CASES)
def test_nth_order_1d_dense_matrix_matches_cpp(
    key: str, order: int, n_points: int
) -> None:
    """Full 7x7 dense matrix — the sharpest possible pin on the stencil placement."""
    mesher = _mesher_1d()
    op = NthOrderDerivativeOp(0, order, n_points, mesher)
    _assert_tight(key, _dense(op.to_matrix()))


def test_nth_order_wide_stencil_on_second_grid() -> None:
    mesher = _mesher_1d_b()
    op = NthOrderDerivativeOp(0, 2, 7, mesher)
    _assert_tight("nth_1db_o2_p7_apply_quad", op.apply(_quad_vec(mesher, 0)))
    _assert_tight("nth_1db_o2_p7_apply_ramp", op.apply(_ramp_vec(mesher)))


_NTH_2D_CASES: list[tuple[str, int, int, int]] = [
    ("nth_2d_d0_o1_p3", 0, 1, 3),
    ("nth_2d_d1_o1_p3", 1, 1, 3),
    ("nth_2d_d1_o2_p3", 1, 2, 3),
    ("nth_2d_d0_o2_p4", 0, 2, 4),
]


@pytest.mark.parametrize(("tag", "direction", "order", "n_points"), _NTH_2D_CASES)
def test_nth_order_2d_apply_matches_cpp(
    tag: str, direction: int, order: int, n_points: int
) -> None:
    mesher = _mesher_2d()
    op = NthOrderDerivativeOp(direction, order, n_points, mesher)
    _assert_tight(f"{tag}_apply_const", op.apply(_const_vec(mesher)))
    _assert_tight(f"{tag}_apply_lin0", op.apply(_lin_vec(mesher, 0)))
    _assert_tight(f"{tag}_apply_lin1", op.apply(_lin_vec(mesher, 1)))
    _assert_tight(f"{tag}_apply_prod", op.apply(_prod_vec(mesher)))
    _assert_tight(f"{tag}_apply_ramp", op.apply(_ramp_vec(mesher)))


def test_nth_order_2d_dense_matrix_matches_cpp() -> None:
    mesher = _mesher_2d()
    op = NthOrderDerivativeOp(1, 1, 3, mesher)
    _assert_tight("nth_2d_d1_o1_p3_matrix", _dense(op.to_matrix()))


def test_nth_order_rejects_too_wide_stencil() -> None:
    """``1 < nPoints <= dim[direction]`` is required."""
    mesher = _mesher_1d()
    with pytest.raises(LibraryException):
        NthOrderDerivativeOp(0, 1, 8, mesher)
    with pytest.raises(LibraryException):
        NthOrderDerivativeOp(0, 1, 1, mesher)
