"""Cross-validate piecewise / multidim integration + Laplace interpolation.

Probe source: migration-harness/cpp/probes/cluster_w6c/probe.cpp
Reference:    migration-harness/references/cluster/w6c.json
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.laplace_interpolation import (
    LaplaceInterpolation,
    laplace_interpolation,
)
from pquantlib.experimental.math.multidim_integrator import MultidimIntegral
from pquantlib.experimental.math.multidim_quadrature import (
    MultidimGaussianQuadrature,
)
from pquantlib.experimental.math.piecewise_function import PiecewiseFunction
from pquantlib.experimental.math.piecewise_integral import PiecewiseIntegral
from pquantlib.math.integrals.trapezoid import TrapezoidIntegral
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w6c")


# ---- piecewise function (RCLL step) ----


def test_piecewise_function(cpp_ref: dict[str, Any]) -> None:
    pf = PiecewiseFunction([1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0])
    tolerance.exact(pf(0.5), cpp_ref["piecewise_at_0_5"])
    tolerance.exact(pf(1.0), cpp_ref["piecewise_at_1_0"])
    tolerance.exact(pf(1.5), cpp_ref["piecewise_at_1_5"])
    tolerance.exact(pf(2.0), cpp_ref["piecewise_at_2_0"])
    tolerance.exact(pf(3.0), cpp_ref["piecewise_at_3_0"])
    tolerance.exact(pf(5.0), cpp_ref["piecewise_at_5_0"])


def test_piecewise_function_rcll_semantics() -> None:
    # value at a breakpoint equals the value of the interval starting there.
    pf = PiecewiseFunction([1.0, 2.0], [10.0, 20.0, 30.0])
    assert pf(0.999) == 10.0
    assert pf(1.0) == 20.0  # right-continuous at the breakpoint
    assert pf(2.0) == 30.0


def test_piecewise_function_constant() -> None:
    pf = PiecewiseFunction([], [42.0])
    assert pf(-100.0) == 42.0
    assert pf(100.0) == 42.0


def test_piecewise_function_empty_y_rejected() -> None:
    with pytest.raises(LibraryException, match="at least one y"):
        PiecewiseFunction([1.0], [])


# ---- piecewise integral ----


def test_piecewise_integral(cpp_ref: dict[str, Any]) -> None:
    base = TrapezoidIntegral(1e-12, 100000)
    pwi = PiecewiseIntegral(base, [2.0], avoid_critical_points=True)
    tolerance.loose(
        pwi(lambda x: x, 0.0, 4.0),
        cpp_ref["piecewise_integral_x_0_4"],
        reason="trapezoid quadrature with critical-point split.",
    )


# ---- multidim Gaussian quadrature (separability) ----


def test_multidim_quadrature_separable(cpp_ref: dict[str, Any]) -> None:
    q2 = MultidimGaussianQuadrature(2, 12)
    tolerance.loose(
        q2(lambda v: v[0] ** 2 * v[1] ** 2),
        cpp_ref["multidim_quad_x2y2_dim2"],
        reason="Golub-Welsch eigen-decomposition vs C++ TqrEigenDecomposition.",
    )
    q1 = MultidimGaussianQuadrature(1, 12)
    one_d = q1(lambda v: v[0] ** 2)
    tolerance.loose(one_d, cpp_ref["multidim_quad_x2_dim1"])
    # the separable product factorises exactly.
    tolerance.loose(one_d * one_d, cpp_ref["multidim_quad_product_ref"])


def test_multidim_quadrature_dim3(cpp_ref: dict[str, Any]) -> None:
    q3 = MultidimGaussianQuadrature(3, 12)
    tolerance.loose(
        q3(lambda v: v[0] ** 2 * v[1] ** 2 * v[2] ** 2),
        cpp_ref["multidim_quad_x2y2z2_dim3"],
    )


def test_multidim_quadrature_order() -> None:
    q = MultidimGaussianQuadrature(2, 10)
    assert q.order() == 10


# ---- multidim (tensor-product) integrator ----


def test_multidim_integrator_box(cpp_ref: dict[str, Any]) -> None:
    t = TrapezoidIntegral(1e-10, 100000)
    mdi = MultidimIntegral([t, t])
    tolerance.loose(
        mdi(lambda v: v[0] * v[1], [0.0, 0.0], [1.0, 2.0]),
        cpp_ref["multidim_trap_xy_box"],
    )


def test_multidim_integrator_dimension_mismatch() -> None:
    t = TrapezoidIntegral(1e-10, 100000)
    mdi = MultidimIntegral([t, t])
    with pytest.raises(LibraryException, match="Incompatible integration"):
        mdi(lambda v: v[0], [0.0], [1.0])


# ---- Laplace interpolation ----


def test_laplace_inner(cpp_ref: dict[str, Any]) -> None:
    na = math.nan
    m = np.array([[1.0, 2.0, 4.0], [6.0, na, 7.0], [5.0, 3.0, 2.0]])
    laplace_interpolation(m)
    tolerance.tight(float(m[1, 1]), cpp_ref["laplace_inner_1_1"])


def test_laplace_boundary(cpp_ref: dict[str, Any]) -> None:
    na = math.nan
    m = np.array([[1.0, na, 4.0], [6.0, 6.5, 7.0], [5.0, 3.0, 2.0]])
    laplace_interpolation(m)
    tolerance.tight(float(m[0, 1]), cpp_ref["laplace_boundary_0_1"])


def test_laplace_corner(cpp_ref: dict[str, Any]) -> None:
    na = math.nan
    m = np.array([[na, 2.0, 4.0], [6.0, 6.5, 7.0], [5.0, 3.0, 2.0]])
    laplace_interpolation(m)
    tolerance.tight(float(m[0, 0]), cpp_ref["laplace_corner_0_0"])


def test_laplace_full_matrix_unchanged() -> None:
    # A matrix with no NaNs is left untouched.
    m = np.array([[1.0, 2.0, 4.0], [6.0, 6.5, 7.0], [5.0, 3.0, 2.0]])
    original = m.copy()
    laplace_interpolation(m)
    assert np.allclose(m, original)


def test_laplace_1d_is_linear() -> None:
    # For n=1 (a single row / single included dim) the method is linear
    # interpolation with flat extrapolation.
    na = math.nan
    m = np.array([[1.0, na, 3.0]])
    laplace_interpolation(m)
    tolerance.tight(float(m[0, 1]), 2.0)


def test_laplace_interpolation_class_query() -> None:
    na = math.nan
    grid_y = [0.0, 1.0, 2.0]
    grid_x = [0.0, 1.0, 2.0]
    values = [[1.0, 2.0, 4.0], [6.0, na, 7.0], [5.0, 3.0, 2.0]]
    interp = LaplaceInterpolation(
        lambda c: values[c[0]][c[1]], [grid_y, grid_x]
    )
    tolerance.tight(interp([1, 1]), 4.5)
