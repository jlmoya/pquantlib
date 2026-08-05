"""Cross-validate KernelInterpolation2D against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``kernel_interpolation_2d`` section. Two cases, both lifted from the C++
test-suite (``testKernelInterpolation2D``):

* a Gaussian kernel over a 10 x 3 grid — a well-behaved surface;
* the test-suite's Epanechnikov kernel over a 4 x 8 grid — compactly
  supported on ``|u| <= 1`` while the x-nodes are 10 apart, so most of the
  plane sees *no* node at all and the interpolation divides 0 by 0.

**Layout.** C++ ``KernelInterpolation2D`` is the only 2-D interpolation in
QuantLib whose ``zData`` is indexed ``[x][y]``. The reference stores the
matrix in that C++ orientation under ``z``; PQuantLib normalises on
``[y][x]`` like every other 2-D interpolation, so the tests transpose on the
way in. If that transpose were dropped the Gaussian case would still
*construct* (the matrix is 10x3, not square, so a shape check catches it) —
which is why the transpose is spelled out at each call site rather than
hidden in a helper.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.interpolations.kernel_interpolation_2d import KernelInterpolation2D
from pquantlib.math.kernel_functions import GaussianKernel
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _epanechnikov(u: float) -> float:
    """The C++ test-suite's kernel (test-suite/interpolations.cpp:199-206)."""
    if abs(u) <= 1:
        return (3.0 / 4.0) * (1 - u * u)
    return 0.0


def _gaussian_case(cpp: dict[str, Any]) -> KernelInterpolation2D:
    block = cpp["kernel_interpolation_2d"]["gaussian"]
    interp = KernelInterpolation2D(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        # C++ z is [x][y]; PQuantLib is [y][x].
        np.asarray(block["z"], dtype=np.float64).T,
        GaussianKernel(float(block["average"]), float(block["sigma"])),
    )
    interp.enable_extrapolation()
    return interp


def _epanechnikov_case(cpp: dict[str, Any]) -> KernelInterpolation2D:
    block = cpp["kernel_interpolation_2d"]["epanechnikov"]
    interp = KernelInterpolation2D(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
        np.asarray(block["z"], dtype=np.float64).T,
        _epanechnikov,
    )
    interp.enable_extrapolation()
    return interp


def test_cpp_layout_is_documented(cpp: dict[str, Any]) -> None:
    """The reference records which way round the C++ matrix is."""
    assert cpp["kernel_interpolation_2d"]["z_layout"] == "rows_are_x"


def test_gaussian_values_match_cpp_within_solve_amplification(
    cpp: dict[str, Any],
) -> None:
    """Gaussian-kernel surface matches C++ within ``cond(M) * eps``.

    Same reasoning as the 1-D test: C++ solves the 30x30 system with pivoted
    QR, this port with LU, and two backward-stable solves of the same system
    agree to ``cond(M) * eps`` relative. Here ``cond(M)`` is about 6.6e4, so
    the bound is ~1.5e-11 and the measured disagreement is ~1.1e-12 — just
    outside TIGHT's 1e-12, which is why the derived bound is used rather than
    the tier.
    """
    block = cpp["kernel_interpolation_2d"]["gaussian"]
    interp = _gaussian_case(cpp)
    cond = float(np.linalg.cond(interp.kernel_matrix))
    rel_tol = max(1e-12, cond * QL_EPSILON)
    reason = (
        f"linear-solve amplification: cond(M) = {cond:.3g} on the 30x30 kernel "
        f"system, so QR (C++) and LU (numpy) agree to cond(M) * eps = "
        f"{cond * QL_EPSILON:.3g} relative"
    )
    for x, y, expected in block["evals"]:
        tolerance.custom(
            interp(float(x), float(y)),
            float(expected),
            abs_tol=1e-14,
            rel_tol=rel_tol,
            reason=reason,
        )


def test_gaussian_pillars_reproduce_inputs(cpp: dict[str, Any]) -> None:
    """The surface passes through every grid node.

    The property C++'s own test-suite asserts, at its own 1e-10 tolerance —
    which is also the constructor's ``invPrec_`` default, i.e. the strongest
    node-recovery claim the algorithm makes.
    """
    block = cpp["kernel_interpolation_2d"]["gaussian"]
    interp = _gaussian_case(cpp)
    z_cpp = np.asarray(block["z"], dtype=np.float64)  # [x][y]
    for i, x in enumerate(block["xs"]):
        for j, y in enumerate(block["ys"]):
            tolerance.custom(
                interp(float(x), float(y)),
                float(z_cpp[i, j]),
                abs_tol=1e-10,
                rel_tol=0.0,
                reason=(
                    "the constructor's inversion guard is |M alpha - z| < 1e-10 "
                    "(kernelinterpolation2d.hpp:175), the same bound the C++ "
                    "test-suite uses for this grid"
                ),
            )


def test_epanechnikov_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Compact-kernel surface matches C++ to TIGHT where it is finite.

    ``cond(M)`` here is about 1.1e3, so the solve amplifies nothing and TIGHT
    is the right tier. The non-finite entries are asserted separately.
    """
    block = cpp["kernel_interpolation_2d"]["epanechnikov"]
    interp = _epanechnikov_case(cpp)
    for x, y, expected in block["evals"]:
        if isinstance(expected, str):
            continue
        tolerance.tight(interp(float(x), float(y)), float(expected))


def test_epanechnikov_out_of_support_is_nan(cpp: dict[str, Any]) -> None:
    """Outside every node's support C++ computes ``0.0 / 0.0`` and gets nan.

    Python raises ``ZeroDivisionError`` on ``0.0 / 0.0``, so the port has to
    spell the IEEE result out. This asserts it does — and that the reference
    actually contains such points, so the test cannot silently become vacuous.
    """
    block = cpp["kernel_interpolation_2d"]["epanechnikov"]
    interp = _epanechnikov_case(cpp)
    nan_count = 0
    for x, y, expected in block["evals"]:
        if expected == "nan":
            nan_count += 1
            assert math.isnan(interp(float(x), float(y))), (x, y)
    assert nan_count > 0, "reference contains no out-of-support points"


def test_epanechnikov_pillars_reproduce_inputs(cpp: dict[str, Any]) -> None:
    """Node recovery for the compact kernel, at the C++ test-suite's 1e-10."""
    block = cpp["kernel_interpolation_2d"]["epanechnikov"]
    interp = _epanechnikov_case(cpp)
    z_cpp = np.asarray(block["z"], dtype=np.float64)  # [x][y]
    for i, x in enumerate(block["xs"]):
        for j, y in enumerate(block["ys"]):
            tolerance.custom(
                interp(float(x), float(y)),
                float(z_cpp[i, j]),
                abs_tol=1e-10,
                rel_tol=0.0,
                reason="the constructor's inversion guard, |M alpha - z| < 1e-10",
            )


def test_wrong_z_shape_is_rejected(cpp: dict[str, Any]) -> None:
    """Feeding the C++ orientation straight through must fail, not silently work."""
    block = cpp["kernel_interpolation_2d"]["gaussian"]
    with pytest.raises(Exception, match="does not match"):
        KernelInterpolation2D(
            np.asarray(block["xs"], dtype=np.float64),
            np.asarray(block["ys"], dtype=np.float64),
            np.asarray(block["z"], dtype=np.float64),  # NOT transposed
            GaussianKernel(float(block["average"]), float(block["sigma"])),
        )


def test_set_inverse_result_precision_applies_from_next_update(
    cpp: dict[str, Any],
) -> None:
    """C++ ``setInverseResultPrecision`` does not re-solve; nor does this.

    Construction already succeeded under the 1e-10 default. Tightening the
    bound to something unattainable must therefore leave the object usable
    until ``update()`` is called, and only then fail.
    """
    interp = _gaussian_case(cpp)
    interp.set_inverse_result_precision(1e-300)
    _ = interp(0.5, 2.0)  # still fine: no re-solve happened
    with pytest.raises(Exception, match="inversion failed"):
        interp.update()
