"""Cross-validate KernelInterpolation against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``kernel_interpolation`` section. The grid and the three y-sets are the ones
the C++ test-suite itself uses (``testKernelInterpolation``); the Gaussian
bandwidths span 0.05 (kernel matrix near-diagonal, ``cond ~ 1``) to 2.55
(near-singular, ``cond ~ 5e9``), which is what makes the tolerance story
below necessary.

Each case pins the intermediate ``gamma`` vector and the gamma-normalised
kernel matrix ``M`` alongside the interpolated values, so a failure localises
to either the matrix construction or the linear solve rather than to "the
number came out wrong".
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.interpolations.kernel_interpolation import KernelInterpolation
from pquantlib.math.kernel_functions import GaussianKernel
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _build(block: dict[str, Any], case: dict[str, Any]) -> KernelInterpolation:
    xs = np.asarray(block["xs"], dtype=np.float64)
    ys = np.asarray(case["ys"], dtype=np.float64)
    interp = KernelInterpolation(xs, ys, GaussianKernel(0.0, float(case["sigma"])))
    interp.enable_extrapolation()
    return interp


def test_kernel_matrix_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """The gamma-normalised matrix ``M`` matches C++ to TIGHT.

    ``M`` is built from ``K(|x_r - x_c|) / gamma(x_r)`` with no solve
    involved, so this isolates the construction from the conditioning.
    """
    block = cpp["kernel_interpolation"]
    for case in block["cases"]:
        interp = _build(block, case)
        expected = np.asarray(case["m"], dtype=np.float64)
        actual = interp.kernel_matrix
        assert actual.shape == expected.shape
        for row_a, row_e in zip(actual, expected, strict=True):
            for a, e in zip(row_a, row_e, strict=True):
                tolerance.tight(float(a), float(e))


def test_gamma_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """``gamma(x_r) = sum_i K(|x_r - x_i|)`` matches C++ to TIGHT.

    Recomputed here from the public kernel rather than read off the
    interpolation, because C++ keeps it as a private helper too.
    """
    block = cpp["kernel_interpolation"]
    xs = [float(x) for x in block["xs"]]
    for case in block["cases"]:
        kernel = GaussianKernel(0.0, float(case["sigma"]))
        for x_r, expected in zip(xs, case["gamma"], strict=True):
            gamma = 0.0
            for x_i in xs:
                gamma += kernel(abs(x_r - x_i))
            tolerance.tight(gamma, float(expected))


def test_values_match_cpp_within_solve_amplification(cpp: dict[str, Any]) -> None:
    """Interpolated values match C++ within ``cond(M) * eps``.

    Not a fixed tier, and deliberately so. C++ solves ``M alpha = y`` with
    MINPACK's pivoted Householder QR; this port uses ``numpy.linalg.solve``
    (LAPACK LU with partial pivoting). Both are backward-stable, so each
    returns the exact solution of a system perturbed by ``O(eps)`` — and the
    two answers therefore differ by at most ``cond(M) * eps`` in relative
    terms. That bound is the tolerance, computed per case from the ``M``
    pinned in the reference, with the TIGHT band as a floor for the
    well-conditioned cases where the solve is not the limiting error.

    The bound is not academic: over the five bandwidths ``cond(M)`` runs from
    1.0 to 5.5e9, and the measured disagreement tracks it (4.8e-16 at
    ``sigma = 0.05``, 8.1e-9 at ``sigma = 2.55``) at 0.5-25 % of the
    predicted amplification. A single fixed tolerance would either fail the
    ill-conditioned cases or wave through a real regression in the
    well-conditioned ones.
    """
    block = cpp["kernel_interpolation"]
    eval_xs = [float(x) for x in block["eval_xs"]]
    for case in block["cases"]:
        interp = _build(block, case)
        cond = float(np.linalg.cond(np.asarray(case["m"], dtype=np.float64)))
        rel_tol = max(1e-12, cond * QL_EPSILON)
        reason = (
            f"linear-solve amplification: cond(M) = {cond:.3g} at sigma = "
            f"{case['sigma']}, so QR (C++) and LU (numpy) agree only to "
            f"cond(M) * eps = {cond * QL_EPSILON:.3g} relative"
        )
        for x, expected in zip(eval_xs, case["values"], strict=True):
            tolerance.custom(
                interp(x), float(expected), abs_tol=1e-14, rel_tol=rel_tol, reason=reason
            )


def test_pillars_reproduce_inputs(cpp: dict[str, Any]) -> None:
    """The interpolation passes through its own nodes.

    This is the property C++'s own test-suite asserts (to 2e-5), and it is
    the point of the ``|M alpha - y| < epsilon`` guard in the constructor.
    We hold it to that guard's own bound rather than to 2e-5.
    """
    block = cpp["kernel_interpolation"]
    xs = [float(x) for x in block["xs"]]
    for case in block["cases"]:
        interp = _build(block, case)
        for x, y in zip(xs, case["ys"], strict=True):
            tolerance.custom(
                interp(x),
                float(y),
                abs_tol=1e-7,
                rel_tol=0.0,
                reason=(
                    "the constructor's own inversion guard is |M alpha - y| < 1e-7 "
                    "(kernelinterpolation.hpp:162), so that is the strongest "
                    "node-recovery claim the algorithm makes"
                ),
            )


def test_calculus_is_not_implemented(cpp: dict[str, Any]) -> None:
    """``primitive`` / ``derivative`` / ``second_derivative`` all QL_FAIL in C++."""
    block = cpp["kernel_interpolation"]
    interp = _build(block, block["cases"][0])
    with pytest.raises(LibraryException, match="Primitive"):
        interp.primitive(0.5)
    with pytest.raises(LibraryException, match="First derivative"):
        interp.derivative(0.5)
    with pytest.raises(LibraryException, match="Second derivative"):
        interp.second_derivative(0.5)


def test_inversion_guard_fires_when_epsilon_is_unattainable(
    cpp: dict[str, Any],
) -> None:
    """C++ has no ``det(M) != 0`` pre-check — the residual bound IS the guard.

    Driving ``epsilon`` below any achievable residual must therefore fail
    construction rather than silently returning a bad fit. The
    ``sigma = 2.55`` case is used because its ``cond(M) ~ 5e9`` guarantees a
    non-zero residual to trip on.
    """
    block = cpp["kernel_interpolation"]
    case = next(c for c in block["cases"] if float(c["sigma"]) == 2.55)
    xs = np.asarray(block["xs"], dtype=np.float64)
    ys = np.asarray(case["ys"], dtype=np.float64)
    with pytest.raises(LibraryException, match="Inversion failed"):
        KernelInterpolation(xs, ys, GaussianKernel(0.0, 2.55), epsilon=1e-300)


def test_non_symmetric_kernel_sees_absolute_differences() -> None:
    """C++ ``kernelAbs`` always passes ``|x1 - x2|``, never the signed value.

    A kernel that answers differently on negative arguments would expose a
    port that forwarded ``x1 - x2``; here it never sees one.
    """
    seen: list[float] = []

    def recorder(u: float) -> float:
        seen.append(u)
        return math.exp(-u * u)

    xs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ys = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    interp = KernelInterpolation(xs, ys, recorder)
    interp(0.5)
    assert seen
    assert min(seen) >= 0.0
