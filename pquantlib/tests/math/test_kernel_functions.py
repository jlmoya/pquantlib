"""Cross-validate KernelFunction / GaussianKernel against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``gaussian_kernel`` section. Nine abscissae from -6 to +6 crossed with three
(average, sigma) pairs, each pinning ``value``, ``derivative`` and
``primitive``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.kernel_functions import GaussianKernel, KernelFunction
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def test_value_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """``GaussianKernel(x)`` matches C++ to TIGHT.

    Both sides evaluate ``exp(-(x-avg)^2 / (2 sigma^2)) * normFact / (sigma
    sqrt(2 pi))``; the only difference is libm's ``exp``, which is correctly
    rounded to well inside the TIGHT relative band.
    """
    for case in cpp["gaussian_kernel"]["cases"]:
        kernel = GaussianKernel(float(case["average"]), float(case["sigma"]))
        tolerance.tight(kernel(float(case["x"])), float(case["value"]))


def test_derivative_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """``derivative`` matches C++ to TIGHT — same pdf times ``(avg - x)/sigma^2``."""
    for case in cpp["gaussian_kernel"]["cases"]:
        kernel = GaussianKernel(float(case["average"]), float(case["sigma"]))
        tolerance.tight(kernel.derivative(float(case["x"])), float(case["derivative"]))


def test_primitive_matches_cpp_tight(cpp: dict[str, Any]) -> None:
    """``primitive`` matches C++ to TIGHT.

    This is the loosest of the three (observed worst ~3e-14 relative, at
    ``x = -6, sigma = 2.05``) and the reason is structural, not sloppiness:
    ``CumulativeNormalDistribution`` computes ``0.5 * (1 + erf(z/sqrt2))``,
    and for ``z`` around -3 the ``erf`` term is about -0.9966, so the sum
    cancels three decimal digits. A 1-ULP difference between C++'s and
    CPython's ``erf`` (2.2e-16 absolute on a quantity of size 1) therefore
    lands as ~6e-14 relative on a result of size 3.4e-3. TIGHT's 1e-12 band
    absorbs that with room to spare; anything tighter would be measuring
    libm, not the port.
    """
    for case in cpp["gaussian_kernel"]["cases"]:
        kernel = GaussianKernel(float(case["average"]), float(case["sigma"]))
        tolerance.tight(kernel.primitive(float(case["x"])), float(case["primitive"]))


def test_gaussian_kernel_is_a_kernel_function() -> None:
    """C++ ``GaussianKernel : public KernelFunction``; so is the port."""
    assert isinstance(GaussianKernel(0.0, 1.0), KernelFunction)


def test_kernel_function_is_abstract() -> None:
    """``KernelFunction`` has a pure-virtual ``operator()`` in C++."""
    with pytest.raises(TypeError):
        KernelFunction()  # type: ignore[abstract]  # pyright: ignore[reportAbstractUsage]


def test_sigma_must_be_positive() -> None:
    """The underlying NormalDistribution rejects a non-positive sigma."""
    with pytest.raises(Exception, match="sigma"):
        GaussianKernel(0.0, 0.0)
