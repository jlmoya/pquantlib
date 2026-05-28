"""Tests for ``LsmBasisSystem``.

# C++ parity: ql/methods/montecarlo/lsmbasissystem.{hpp,cpp} (v1.42.1).

Cross-validates the Python recurrence-based weighted-value implementation
against ``migration-harness/references/cluster/l6a.json`` (TIGHT tier
per basis-value, since the recurrence is closed-form bit-for-bit
identical to C++ when the same floating-point ops are used).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.methods.montecarlo.lsm_basis_system import (
    LsmBasisSystem,
    PolynomialType,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    return reference_reader.load("cluster/l6a")


def _assert_basis_block_matches(
    block: dict[str, Any], polynomial_type: PolynomialType
) -> None:
    order = int(block["order"])
    xs = [float(x) for x in block["xs"]]
    expected = block["values"]
    basis = LsmBasisSystem.path_basis_system(order, polynomial_type)
    assert len(basis) == order + 1
    for i, fn in enumerate(basis):
        for j, x in enumerate(xs):
            tight(fn(x), float(expected[i][j]))


def test_monomial_order_3_matches_probe(reference_data: dict[str, Any]) -> None:
    _assert_basis_block_matches(
        reference_data["lsm_monomial_order_3"], PolynomialType.Monomial
    )


def test_laguerre_order_3_matches_probe(reference_data: dict[str, Any]) -> None:
    _assert_basis_block_matches(
        reference_data["lsm_laguerre_order_3"], PolynomialType.Laguerre
    )


def test_hermite_order_3_matches_probe(reference_data: dict[str, Any]) -> None:
    _assert_basis_block_matches(
        reference_data["lsm_hermite_order_3"], PolynomialType.Hermite
    )


def test_chebyshev2nd_order_3_matches_probe(reference_data: dict[str, Any]) -> None:
    _assert_basis_block_matches(
        reference_data["lsm_chebyshev2nd_order_3"], PolynomialType.Chebyshev2nd
    )


def test_monomial_zero_order_is_constant_one() -> None:
    basis = LsmBasisSystem.path_basis_system(0, PolynomialType.Monomial)
    assert len(basis) == 1
    assert basis[0](0.5) == 1.0
    assert basis[0](5.0) == 1.0


def test_monomial_higher_orders_are_powers() -> None:
    basis = LsmBasisSystem.path_basis_system(4, PolynomialType.Monomial)
    x = 2.5
    for i, fn in enumerate(basis):
        tight(fn(x), x**i)


def test_hyperbolic_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Hyperbolic"):
        LsmBasisSystem.path_basis_system(2, PolynomialType.Hyperbolic)


def test_legendre_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Legendre"):
        LsmBasisSystem.path_basis_system(2, PolynomialType.Legendre)


def test_chebyshev_first_kind_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Chebyshev"):
        LsmBasisSystem.path_basis_system(2, PolynomialType.Chebyshev)


def test_multi_path_basis_size_at_order_2_dim_2() -> None:
    """Multi-path basis at (dim=2, order=2, Monomial) has C(2+2,2)=6 terms.

    # C++ parity: pathBasisSystem(2, Monomial) = [1, x, x^2] per axis,
    # so the (dim=2, order=2) bivariate basis = {1, x, y, x^2, xy, y^2}.
    """
    basis = LsmBasisSystem.multi_path_basis_system(2, 2, PolynomialType.Monomial)
    assert len(basis) == 6


def test_multi_path_basis_values_match_monomial_products() -> None:
    """The (dim=2, order=2, Monomial) basis evaluated at (1.0, 2.0) is
    a permutation of {1, 1, 2, 1, 2, 4}.
    """
    basis = LsmBasisSystem.multi_path_basis_system(2, 2, PolynomialType.Monomial)
    a = np.array([1.0, 2.0], dtype=np.float64)
    vals = sorted(fn(a) for fn in basis)
    # {1, 1*1=1, 2*1=2, 1*1*1=1, 1*1*2=2, 2*2*1=4}
    # — i.e. all degree<=2 monomials in (x=1, y=2).
    # Sorted = [1, 1, 1, 2, 2, 4]
    assert vals == [1.0, 1.0, 1.0, 2.0, 2.0, 4.0]


def test_multi_path_basis_dim_1_collapses_to_single() -> None:
    """At dim=1, multi-path basis = single-path basis."""
    multi = LsmBasisSystem.multi_path_basis_system(1, 3, PolynomialType.Monomial)
    single = LsmBasisSystem.path_basis_system(3, PolynomialType.Monomial)
    x = 1.5
    a = np.array([x], dtype=np.float64)
    multi_vals = sorted(fn(a) for fn in multi)
    single_vals = sorted(fn(x) for fn in single)
    assert multi_vals == single_vals


def test_multi_path_dim_zero_raises() -> None:
    with pytest.raises(Exception, match="zero dimension"):
        LsmBasisSystem.multi_path_basis_system(0, 2, PolynomialType.Monomial)


def test_path_basis_negative_order_raises() -> None:
    with pytest.raises(Exception, match="order must be nonneg"):
        LsmBasisSystem.path_basis_system(-1, PolynomialType.Monomial)
