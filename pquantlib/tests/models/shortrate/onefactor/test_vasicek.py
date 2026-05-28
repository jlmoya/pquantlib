"""Tests for the Vasicek (1977) single-factor short-rate model.

Cross-validates against ``migration-harness/references/cluster/l4b.json``.

C++ parity: ql/models/shortrate/onefactormodels/vasicek.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.models.shortrate.onefactor.vasicek import Vasicek
from pquantlib.models.shortrate.short_rate_tree import ShortRateTree
from pquantlib.payoffs import OptionType
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.time_grid import TimeGrid


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4b")


def _build_vasicek() -> Vasicek:
    """Default-parameterised Vasicek (r0=0.05, a=0.1, b=0.05, sigma=0.01)."""
    return Vasicek(r0=0.05, a=0.1, b=0.05, sigma=0.01, lambda_=0.0)


def test_vasicek_inspectors(reference_data: dict[str, Any]) -> None:
    """Ctor params round-trip via the inspector accessors."""
    ref = reference_data["vasicek"]
    v = _build_vasicek()
    tight(v.r0(), ref["r0"])
    tight(v.a(), ref["a"])
    tight(v.b(), ref["b"])
    tight(v.sigma(), ref["sigma"])
    tight(v.lambda_(), ref["lambda"])


def test_vasicek_discount(reference_data: dict[str, Any]) -> None:
    """``discount(t)`` matches the C++ implied curve at multiple horizons."""
    ref = reference_data["vasicek"]
    v = _build_vasicek()
    tight(v.discount(1.0), ref["discount_t1"])
    tight(v.discount(5.0), ref["discount_t5"])
    tight(v.discount(10.0), ref["discount_t10"])


def test_vasicek_discount_bond_at_r0(reference_data: dict[str, Any]) -> None:
    """``discount_bond(0, T, r0)`` matches ``discount(T)`` (sanity)."""
    ref = reference_data["vasicek"]
    v = _build_vasicek()
    tight(v.discount_bond_scalar(0.0, 1.0, 0.05), ref["discount_bond_0_1"])
    tight(v.discount_bond_scalar(0.0, 5.0, 0.05), ref["discount_bond_0_5"])
    tight(v.discount_bond_scalar(0.0, 10.0, 0.05), ref["discount_bond_0_10"])


def test_vasicek_discount_bond_at_other_rates(reference_data: dict[str, Any]) -> None:
    """``discount_bond(t, T, r)`` for r != r0 — exercises B(t, T) sensitivity."""
    ref = reference_data["vasicek"]
    v = _build_vasicek()
    tight(v.discount_bond_scalar(1.0, 5.0, 0.03), ref["discount_bond_1_5_at_r03"])
    tight(v.discount_bond_scalar(1.0, 5.0, 0.07), ref["discount_bond_1_5_at_r07"])


def test_vasicek_discount_bond_array_dispatch(reference_data: dict[str, Any]) -> None:
    """Vector-factor ``discount_bond`` dispatches into the scalar via factors[0]."""
    ref = reference_data["vasicek"]
    v = _build_vasicek()
    factors = np.array([0.05], dtype=np.float64)
    tight(v.discount_bond(0.0, 5.0, factors), ref["discount_bond_0_5"])


def test_vasicek_discount_bond_option_call(reference_data: dict[str, Any]) -> None:
    """Call option on a discount bond via the Black formula."""
    ref = reference_data["vasicek"]
    v = _build_vasicek()
    actual = v.discount_bond_option(OptionType.Call, 0.85, 1.0, 5.0)
    tight(actual, ref["discount_bond_option_call"])


def test_vasicek_discount_bond_option_put(reference_data: dict[str, Any]) -> None:
    """Put option on a discount bond via the Black formula."""
    ref = reference_data["vasicek"]
    v = _build_vasicek()
    actual = v.discount_bond_option(OptionType.Put, 0.85, 1.0, 5.0)
    tight(actual, ref["discount_bond_option_put"])


def test_vasicek_tree_returns_short_rate_tree() -> None:
    """``tree()`` builds a recombining trinomial lattice (L5-B).

    The L4-B carve-out is closed by porting TrinomialTree /
    TreeLattice1D / ShortRateTree in L5-B.
    """
    v = _build_vasicek()
    grid = TimeGrid.regular(end=1.0, steps=10)
    tree = v.tree(grid)
    assert isinstance(tree, ShortRateTree)
    # Time-grid identity carried through.
    assert tree.time_grid() is grid


def test_vasicek_params_writethrough() -> None:
    """``set_params`` mutates the underlying parameter slots.

    Validates the L4-A CalibratedModel.set_params orchestration on a
    real concrete model.
    """
    v = _build_vasicek()
    p = v.params()
    assert p.shape == (4,)
    tight(float(p[0]), 0.1)  # a
    tight(float(p[1]), 0.05)  # b
    tight(float(p[2]), 0.01)  # sigma
    tight(float(p[3]), 0.0)  # lambda

    new_params = np.array([0.2, 0.04, 0.02, 0.005], dtype=np.float64)
    v.set_params(new_params)
    tight(v.a(), 0.2)
    tight(v.b(), 0.04)
    tight(v.sigma(), 0.02)
    tight(v.lambda_(), 0.005)
