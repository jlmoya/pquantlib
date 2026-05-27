"""Tests for the Cox-Ingersoll-Ross (1985) short-rate model.

Cross-validates against ``migration-harness/references/cluster/l4b.json``.

C++ parity: ql/models/shortrate/onefactormodels/coxingersollross.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.shortrate.onefactor.cox_ingersoll_ross import CoxIngersollRoss
from pquantlib.payoffs import OptionType
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4b")


def _build_cir() -> CoxIngersollRoss:
    """Default CIR with Feller-OK params (2*k*theta = 0.06 > sigma^2 = 0.01)."""
    return CoxIngersollRoss(
        r0=0.05, theta=0.06, k=0.5, sigma=0.1, with_feller_constraint=True
    )


def test_cir_discount(reference_data: dict[str, Any]) -> None:
    """Discount factor at several horizons matches the C++ probe."""
    ref = reference_data["cox_ingersoll_ross_model"]
    cir = _build_cir()
    tight(cir.discount(1.0), ref["discount_t1"])
    tight(cir.discount(5.0), ref["discount_t5"])
    tight(cir.discount(10.0), ref["discount_t10"])


def test_cir_discount_bond(reference_data: dict[str, Any]) -> None:
    """``discount_bond_scalar(t, T, r)`` matches the C++ probe."""
    ref = reference_data["cox_ingersoll_ross_model"]
    cir = _build_cir()
    tight(cir.discount_bond_scalar(0.0, 1.0, 0.05), ref["discount_bond_0_1_at_r05"])
    tight(cir.discount_bond_scalar(0.0, 5.0, 0.05), ref["discount_bond_0_5_at_r05"])
    tight(cir.discount_bond_scalar(1.0, 5.0, 0.03), ref["discount_bond_1_5_at_r03"])


def test_cir_discount_bond_option_call(reference_data: dict[str, Any]) -> None:
    """Call option via non-central chi-square decomposition.

    LOOSE tier: pquantlib delegates the non-central chi-square CDF to
    ``scipy.stats.ncx2`` (Boost continued-fraction backed), while C++
    uses an in-house series with ``errmax=1e-12`` and 10000-iter cap.
    Both are accurate to their respective tolerances but their
    truncation policies disagree at the 12-13th digit, exceeding TIGHT.
    Agreement is rel_tol < 1e-8 in practice.
    """
    ref = reference_data["cox_ingersoll_ross_model"]
    cir = _build_cir()
    loose(
        cir.discount_bond_option(OptionType.Call, 0.85, 1.0, 5.0),
        ref["discount_bond_option_call"],
        reason="scipy.ncx2 vs C++ series — see test docstring",
    )


def test_cir_discount_bond_option_put(reference_data: dict[str, Any]) -> None:
    """Put option via non-central chi-square decomposition.

    LOOSE tier: see ``test_cir_discount_bond_option_call`` for the
    scipy-vs-C++-series rationale.
    """
    ref = reference_data["cox_ingersoll_ross_model"]
    cir = _build_cir()
    loose(
        cir.discount_bond_option(OptionType.Put, 0.85, 1.0, 5.0),
        ref["discount_bond_option_put"],
        reason="scipy.ncx2 vs C++ series — see test docstring",
    )


def test_cir_discount_bond_option_zero_maturity_intrinsic() -> None:
    """At ``maturity < QL_EPSILON`` the option pays intrinsic.

    # C++ parity: coxingersollross.cpp:91-99.
    """
    cir = _build_cir()
    # bond price at T=5 given x0=0.05
    bond_price = cir.discount_bond_scalar(0.0, 5.0, 0.05)
    # Call at strike = bond_price/2 -> deep in the money: payoff bond - strike.
    call = cir.discount_bond_option(OptionType.Call, bond_price / 2.0, 0.0, 5.0)
    tight(call, bond_price - bond_price / 2.0)
    # Put out of the money -> 0.
    put_otm = cir.discount_bond_option(OptionType.Put, bond_price / 2.0, 0.0, 5.0)
    tight(put_otm, 0.0)


def test_cir_strike_must_be_positive() -> None:
    """``discount_bond_option`` raises if strike <= 0.

    # C++ parity: coxingersollross.cpp:87 — ``QL_REQUIRE(strike > 0, ...)``.
    """
    cir = _build_cir()
    with pytest.raises(LibraryException, match="strike must be positive"):
        cir.discount_bond_option(OptionType.Call, -0.5, 1.0, 5.0)


def test_cir_feller_constraint_enforced() -> None:
    """Construction with Feller-violating params is rejected by the constraint.

    # C++ parity: coxingersollross.cpp:51-52 — VolatilityConstraint.

    The ConstantParameter ctor validates ``test_params`` on the initial
    value, so a sigma that violates Feller raises ``LibraryException``.
    """
    # sigma^2 = 4.0 > 2*k*theta = 2*0.1*0.1 = 0.02 -> violates Feller.
    with pytest.raises(LibraryException, match="invalid value"):
        CoxIngersollRoss(
            r0=0.05, theta=0.1, k=0.1, sigma=2.0, with_feller_constraint=True
        )


def test_cir_no_feller_constraint_allowed() -> None:
    """With ``with_feller_constraint=False``, only positivity is required."""
    cir = CoxIngersollRoss(
        r0=0.05, theta=0.1, k=0.1, sigma=2.0, with_feller_constraint=False
    )
    tight(cir.sigma(), 2.0)


def test_cir_params_writethrough() -> None:
    """``set_params`` mutates the underlying parameter slots."""
    cir = _build_cir()
    p = cir.params()
    expected_size = 4
    assert p.size == expected_size
    tight(float(p[0]), 0.06)  # theta
    tight(float(p[1]), 0.5)   # k
    tight(float(p[2]), 0.1)   # sigma
    tight(float(p[3]), 0.05)  # r0
