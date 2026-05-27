"""Tests for the Extended Cox-Ingersoll-Ross (curve-fitted) model.

Cross-validates against ``migration-harness/references/cluster/l4b.json``.

C++ parity: ql/models/shortrate/onefactormodels/extendedcoxingersollross.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.shortrate.onefactor.extended_cox_ingersoll_ross import (
    ExtendedCoxIngersollRoss,
)
from pquantlib.payoffs import OptionType
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4b")


def _build_flat_curve() -> FlatForward:
    """Flat 5% Continuous/Annual yield curve at 15-May-2026."""
    return FlatForward(
        Date.from_ymd(15, Month.May, 2026), SimpleQuote(0.05), Actual365Fixed()
    )


def _build_ecir() -> ExtendedCoxIngersollRoss:
    """Default-parameterised ECIR with Feller-OK params."""
    return ExtendedCoxIngersollRoss(
        _build_flat_curve(), theta=0.06, k=0.5, sigma=0.1, x0=0.05,
        with_feller_constraint=True,
    )


def test_ecir_discount_roundtrips_curve(reference_data: dict[str, Any]) -> None:
    """``discount(t)`` matches the input curve discount (curve-fitted by construction)."""
    ref = reference_data["extended_cox_ingersoll_ross"]
    ecir = _build_ecir()
    tight(ecir.discount(1.0), ref["model_discount_1"])
    tight(ecir.discount(5.0), ref["model_discount_5"])
    tight(ecir.discount(10.0), ref["model_discount_10"])


def test_ecir_discount_bond(reference_data: dict[str, Any]) -> None:
    """``discount_bond_scalar`` matches the C++ probe."""
    ref = reference_data["extended_cox_ingersoll_ross"]
    ecir = _build_ecir()
    tight(ecir.discount_bond_scalar(0.0, 1.0, 0.05), ref["discount_bond_0_1_at_r05"])
    tight(ecir.discount_bond_scalar(0.0, 5.0, 0.05), ref["discount_bond_0_5_at_r05"])
    tight(ecir.discount_bond_scalar(1.0, 5.0, 0.03), ref["discount_bond_1_5_at_r03"])


def test_ecir_discount_bond_option_call(reference_data: dict[str, Any]) -> None:
    """Call option via shifted non-central chi-square decomposition.

    LOOSE tier: scipy.ncx2 vs C++ series — see L4-B CIR test docstring.
    """
    ref = reference_data["extended_cox_ingersoll_ross"]
    ecir = _build_ecir()
    loose(
        ecir.discount_bond_option(OptionType.Call, 0.85, 1.0, 5.0),
        ref["discount_bond_option_call"],
        reason="scipy.ncx2 vs C++ series",
    )


def test_ecir_discount_bond_option_put(reference_data: dict[str, Any]) -> None:
    """Put option via shifted non-central chi-square decomposition.

    LOOSE tier: scipy.ncx2 vs C++ series — see L4-B CIR test docstring.
    """
    ref = reference_data["extended_cox_ingersoll_ross"]
    ecir = _build_ecir()
    loose(
        ecir.discount_bond_option(OptionType.Put, 0.85, 1.0, 5.0),
        ref["discount_bond_option_put"],
        reason="scipy.ncx2 vs C++ series",
    )
