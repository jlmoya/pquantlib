"""Tests for the Hull-White extended-Vasicek short-rate model.

Cross-validates against ``migration-harness/references/cluster/l4b.json``.

C++ parity: ql/models/shortrate/onefactormodels/hullwhite.{hpp,cpp} @ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.payoffs import OptionType
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/l4b")


def _build_flat_curve() -> FlatForward:
    """Flat 5% Continuous/Annual yield curve (Actual/365 Fixed) at 15-May-2026.

    Matches the C++ probe ``HullWhite`` setup exactly.
    """
    return FlatForward(
        Date.from_ymd(15, Month.May, 2026), SimpleQuote(0.05), Actual365Fixed()
    )


def _build_hull_white() -> HullWhite:
    """Default-parameterised HullWhite over the 5% flat curve."""
    return HullWhite(_build_flat_curve(), a=0.1, sigma=0.01)


def test_hull_white_inspectors(reference_data: dict[str, Any]) -> None:
    """``a()`` and ``sigma()`` ctor round-trip."""
    ref = reference_data["hull_white"]
    hw = _build_hull_white()
    tight(hw.a(), ref["a"])
    tight(hw.sigma(), ref["sigma"])


def test_hull_white_discount_roundtrips_curve(reference_data: dict[str, Any]) -> None:
    """``discount(t)`` round-trips ``curve.discount(t)`` by construction.

    The match is at TIGHT tolerance because the model_discount uses
    the analytical fitting parameter ``phi(t)`` and the curve_discount
    uses the FlatForward exp formula; they agree to ~14 digits.
    """
    ref = reference_data["hull_white"]
    hw = _build_hull_white()
    curve = hw.term_structure
    tight(curve.discount(1.0), ref["curve_discount_1"])
    tight(curve.discount(5.0), ref["curve_discount_5"])
    tight(curve.discount(10.0), ref["curve_discount_10"])
    tight(hw.discount(1.0), ref["model_discount_1"])
    tight(hw.discount(5.0), ref["model_discount_5"])
    tight(hw.discount(10.0), ref["model_discount_10"])


def test_hull_white_discount_bond_scalar(reference_data: dict[str, Any]) -> None:
    """``discount_bond_scalar(now, T, r)`` matches the C++ probe."""
    ref = reference_data["hull_white"]
    hw = _build_hull_white()
    tight(hw.discount_bond_scalar(0.0, 1.0, 0.05), ref["discount_bond_0_1_at_r05"])
    tight(hw.discount_bond_scalar(0.0, 5.0, 0.05), ref["discount_bond_0_5_at_r05"])
    tight(hw.discount_bond_scalar(1.0, 5.0, 0.05), ref["discount_bond_1_5_at_r05"])
    tight(hw.discount_bond_scalar(1.0, 5.0, 0.03), ref["discount_bond_1_5_at_r03"])


def test_hull_white_discount_bond_option_4args(reference_data: dict[str, Any]) -> None:
    """4-arg discount-bond option (uses term_structure for forward/strike)."""
    ref = reference_data["hull_white"]
    hw = _build_hull_white()
    tight(
        hw.discount_bond_option(OptionType.Call, 0.85, 1.0, 5.0),
        ref["discount_bond_option_call"],
    )
    tight(
        hw.discount_bond_option(OptionType.Put, 0.85, 1.0, 5.0),
        ref["discount_bond_option_put"],
    )


def test_hull_white_discount_bond_option_5args(reference_data: dict[str, Any]) -> None:
    """5-arg discount-bond option (bond_start carries information)."""
    ref = reference_data["hull_white"]
    hw = _build_hull_white()
    tight(
        hw.discount_bond_option_3args(OptionType.Call, 0.85, 1.0, 1.5, 5.0),
        ref["discount_bond_option_3args_call"],
    )
    tight(
        hw.discount_bond_option_3args(OptionType.Put, 0.85, 1.0, 1.5, 5.0),
        ref["discount_bond_option_3args_put"],
    )


def test_hull_white_convexity_bias(reference_data: dict[str, Any]) -> None:
    """Static ``convexity_bias`` (Kirikos/Novak 1997 closed form)."""
    expected = reference_data["hull_white_convexity_bias"]
    actual = HullWhite.convexity_bias(99.0, 0.5, 0.75, 0.01, 0.1)
    tight(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"futures_price": -1.0, "t": 0.5, "t_horizon": 0.75, "sigma": 0.01, "a": 0.1},
        {"futures_price": 99.0, "t": -0.5, "t_horizon": 0.75, "sigma": 0.01, "a": 0.1},
        {"futures_price": 99.0, "t": 0.5, "t_horizon": 0.25, "sigma": 0.01, "a": 0.1},
        {"futures_price": 99.0, "t": 0.5, "t_horizon": 0.75, "sigma": -0.01, "a": 0.1},
        {"futures_price": 99.0, "t": 0.5, "t_horizon": 0.75, "sigma": 0.01, "a": -0.1},
    ],
)
def test_hull_white_convexity_bias_rejects_invalid(kwargs: dict[str, float]) -> None:
    """``convexity_bias`` raises on each of the five C++ preconditions."""
    with pytest.raises(LibraryException):
        HullWhite.convexity_bias(**kwargs)


def test_hull_white_generate_arguments_after_set_params() -> None:
    """``generate_arguments`` is invoked via ``set_params`` and rebuilds phi.

    Sanity check that the L4-A CalibratedModel orchestration triggers
    the L4-B model-specific argument refresh hook.
    """
    hw = _build_hull_white()
    # set_params with (a, sigma) — Hull-White nulls out b and lambda
    # so the params vector length is 2.
    p = hw.params()
    # HW exposes 2 free params (a, sigma) after nulling b and lambda.
    expected_free_params = 2
    assert p.size == expected_free_params
    # Set new (a, sigma) and confirm a()/sigma() reflect it.
    hw.set_params(np.array([0.2, 0.005], dtype=np.float64))
    tight(hw.a(), 0.2)
    tight(hw.sigma(), 0.005)
    # The phi(t) value will have changed (different a, sigma); just
    # confirm the discount-curve roundtrip still holds with the new
    # parameters.
    curve = hw.term_structure
    tight(hw.discount(1.0), curve.discount(1.0))
    tight(hw.discount(5.0), curve.discount(5.0))
