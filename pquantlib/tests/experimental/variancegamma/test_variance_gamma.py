"""Tests for the Variance-Gamma process + model + analytic engine.

# C++ parity: ql/experimental/variancegamma/{variancegammaprocess,
# variancegammamodel,analyticvariancegammaengine}.hpp.

Cross-validates against ``migration-harness/references/cluster/w7a.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.variancegamma.variance_gamma_model import (
    VarianceGammaModel,
)
from pquantlib.experimental.variancegamma.variance_gamma_process import (
    VarianceGammaProcess,
)
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_variance_gamma_engine import (
    VarianceGammaEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, loose, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w7a")


def _build_process() -> VarianceGammaProcess:
    """Canonical variancegamma.cpp case 0 (spot 6000, r=0.05, q=0, VG params)."""
    dc = Actual360()
    ref = Date.from_ymd(15, Month.January, 2024)
    spot = SimpleQuote(6000.0)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.00, day_counter=dc)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    return VarianceGammaProcess(spot, div, rf, 0.20, 0.05, -0.50)


# --- process -----------------------------------------------------------------


def test_vg_process_accessors(reference_data: dict[str, Any]) -> None:
    """x0 + sigma/nu/theta round-trips vs C++ probe.

    EXACT/TIGHT: scalar inspectors.
    """
    p = _build_process()
    exact(p.x0(), float(reference_data["vg_x0"]))
    tight(p.sigma(), float(reference_data["vg_sigma"]))
    tight(p.nu(), float(reference_data["vg_nu"]))
    tight(p.theta(), float(reference_data["vg_theta"]))


def test_vg_process_drift_diffusion_not_implemented() -> None:
    """drift/diffusion raise (parity with C++ QL_FAIL)."""
    p = _build_process()
    with pytest.raises(LibraryException):
        p.drift_1d(0.5, 6000.0)
    with pytest.raises(LibraryException):
        p.diffusion_1d(0.5, 6000.0)


# --- model -------------------------------------------------------------------


def test_vg_model_params_match_process() -> None:
    """Model arguments reflect the process params.

    TIGHT: ConstantParameter round-trip.
    """
    p = _build_process()
    model = VarianceGammaModel(p)
    tight(model.sigma(), 0.20)
    tight(model.nu(), 0.05)
    tight(model.theta(), -0.50)


def test_vg_model_generate_arguments_rebuilds_process() -> None:
    """set_params triggers generate_arguments rebuilding the process.

    TIGHT: after pushing new params, the wrapped process reflects them.
    """
    p = _build_process()
    model = VarianceGammaModel(p)
    model.set_params(np.array([0.25, 0.10, -0.30], dtype=np.float64))
    tight(model.sigma(), 0.25)
    tight(model.nu(), 0.10)
    tight(model.theta(), -0.30)
    tight(model.process().sigma(), 0.25)
    tight(model.process().nu(), 0.10)
    tight(model.process().theta(), -0.30)


def test_vg_model_constraints_reject_negative_sigma_nu() -> None:
    """Positive constraints on sigma/nu fail the model constraint test."""
    p = _build_process()
    model = VarianceGammaModel(p)
    # sigma < 0 violates PositiveConstraint
    assert not model.constraint.test(np.array([-0.1, 0.05, -0.5], dtype=np.float64))
    # nu < 0 violates PositiveConstraint
    assert not model.constraint.test(np.array([0.2, -0.05, -0.5], dtype=np.float64))
    # theta < 0 is fine (NoConstraint)
    assert model.constraint.test(np.array([0.2, 0.05, -0.5], dtype=np.float64))


# --- analytic engine ---------------------------------------------------------


def _price(strike: float, option_type: OptionType) -> float:
    p = _build_process()
    ref = Date.from_ymd(15, Month.January, 2024)
    ex_date = ref + 360  # t = 1.0 under Actual/360
    exercise = EuropeanExercise(ex_date)
    payoff = PlainVanillaPayoff(option_type, strike)
    opt = EuropeanOption(payoff, exercise)
    opt.set_pricing_engine(VarianceGammaEngine(p))
    return opt.npv()


def test_vg_analytic_call_5550(reference_data: dict[str, Any]) -> None:
    """ATM-ish call NPV vs C++ analytic engine.

    LOOSE: special-function quadrature (scipy.quad vs Gauss-Kronrod/Lobatto).
    """
    loose(_price(5550.0, OptionType.Call), float(reference_data["vg_analytic_call_5550"]))


def test_vg_analytic_call_6000(reference_data: dict[str, Any]) -> None:
    """ATM call NPV vs C++ analytic engine.

    LOOSE: special-function quadrature.
    """
    loose(_price(6000.0, OptionType.Call), float(reference_data["vg_analytic_call_6000"]))


def test_vg_analytic_call_6500(reference_data: dict[str, Any]) -> None:
    """OTM call NPV vs C++ analytic engine.

    LOOSE: special-function quadrature.
    """
    loose(_price(6500.0, OptionType.Call), float(reference_data["vg_analytic_call_6500"]))


def test_vg_analytic_put_5550(reference_data: dict[str, Any]) -> None:
    """ITM put NPV vs C++ analytic engine.

    LOOSE: special-function quadrature.
    """
    loose(_price(5550.0, OptionType.Put), float(reference_data["vg_analytic_put_5550"]))
