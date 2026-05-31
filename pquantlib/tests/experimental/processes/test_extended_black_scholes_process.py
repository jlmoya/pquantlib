"""Tests for ExtendedBlackScholesMertonProcess.

# C++ parity: ql/experimental/processes/extendedblackscholesprocess.hpp.

Cross-validates against ``migration-harness/references/cluster/w7a.json``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.processes.extended_black_scholes_process import (
    Discretization,
    ExtendedBlackScholesMertonProcess,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w7a")


def _build(
    evol_disc: Discretization = Discretization.Milstein,
) -> ExtendedBlackScholesMertonProcess:
    """Mirror the probe's EBSM setup (spot 100, r=0.05, q=0.02, vol=0.25)."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.January, 2024)
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)
    return ExtendedBlackScholesMertonProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol, evol_disc=evol_disc
    )


def test_ebsm_drift_diffusion_x0(reference_data: dict[str, Any]) -> None:
    """drift/diffusion/x0 at (t=0.5, x=105).

    TIGHT: closed-form against C++ probe.
    """
    p = _build()
    tight(p.x0(), float(reference_data["ebsm_x0"]))
    tight(p.diffusion_1d(0.5, 105.0), float(reference_data["ebsm_diffusion"]))
    tight(p.drift_1d(0.5, 105.0), float(reference_data["ebsm_drift"]))


def test_ebsm_evolve_euler(reference_data: dict[str, Any]) -> None:
    """Euler-scheme evolve reproduces the C++ value.

    TIGHT: deterministic arithmetic.
    """
    p = _build(Discretization.Euler)
    tight(p.evolve_1d(0.5, 105.0, 0.25, 0.5), float(reference_data["ebsm_evolve_euler"]))


def test_ebsm_evolve_milstein(reference_data: dict[str, Any]) -> None:
    """Milstein-scheme evolve reproduces the C++ value.

    TIGHT: deterministic arithmetic.
    """
    p = _build(Discretization.Milstein)
    tight(
        p.evolve_1d(0.5, 105.0, 0.25, 0.5),
        float(reference_data["ebsm_evolve_milstein"]),
    )


def test_ebsm_evolve_predictor_corrector(reference_data: dict[str, Any]) -> None:
    """Predictor-corrector evolve reproduces the C++ value.

    TIGHT: deterministic arithmetic.
    """
    p = _build(Discretization.PredictorCorrector)
    tight(
        p.evolve_1d(0.5, 105.0, 0.25, 0.5),
        float(reference_data["ebsm_evolve_predcorr"]),
    )


def test_ebsm_euler_diverges_from_plain_gbsm_euler(
    reference_data: dict[str, Any],
) -> None:
    """The extended Euler evolve genuinely differs from plain GBSM Euler.

    # C++ parity: the strike-dependent raw-Black-vol diffusion override
    # makes the extended Euler evolve differ from the forced-Euler GBSM
    # evolve — both reference values come from the same probe.
    """
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.January, 2024)
    spot = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.25)
    gbsm = GeneralizedBlackScholesProcess(
        x0=spot,
        dividend_ts=div,
        risk_free_ts=rf,
        black_vol_ts=vol,
        force_discretization=True,
    )
    tight(
        gbsm.evolve_1d(0.5, 105.0, 0.25, 0.5),
        float(reference_data["ebsm_gbsm_evolve_euler"]),
    )
    assert float(reference_data["ebsm_evolve_euler"]) != pytest.approx(
        float(reference_data["ebsm_gbsm_evolve_euler"])
    )
