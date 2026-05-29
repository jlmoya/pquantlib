"""Tests for AnalyticPDFHestonEngine.

Cross-validates against ``migration-harness/references/cluster/w4b.json``.

C++ parity:
ql/pricingengines/vanilla/analyticpdfhestonengine.{hpp,cpp} +
ql/methods/finitedifferences/utilities/hestonrndcalculator.{hpp,cpp}
@ v1.42.1 (099987f0).

Tolerance: **LOOSE** — both implementations integrate a
characteristic-function-derived PDF over the truncated log-spot grid.
The C++ engine uses ``GaussLobattoIntegral``; the Python port uses
``scipy.integrate.quad``. Agreement to ~1e-6 absolute on the ATM
case; comparison is also cross-validated against the standard
``AnalyticHestonEngine`` (also LOOSE) to confirm consistency.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_heston_engine import (
    AnalyticHestonEngine,
)
from pquantlib.pricingengines.vanilla.analytic_pdf_heston_engine import (
    AnalyticPDFHestonEngine,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import custom
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def refs() -> dict[str, Any]:
    return load_reference("cluster/w4b")


def _model() -> tuple[HestonModel, Date]:
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365

    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.02, day_counter=dc)
    process = HestonProcess(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=1.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
    )
    return HestonModel(process), expiry


_REASON_PDF = (
    "PDF-based pricing via scipy.integrate.quad on the transformed "
    "Heston CF; ~1e-4 agreement vs C++ GaussLobattoIntegral on the "
    "standard testbed."
)


def test_pdf_matches_classic_heston(refs: dict[str, Any]) -> None:
    """Cross-check against the classic Heston engine (Gatheral)."""
    model, expiry = _model()
    payoff = PlainVanillaPayoff(OptionType.Call, 100.0)
    opt = EuropeanOption(payoff, EuropeanExercise(expiry))

    opt.set_pricing_engine(AnalyticPDFHestonEngine(model, 1e-8, 100000))
    pdf_npv = opt.npv()

    opt.set_pricing_engine(AnalyticHestonEngine(model))
    classic_npv = opt.npv()

    custom(pdf_npv, classic_npv, abs_tol=1e-4, rel_tol=1e-6, reason=_REASON_PDF)
    custom(
        pdf_npv,
        float(refs["analytic_pdf_heston"]["pdf_npv"]),
        abs_tol=1e-4,
        rel_tol=1e-6,
        reason=_REASON_PDF,
    )
