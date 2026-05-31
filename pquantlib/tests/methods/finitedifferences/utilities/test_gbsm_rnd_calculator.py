"""Tests for GBSMRNDCalculator.

Cross-validates against ``migration-harness/references/cluster/w8d.json``.

C++ parity: ql/methods/finitedifferences/utilities/gbsmrndcalculator.{hpp,cpp}
@ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.finitedifferences.utilities.gbsm_rnd_calculator import (
    GBSMRNDCalculator,
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
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8d")


def _calc() -> GBSMRNDCalculator:
    refd = Date.from_ymd(15, Month.January, 2024)
    dc = Actual365Fixed()
    bs = GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=FlatForward(refd, SimpleQuote(0.02), dc),
        risk_free_ts=FlatForward(refd, SimpleQuote(0.05), dc),
        black_vol_ts=BlackConstantVol(
            reference_date=refd, calendar=TARGET(), day_counter=dc, volatility=0.25
        ),
    )
    return GBSMRNDCalculator(bs)


def test_cdf(ref: dict[str, Any]) -> None:
    c = _calc()
    tight(c.cdf(100.0, 1.0), ref["rnd_cdf_t1_k100"])
    tight(c.cdf(120.0, 1.0), ref["rnd_cdf_t1_k120"])


def test_pdf(ref: dict[str, Any]) -> None:
    # central-difference pdf -> LOOSE (finite-difference noise at 1e-3 step)
    loose(_calc().pdf(100.0, 1.0), ref["rnd_pdf_t1_k100"])


def test_invcdf(ref: dict[str, Any]) -> None:
    c = _calc()
    tight(c.invcdf(0.5, 1.0), ref["rnd_invcdf_t1_q05"])
    tight(c.invcdf(0.9, 1.0), ref["rnd_invcdf_t1_q09"])


def test_cdf_invcdf_roundtrip() -> None:
    """invcdf(cdf(k)) == k to solver tolerance."""
    c = _calc()
    k = 110.0
    q = c.cdf(k, 1.0)
    loose(c.invcdf(q, 1.0), k)
