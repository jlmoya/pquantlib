"""Tests for VarianceOption + IntegralHestonVarianceOptionEngine.

# C++ parity:
# ql/experimental/varianceoption/varianceoption.{hpp,cpp} +
# ql/experimental/varianceoption/integralhestonvarianceoptionengine.{hpp,cpp}
# @ v1.42.1.

Cross-validates against the ``variance_option_*`` keys of
``migration-harness/references/cluster/w4c.json``.

Test setup:
* Heston params: v0=0.04, kappa=4.0, theta=0.04, sigma_v=0.25,
  rho=-0.5 (Feller-respecting: 2*4*0.04 = 0.32 > 0.0625).
* r=4%, q=0%, spot=100.
* Tenor: T = 182 days ~ 0.5y, Actual/365Fixed.
* Two strikes (0.04, 0.05) on a notional of 10000.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.varianceoption.variance_option import (
    VarianceOption,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.integral_heston_variance_option_engine import (
    IntegralHestonVarianceOptionEngine,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import loose
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w4c")


@pytest.fixture
def today() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


@pytest.fixture
def heston_process(today: Date) -> HestonProcess:
    """Heston process fixture mirroring the W4-C probe Heston setup.

    Note: the C++ engine asserts ``dividendYield().empty()``; the
    Python port doesn't enforce that, so we pass a flat-zero
    ``FlatForward`` as a dividend curve to get the same effect.
    """
    dc = Actual365Fixed()
    return HestonProcess(
        risk_free_rate=FlatForward.from_rate(today, 0.04, dc),
        dividend_yield=FlatForward.from_rate(today, 0.0, dc),
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=4.0,
        theta=0.04,
        sigma=0.25,
        rho=-0.5,
    )


def test_variance_option_heston_call_strike004(
    today: Date,
    heston_process: HestonProcess,
    reference_data: dict[str, Any],
) -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 0.04)
    vopt = VarianceOption(
        payoff=payoff,
        notional=10000.0,
        start_date=today,
        maturity_date=today + 182,
    )
    vopt.set_pricing_engine(
        IntegralHestonVarianceOptionEngine(heston_process)
    )
    loose(opt_npv := vopt.npv(), reference_data["variance_option_heston_call_npv"])
    # Sanity: oscillatory integral can produce small negative values
    # near at-the-money strikes (this is documented engine behaviour).
    assert opt_npv < 0  # confirms parity sign with C++ probe output


def test_variance_option_heston_call_strike005(
    today: Date,
    heston_process: HestonProcess,
    reference_data: dict[str, Any],
) -> None:
    payoff = PlainVanillaPayoff(OptionType.Call, 0.05)
    vopt = VarianceOption(
        payoff=payoff,
        notional=10000.0,
        start_date=today,
        maturity_date=today + 182,
    )
    vopt.set_pricing_engine(
        IntegralHestonVarianceOptionEngine(heston_process)
    )
    loose(
        vopt.npv(),
        reference_data["variance_option_heston_call_npv_strike005"],
    )
