"""CreditRisk+ and sensitivity analysis (W8-B batch c) — vs C++ probe.

Probe source: migration-harness/cpp/probes/cluster_w8b/probe_creditriskplus.cpp
Reference:    migration-harness/references/cluster/w8b.json

CreditRisk+ reproduces the canonical reference example from
[1] Integrating Correlations, Risk, July 1999 (two sectors of 1000
obligors; overall EL 80, UL ~53.1, 99% loss quantile ~250).

NOTE: ``CreditRiskPlus`` + ``sensitivityanalysis`` were removed from
QuantLib in v1.36; both are ported from the recovered pre-removal source
per the W8-B brief (see pquantlib.experimental.risk package note).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.experimental.risk.credit_risk_plus import CreditRiskPlus
from pquantlib.experimental.risk.sensitivity_analysis import (
    SensitivityAnalysis,
    aggregate_npv,
    bucket_analysis,
    parallel_analysis,
)
from pquantlib.instruments.instrument import Instrument, InstrumentResults
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w8b")


def _credit_risk_plus() -> CreditRiskPlus:
    """The Risk-1999 two-sector reference setup (matches the probe)."""
    exposure = [1.0] * 1000 + [2.0] * 1000
    pd = [0.04] * 1000 + [0.02] * 1000
    sector = [0] * 1000 + [1] * 1000
    relative_default_variance = [0.75 * 0.75, 0.75 * 0.75]
    rho = np.array([[1.0, 0.50], [0.50, 1.0]], dtype=np.float64)
    return CreditRiskPlus(exposure, pd, sector, relative_default_variance, rho, 0.1)


# ----------------------------------------------------------------------
# CreditRisk+
# ----------------------------------------------------------------------


def test_credit_risk_plus_sector_figures(cpp_ref: dict[str, Any]) -> None:
    cr = _credit_risk_plus()
    tolerance.loose(cr.sector_exposures()[0], float(cpp_ref["cr_sector0_exposure"]))
    tolerance.loose(cr.sector_exposures()[1], float(cpp_ref["cr_sector1_exposure"]))
    tolerance.loose(cr.sector_expected_loss()[0], float(cpp_ref["cr_sector0_el"]))
    tolerance.loose(cr.sector_expected_loss()[1], float(cpp_ref["cr_sector1_el"]))
    tolerance.loose(cr.sector_unexpected_loss()[0], float(cpp_ref["cr_sector0_ul"]))
    tolerance.loose(cr.sector_unexpected_loss()[1], float(cpp_ref["cr_sector1_ul"]))


def test_credit_risk_plus_portfolio_figures(cpp_ref: dict[str, Any]) -> None:
    cr = _credit_risk_plus()
    tolerance.loose(cr.exposure(), float(cpp_ref["cr_exposure"]))
    tolerance.loose(cr.expected_loss(), float(cpp_ref["cr_expected_loss"]))
    tolerance.loose(cr.unexpected_loss(), float(cpp_ref["cr_unexpected_loss"]))
    tolerance.loose(
        cr.relative_default_variance(), float(cpp_ref["cr_relative_default_variance"])
    )


def test_credit_risk_plus_loss_distribution(cpp_ref: dict[str, Any]) -> None:
    cr = _credit_risk_plus()
    tolerance.loose(cr.loss()[0], float(cpp_ref["cr_loss_0"]))
    tolerance.loose(cr.loss_quantile(0.99), float(cpp_ref["cr_loss_quantile_99"]))
    tolerance.loose(cr.loss_quantile(0.999), float(cpp_ref["cr_loss_quantile_999"]))


def test_credit_risk_plus_reference_values_paper() -> None:
    """The paper's rounded reference figures (independent of the probe)."""
    cr = _credit_risk_plus()
    # EL 80, UL ~53.1, sector ULs ~30.7 / ~31.3, 99% quantile ~250.
    tolerance.custom(cr.expected_loss(), 80.0, abs_tol=1e-8, rel_tol=1e-8, reason="paper Table A EL")
    tolerance.custom(cr.unexpected_loss(), 53.1, abs_tol=0.01, rel_tol=1e-3, reason="paper Table A UL")
    tolerance.custom(cr.loss_quantile(0.99), 250.0, abs_tol=0.5, rel_tol=1e-2, reason="paper Fig 1 99%ile")


# ----------------------------------------------------------------------
# Sensitivity analysis
# ----------------------------------------------------------------------


class _QuadraticInstrument(Instrument):
    """Test double: NPV = coefficient * quote^2 (delta = 2*c*q, gamma = 2*c).

    Used to verify the finite-difference derivatives against closed forms.
    """

    def __init__(self, quote: SimpleQuote, coefficient: float = 1.0) -> None:
        super().__init__()
        self._quote: SimpleQuote = quote
        self._coefficient: float = coefficient
        quote.register_with(self)

    def is_expired(self) -> bool:
        return False

    def _perform_calculations(self) -> None:
        results = InstrumentResults()
        results.value = self._coefficient * self._quote.value() ** 2
        self.fetch_results(results)


def test_aggregate_npv() -> None:
    q = SimpleQuote(3.0)
    a = _QuadraticInstrument(q, 1.0)  # NPV = 9
    b = _QuadraticInstrument(q, 2.0)  # NPV = 18
    tolerance.loose(aggregate_npv([a, b], []), 27.0)
    tolerance.loose(aggregate_npv([a, b], [1.0, 0.5]), 9.0 + 0.5 * 18.0)


def test_bucket_analysis_centered_quadratic() -> None:
    """Centered differencing of c*q^2 recovers delta=2cq, gamma=2c."""
    q = SimpleQuote(5.0)
    inst = _QuadraticInstrument(q, 2.0)  # NPV = 2*q^2; d/dq = 4q, d2/dq2 = 4
    delta, gamma = bucket_analysis(q, [inst], [], shift=1e-4)
    tolerance.custom(delta, 4.0 * 5.0, abs_tol=1e-4, rel_tol=1e-5, reason="central FD delta vs 4q")
    assert gamma is not None
    tolerance.custom(gamma, 4.0, abs_tol=1e-2, rel_tol=1e-3, reason="central FD gamma vs 4")
    # The quote is restored to its original value after the analysis.
    tolerance.exact(q.value(), 5.0)


def test_bucket_analysis_one_side() -> None:
    """One-sided differencing returns gamma=None."""
    q = SimpleQuote(5.0)
    inst = _QuadraticInstrument(q, 1.0)
    delta, gamma = bucket_analysis(
        q, [inst], [], shift=1e-6, sensitivity_type=SensitivityAnalysis.OneSide
    )
    # Forward difference of q^2 at 5 with h=1e-6 ~ 2q + h ~ 10.000001.
    tolerance.custom(delta, 10.0, abs_tol=1e-3, rel_tol=1e-4, reason="forward FD delta ~ 2q")
    assert gamma is None


def test_parallel_analysis_centered() -> None:
    """Parallel shift of two quotes driving two instruments."""
    q1 = SimpleQuote(4.0)
    q2 = SimpleQuote(4.0)
    # Two instruments each on one quote; both shifted together.
    a = _QuadraticInstrument(q1, 1.0)
    b = _QuadraticInstrument(q2, 1.0)
    delta, gamma = parallel_analysis([q1, q2], [a, b], [], shift=1e-4)
    # NPV(s) = (q1+s)^2 + (q2+s)^2; d/ds at s=0 = 2q1 + 2q2 = 16; d2/ds2 = 4.
    tolerance.custom(delta, 16.0, abs_tol=1e-3, rel_tol=1e-4, reason="parallel central FD delta")
    assert gamma is not None
    tolerance.custom(gamma, 4.0, abs_tol=1e-2, rel_tol=1e-3, reason="parallel central FD gamma")
    tolerance.exact(q1.value(), 4.0)
    tolerance.exact(q2.value(), 4.0)


def test_parallel_analysis_empty_instruments() -> None:
    """Empty instrument list returns (0, 0) without touching quotes."""
    q = SimpleQuote(1.0)
    delta, gamma = parallel_analysis([q], [], [])
    tolerance.exact(delta, 0.0)
    assert gamma == 0.0
