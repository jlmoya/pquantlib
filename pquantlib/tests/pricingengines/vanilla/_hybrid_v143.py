"""Shared market builders for the v1.43 hybrid-engine cross-validation tests.

Every case in ``migration-harness/references/v143/pe/hybrid.json`` carries its
whole market description in ``inputs``; these helpers rebuild it. Nothing here
is a port — it is test scaffolding shared by
``test_analytic_bsm_hull_white_engine.py``,
``test_analytic_european_vasicek_engine.py``,
``test_analytic_heston_hull_white_engine.py``,
``test_analytic_h1_hw_engine.py``,
``test_analytic_cev_engine.py`` and
``test_analytic_gjr_garch_engine.py``.

The probe pins one evaluation date for every case
(``v143_pe_hybrid/probe.cpp:1365`` — ``Settings::instance().evaluationDate() =
kToday`` with ``kToday = Date(1, March, 2025)`` at ``probe.cpp:225``), and every
maturity is a whole number of 365-day years, so with ``Actual365Fixed`` the
exercise times are exactly 1.0 / 5.0 / 10.0 / 20.0. That matters: the Hull-White
engines branch on ``a*t > 2**-13`` and the probe straddles that threshold by one
ULP of ``a``.
"""

from __future__ import annotations

from typing import Any

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.payoffs import OptionType
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# probe.cpp:225 — const Date kToday(1, March, 2025);
TODAY: Date = Date.from_ymd(1, Month.March, 2025)
DAY_COUNTER: DayCounter = Actual365Fixed()

REFERENCE_KEY = "v143/pe/hybrid"

# C++ ``Null<Real>()`` == ``QL_NULL_REAL`` == ``std::numeric_limits<float>::max()``.
# ``AnalyticHestonEngine::Integration::gaussLobatto(relTol, Null<Real>(), maxEval)``
# passes it as the ABSOLUTE tolerance, where ``min(absAccuracy, acc*relTol)``
# always selects the relative criterion. Python's ``None`` is not a substitute:
# ``Integrator.__init__`` compares ``absolute_accuracy > QL_EPSILON`` and raises
# ``TypeError`` on ``None``.
NULL_REAL: float = 3.4028234663852886e38


def flat_curve(rate: float) -> FlatForward:
    """probe.cpp — ``flatCurve``: FlatForward(kToday, r, Actual365Fixed())."""
    return FlatForward.from_rate(
        reference_date=TODAY, forward_rate=rate, day_counter=DAY_COUNTER
    )


def flat_vol(vol: float) -> BlackConstantVol:
    """probe.cpp — ``flatVol``: BlackConstantVol(kToday, NullCalendar(), v, dc)."""
    return BlackConstantVol(
        reference_date=TODAY,
        calendar=NullCalendar(),
        day_counter=DAY_COUNTER,
        volatility=vol,
    )


def term_vol() -> BlackVarianceCurve:
    """probe.cpp — ``termVol``: a curve whose short and long ends genuinely differ.

    Used only by the Vasicek cases, to prove the engine samples the vol at
    ``t == 0`` rather than at maturity.
    """
    return BlackVarianceCurve(
        reference_date=TODAY,
        dates=[TODAY + 365, TODAY + 5 * 365, TODAY + 10 * 365],
        black_vol_curve=[0.15, 0.35, 0.40],
        day_counter=DAY_COUNTER,
        force_monotone_variance=True,
    )


def option_type(name: str) -> OptionType:
    """Map the probe's ``"Call"`` / ``"Put"`` string onto the enum."""
    return OptionType.Call if name == "Call" else OptionType.Put


def maturity_date(inputs: dict[str, Any]) -> Date:
    """Exercise date from a case's ``maturityDays`` offset."""
    return TODAY + int(inputs["maturityDays"])


def bsm_process(
    inputs: dict[str, Any], *, vol_ts: BlackVolTermStructure | None = None
) -> BlackScholesMertonProcess:
    """probe.cpp — ``bsmProcess``: BlackScholesMertonProcess(spot, qTS, rTS, volTS)."""
    return BlackScholesMertonProcess(
        x0=SimpleQuote(inputs["spot"]),
        dividend_ts=flat_curve(inputs["q"]),
        risk_free_ts=flat_curve(inputs["r"]),
        black_vol_ts=vol_ts if vol_ts is not None else flat_vol(inputs["vol"]),
    )


def heston_model(inputs: dict[str, Any]) -> HestonModel:
    """probe.cpp — ``hestonModel`` / ``h1hwHestonModel``."""
    return HestonModel(
        HestonProcess(
            risk_free_rate=flat_curve(inputs["r"]),
            dividend_yield=flat_curve(inputs["q"]),
            s0=SimpleQuote(inputs["spot"]),
            v0=inputs["hestonV0"],
            kappa=inputs["hestonKappa"],
            theta=inputs["hestonTheta"],
            sigma=inputs["hestonSigma"],
            rho=inputs["hestonRho"],
        )
    )


__all__ = [
    "DAY_COUNTER",
    "NULL_REAL",
    "REFERENCE_KEY",
    "TODAY",
    "bsm_process",
    "flat_curve",
    "flat_vol",
    "heston_model",
    "maturity_date",
    "option_type",
    "term_vol",
]
