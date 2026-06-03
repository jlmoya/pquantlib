"""Shared builders: reference date, yield curves, and stochastic processes.

These are the primitives every other module composes. Keeping them in one
place means the whole showcase prices off a consistent market setup.
"""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date

# A single Actual/365 Fixed day counter is fine for the whole showcase: it is
# the QuantLib convention for option time-to-expiry and keeps year fractions
# clean (T = days / 365).
DAY_COUNTER = Actual365Fixed()


def reference_date() -> Date:
    """The global 'as of' date. Live (today) so the demo always feels current."""
    return Date.todays_date()


def pin_evaluation_date(ref: Date) -> None:
    """Pin the library-wide evaluation date (needed for swaps / bootstrapping)."""
    ObservableSettings().evaluation_date = ref


def expiry_from_years(ref: Date, t_years: float) -> Date:
    """Map a maturity in years to a concrete date under Actual/365 Fixed."""
    return ref + max(1, round(t_years * 365.0))


def flat_curve(rate: float, ref: Date | None = None) -> FlatForward:
    """A flat continuously-compounded yield curve at ``rate``."""
    return FlatForward.from_rate(
        reference_date=ref or reference_date(), forward_rate=rate, day_counter=DAY_COUNTER
    )


def bsm_process(
    spot: float, r: float, q: float, vol: float, ref: Date | None = None
) -> GeneralizedBlackScholesProcess:
    """The textbook Black-Scholes-Merton process (flat r, q, sigma)."""
    ref = ref or reference_date()
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(spot),
        dividend_ts=flat_curve(q, ref),
        risk_free_ts=flat_curve(r, ref),
        black_vol_ts=BlackConstantVol(
            reference_date=ref, calendar=NullCalendar(), day_counter=DAY_COUNTER, volatility=vol
        ),
    )


def heston_process(
    spot: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    ref: Date | None = None,
) -> HestonProcess:
    """The Heston stochastic-volatility process."""
    ref = ref or reference_date()
    return HestonProcess(
        risk_free_rate=flat_curve(r, ref),
        dividend_yield=flat_curve(q, ref),
        s0=SimpleQuote(spot),
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
    )
