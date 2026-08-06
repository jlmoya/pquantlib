"""Shared setups for the v1.43 ``solvers/`` cross-validation.

# C++ parity: migration-harness/cpp/probes/v143_methods_solvers/probe.cpp
# @ v1.43 (submodule 6b57206e0).

Every expected value in the sibling ``test_v143_solvers_*`` modules comes
from running that probe against C++ QuantLib v1.43. This module only
rebuilds the *inputs* — curves, processes, meshers, solver descriptions —
bit-identically on the Python side, and provides the tolerance helpers.

**Tolerance tiers used by this cluster.**

``TIGHT`` (abs 1e-14 / rel 1e-12) covers the mesher locations, the maturity
inner values, one operator ``apply``, and everything the solvers read
straight off their interpolation: values and analytic first/second
derivatives. Those agree with C++ to 1e-14 relative or better even after a
100-to-400-step rollback, because each step is a handful of BLAS-level
operations whose rounding does not accumulate coherently.

Two families need a looser bound, and the reason is cancellation, not
sloppiness:

* ``theta_at`` differences the solution at ``t = 0.99/365`` against the one
  at ``t = 0`` and divides by that 2.7e-3 step. Two values of order 9 that
  differ by ~1.3e-2 lose three digits in the subtraction and another 2.6
  in the division, so a 1e-15 relative error on each end value lands at
  ~1e-12 relative on theta.
* ``FdmSimple2dBSSolver`` and ``FdmHestonHullWhiteSolver`` build their
  greeks by bumping their own ``value_at``. The gamma numerator
  ``V(s+e) + V(s-e) - 2 V(s)`` is ~4.7e-3 formed out of three values of
  order 9.9 — a 2000:1 cancellation, which turns 1e-15 into ~2e-12.

Both families therefore assert at ``rel 1e-10 / abs 1e-12``: roughly two
orders of magnitude above the derived error bound and two to three above
what is actually observed (1.3e-13 for theta, 7.6e-13 for the bumped
gamma), yet still four orders tighter than the LOOSE tier.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.basket_option import MaxBasketPayoff
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmInnerValueCalculator,
    FdmLogBasketInnerValue,
    FdmLogInnerValue,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

REFERENCE_KEY = "v143/methods/solvers"

#: The six ``FdmSchemeDesc`` values the probe sweeps, keyed by the suffix used
#: in the emitted reference keys.
SCHEMES: dict[str, FdmSchemeDesc] = {
    "douglas": FdmSchemeDesc.douglas(),
    "cn": FdmSchemeDesc.crank_nicolson(),
    "implicit": FdmSchemeDesc.implicit_euler(),
    "explicit": FdmSchemeDesc.explicit_euler(),
    "craigsneyd": FdmSchemeDesc.craig_sneyd(),
    "hundsdorfer": FdmSchemeDesc.hundsdorfer(),
}

SCHEME_NAMES: list[str] = list(SCHEMES)

ToleranceFn = Callable[[float, float], None]


@lru_cache(maxsize=1)
def reference() -> dict[str, float]:
    """The probe's JSON output, loaded once."""
    return {k: float(v) for k, v in reference_reader.load(REFERENCE_KEY).items()}


def check_tight(key: str, actual: float) -> None:
    """Assert ``actual`` matches reference ``key`` at the TIGHT tier."""
    tight(actual, reference()[key])


def check_cancelling(key: str, actual: float) -> None:
    """Assert a cancellation-dominated quantity (theta, bumped greeks).

    See the module docstring for the derivation of the bound.
    """
    custom(
        actual,
        reference()[key],
        abs_tol=1e-12,
        rel_tol=1e-10,
        reason=(
            "cancellation-dominated: theta divides a ~1e-2 difference of two "
            "order-9 solution values by a 2.7e-3 time step, and the bumped "
            "gamma forms a ~4.7e-3 numerator out of three order-9.9 values; "
            "either turns a 1e-15 rounding into ~1e-12 relative"
        ),
    )


# --- market data (mirrors probe.cpp ``makeMarket``) ------------------------


def day_counter() -> DayCounter:
    return Actual365Fixed()


def today() -> Date:
    return Date.from_ymd(15, Month.January, 2024)


@lru_cache(maxsize=1)
def r_ts() -> FlatForward:
    """Flat 5% risk-free curve."""
    return FlatForward.from_rate(today(), 0.05, day_counter())


@lru_cache(maxsize=1)
def q_ts() -> FlatForward:
    """Flat 2% dividend curve."""
    return FlatForward.from_rate(today(), 0.02, day_counter())


@lru_cache(maxsize=1)
def vol_ts() -> BlackConstantVol:
    """Flat 20% Black vol surface."""
    return BlackConstantVol(
        reference_date=today(),
        calendar=NullCalendar(),
        volatility=0.20,
        day_counter=day_counter(),
    )


def bs_process() -> BlackScholesMertonProcess:
    """S0 = 100, q = 2%, r = 5%, sigma = 20%."""
    return BlackScholesMertonProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=q_ts(),
        risk_free_ts=r_ts(),
        black_vol_ts=vol_ts(),
    )


def empty_conditions() -> FdmStepConditionComposite:
    """An ``FdmStepConditionComposite`` with no conditions and no stopping times."""
    return FdmStepConditionComposite([], [])


# --- setups ---------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    """A mesher + inner-value calculator + solver description triple."""

    mesher: FdmMesherComposite
    calculator: FdmInnerValueCalculator
    desc: FdmSolverDesc


def _log_call_setup(
    mesher: FdmMesherComposite,
    strike: float,
    maturity: float,
    time_steps: int,
    damping_steps: int = 0,
) -> Setup:
    calc = FdmLogInnerValue(PlainVanillaPayoff(OptionType.Call, strike), mesher, 0)
    desc = FdmSolverDesc(
        mesher, empty_conditions(), calc.avg_inner_value, maturity, time_steps, damping_steps
    )
    return Setup(mesher, calc, desc)


def _max_basket_setup(
    mesher: FdmMesherComposite,
    strike: float,
    maturity: float,
    time_steps: int,
) -> Setup:
    calc = FdmLogBasketInnerValue(
        MaxBasketPayoff(PlainVanillaPayoff(OptionType.Call, strike)), mesher
    )
    desc = FdmSolverDesc(
        mesher, empty_conditions(), calc.avg_inner_value, maturity, time_steps, 0
    )
    return Setup(mesher, calc, desc)


@lru_cache(maxsize=1)
def bs_1d_setup() -> Setup:
    """25-node log-spot mesh over [log 50, log 150]; 1y call, 100 steps.

    # C++ parity: probe.cpp ``block_fdm1dimsolver`` / ``block_fdmblackscholessolver``.
    """
    mesher = FdmMesherComposite(Uniform1dMesher(math.log(50.0), math.log(150.0), 25))
    return _log_call_setup(mesher, 100.0, 1.0, 100)


@lru_cache(maxsize=1)
def bs_1d_damped_setup() -> Setup:
    """Same as :func:`bs_1d_setup` but with 5 implicit damping steps."""
    mesher = bs_1d_setup().mesher
    return _log_call_setup(mesher, 100.0, 1.0, 100, damping_steps=5)


@lru_cache(maxsize=1)
def hull_white_setup() -> Setup:
    """21-node short-rate mesh over [-0.05, 0.15]; 1y, 50 steps.

    # C++ parity: probe.cpp ``block_fdmhullwhitesolver``.
    """
    mesher = FdmMesherComposite(Uniform1dMesher(-0.05, 0.15, 21))
    return _log_call_setup(mesher, 1.0, 1.0, 50)


@lru_cache(maxsize=1)
def heston_setup() -> Setup:
    """13x9 (log-spot, variance) mesh; 3m call, 400 steps.

    # C++ parity: probe.cpp ``makeHestonSetup``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(math.log(60.0), math.log(160.0), 13),
        Uniform1dMesher(0.0, 0.4, 9),
    )
    return _log_call_setup(mesher, 100.0, 0.25, 400)


@lru_cache(maxsize=1)
def heston_process() -> HestonProcess:
    """v0 = theta = 4%, kappa = 2, sigma = 40%, rho = -50%."""
    return HestonProcess(
        risk_free_rate=r_ts(),
        dividend_yield=q_ts(),
        s0=SimpleQuote(100.0),
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.4,
        rho=-0.5,
    )


@lru_cache(maxsize=1)
def simple_2d_bs_setup() -> Setup:
    """21x11 (log-spot, log-average) mesh with a max-basket payoff; 1y, 100 steps.

    # C++ parity: probe.cpp ``block_fdmsimple2dbssolver``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(math.log(50.0), math.log(150.0), 21),
        Uniform1dMesher(math.log(50.0), math.log(150.0), 11),
    )
    return _max_basket_setup(mesher, 100.0, 1.0, 100)


@lru_cache(maxsize=1)
def two_asset_setup() -> Setup:
    """13x13 two-log-spot mesh with a max-basket (rainbow) payoff; 3m, 300 steps.

    # C++ parity: probe.cpp ``block_fdm2dblackscholessolver``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(math.log(60.0), math.log(160.0), 13),
        Uniform1dMesher(math.log(60.0), math.log(160.0), 13),
    )
    return _max_basket_setup(mesher, 100.0, 0.25, 300)


def second_asset_process() -> BlackScholesMertonProcess:
    """S0 = 95, sigma = 25%; same curves as :func:`bs_process`."""
    return BlackScholesMertonProcess(
        x0=SimpleQuote(95.0),
        dividend_ts=q_ts(),
        risk_free_ts=r_ts(),
        black_vol_ts=BlackConstantVol(
            reference_date=today(),
            calendar=NullCalendar(),
            volatility=0.25,
            day_counter=day_counter(),
        ),
    )


@lru_cache(maxsize=1)
def cir_setup() -> Setup:
    """13x9 (log-spot, CIR short rate) mesh; 3m call, 300 steps.

    # C++ parity: probe.cpp ``block_fdmcirsolver``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(math.log(60.0), math.log(160.0), 13),
        Uniform1dMesher(0.01, 0.10, 9),
    )
    return _log_call_setup(mesher, 100.0, 0.25, 300)


@lru_cache(maxsize=1)
def g2_setup() -> Setup:
    """11x11 two-factor state mesh over [-0.1, 0.1]^2; 1y, 100 steps.

    # C++ parity: probe.cpp ``block_fdmg2solver``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(-0.1, 0.1, 11), Uniform1dMesher(-0.1, 0.1, 11)
    )
    return _log_call_setup(mesher, 1.0, 1.0, 100)


@lru_cache(maxsize=1)
def bates_setup() -> Setup:
    """Same mesh as :func:`heston_setup`, built separately so the caches stay disjoint.

    # C++ parity: probe.cpp ``block_fdmbatessolver``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(math.log(60.0), math.log(160.0), 13),
        Uniform1dMesher(0.0, 0.4, 9),
    )
    return _log_call_setup(mesher, 100.0, 0.25, 400)


@lru_cache(maxsize=1)
def heston_hull_white_setup() -> Setup:
    """11x7x5 (log-spot, variance, short rate) mesh; 3m call, 400 steps.

    # C++ parity: probe.cpp ``block_3dim``.
    """
    mesher = FdmMesherComposite(
        Uniform1dMesher(math.log(70.0), math.log(140.0), 11),
        Uniform1dMesher(0.0, 0.3, 7),
        Uniform1dMesher(-0.05, 0.15, 5),
    )
    return _log_call_setup(mesher, 100.0, 0.25, 400)
