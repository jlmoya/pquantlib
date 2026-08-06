"""HestonBlackVolSurface, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/eqfx/hestonblackvol.json
Probe:     migration-harness/cpp/probes/v143_eqfx_hestonblackvol/probe.cpp

Tolerance — why this file derives a per-cell budget instead of naming a tier.

The surface is an implied vol: a Heston price, inverted through Black. C++
computes the price with ``AngledContour`` + 160-point Gauss-Laguerre;
PQuantLib's ``AnalyticHestonEngine`` implements only the Gatheral branch and
integrates with ``scipy.integrate.quad``, whose default absolute accuracy is
1.49e-8. So the two prices differ by an absolute amount of order 1e-8, and the
vol difference that produces is

    |d(vol)| = |d(price)| / vega

which is flat at the money (vega ~ 20 for a 100-forward, so ~1e-9 in vol) and
unbounded in the far wings, where vega vanishes exponentially. At
``t = 0.05, K = 250`` vega is ~1e-30 and the two implementations do not even
agree on whether the option has a positive price: one hits the
``npv <= 0 -> sqrt(theta)`` fallback and the other does not. Naming a single
tolerance would either fail on those cells or be so loose as to test nothing.

So each cell gets its own budget, ``_PRICE_ABS_TOL / vega``, computed from the
reference vol. Cells whose budget exceeds ``_UNINFORMATIVE`` (10% of the vol)
are counted rather than asserted, and the test fails if too few cells remain
informative — that is the guard against the budget quietly swallowing the
whole grid.

The divergence is real and worth reporting at merge: it is the missing
``ComplexLogFormula`` / ``Integration`` support in the engine, not a defect in
this class.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.pricingengines.black_formula import black_formula_vol_derivative
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.heston_black_vol_surface import (
    HestonBlackVolSurface,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

_REF = reference_reader.load("v143/eqfx/hestonblackvol")

_DC = Actual365Fixed()
_REF_DATE = Date.from_ymd(15, Month.June, 2026)
_R = 0.04
_Q = 0.02
_SPOT = 100.0

# scipy.integrate.quad's default absolute accuracy, which is what bounds the
# price difference between the two integration schemes.
_PRICE_ABS_TOL = 1.49e-8
# A cell whose derived vol budget is worse than 10% of the vol tests nothing.
_UNINFORMATIVE = 0.10
# At least this many of the 35 cells per case must stay informative.
_MIN_INFORMATIVE = 25

_TIMES: list[tuple[str, float]] = [
    ("t0p05", 0.05),
    ("t0p25", 0.25),
    ("t1p0", 1.0),
    ("t3p0", 3.0),
    ("t10p0", 10.0),
]
_STRIKES: list[tuple[str, float]] = [
    ("k40", 40.0),
    ("k60", 60.0),
    ("k80", 80.0),
    ("k100", 100.0),
    ("k120", 120.0),
    ("k160", 160.0),
    ("k250", 250.0),
]

# (v0, kappa, theta, sigma, rho) — mirrors the probe.
_CASES: dict[str, tuple[float, float, float, float, float]] = {
    "feller_ok": (0.04, 2.5, 0.05, 0.4, -0.6),
    "feller_violated": (0.06, 0.5, 0.04, 0.6, -0.75),
}


def _surface(case: str) -> HestonBlackVolSurface:
    v0, kappa, theta, sigma, rho = _CASES[case]
    risk_free = FlatForward.from_rate(
        _REF_DATE, _R, _DC, Compounding.Continuous, Frequency.Annual
    )
    dividend = FlatForward.from_rate(
        _REF_DATE, _Q, _DC, Compounding.Continuous, Frequency.Annual
    )
    process = HestonProcess(
        risk_free_rate=risk_free,
        dividend_yield=dividend,
        s0=SimpleQuote(_SPOT),
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
    )
    return HestonBlackVolSurface(HestonModel(process))


def _vol_budget(t: float, strike: float, reference_vol: float) -> float:
    """Vol error a 1.49e-8 price error produces at this (t, K): |dp| / vega."""
    discount = math.exp(-_R * t)
    forward = _SPOT * math.exp(-_Q * t) / discount
    vega = black_formula_vol_derivative(
        strike, forward, reference_vol * math.sqrt(t), t, discount
    )
    if vega <= 0.0:
        return math.inf
    return _PRICE_ABS_TOL / vega


@pytest.mark.parametrize("case", list(_CASES))
def test_accessors(case: str) -> None:
    surface = _surface(case)
    expected = _REF[case]
    assert surface.reference_date().serial_number() == expected["reference_date"]
    tolerance.exact(surface.min_strike(), expected["min_strike"])
    tolerance.exact(surface.max_strike(), expected["max_strike"])
    assert surface.max_date() == Date.max_date()


@pytest.mark.parametrize("case", list(_CASES))
def test_atm_level(case: str) -> None:
    """The forward — pure discounting, so TIGHT applies with no caveats."""
    surface = _surface(case)
    expected = _REF[case]["atm_level"]
    for t_label, t in _TIMES:
        tolerance.tight(surface.atm_level(t), expected[t_label])


def _check_surface_grid(case: str, kind: str) -> None:
    surface = _surface(case)
    expected: dict[str, Any] = _REF[case][kind]
    vols: dict[str, Any] = _REF[case]["vol"]
    informative = 0
    for t_label, t in _TIMES:
        for k_label, strike in _STRIKES:
            key = f"{t_label}_{k_label}"
            reference_vol = float(vols[key])
            want = float(expected[key])
            got = (
                surface.black_vol_at_time(t, strike, extrapolate=True)
                if kind == "vol"
                else surface.black_variance_at_time(t, strike, extrapolate=True)
            )
            vol_budget = _vol_budget(t, strike, reference_vol)
            # variance = vol^2 * t, so d(var) = 2 * vol * t * d(vol).
            budget = vol_budget if kind == "vol" else 2.0 * reference_vol * t * vol_budget
            scale = abs(want) if want else 1.0
            if vol_budget / max(reference_vol, 1e-300) > _UNINFORMATIVE:
                continue
            informative += 1
            assert abs(got - want) <= max(budget, 1e-14 + 1e-12 * scale), (
                f"{case}/{kind}/{key}: got {got!r}, want {want!r}, "
                f"derived budget {budget!r}"
            )
    assert informative >= _MIN_INFORMATIVE, (
        f"{case}/{kind}: only {informative} of "
        f"{len(_TIMES) * len(_STRIKES)} cells were informative; the derived "
        "budget has swallowed the grid"
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_black_vol_grid(case: str) -> None:
    _check_surface_grid(case, "vol")


@pytest.mark.parametrize("case", list(_CASES))
def test_black_variance_grid(case: str) -> None:
    _check_surface_grid(case, "variance")


@pytest.mark.parametrize("case", list(_CASES))
def test_variance_is_vol_squared_times_t(case: str) -> None:
    """Self-consistency: ``blackVarianceImpl = squared(blackVolImpl) * t``."""
    surface = _surface(case)
    for _, t in _TIMES:
        for _, strike in _STRIKES:
            vol = surface.black_vol_at_time(t, strike, extrapolate=True)
            variance = surface.black_variance_at_time(t, strike, extrapolate=True)
            tolerance.tight(variance, vol * vol * t)


def test_far_wing_falls_back_to_sqrt_theta() -> None:
    """The ``npv <= 0`` branch, pinned where C++ takes it too.

    At t = 0.05, K = 40 the Feller-satisfying model's put price underflows in
    both implementations, and both return sqrt(theta) exactly.
    """
    theta = float(_REF["feller_ok"]["theta"])
    expected = float(_REF["feller_ok"]["vol"]["t0p05_k40"])
    tolerance.exact(expected, math.sqrt(theta))
    surface = _surface("feller_ok")
    tolerance.exact(
        surface.black_vol_at_time(0.05, 40.0, extrapolate=True), math.sqrt(theta)
    )


def test_option_type_switches_at_the_forward_not_the_spot() -> None:
    """With r > q the forward is above spot, so K = 100 prices as a Put.

    The switch is not directly observable from the surface, so this asserts
    the property that makes it matter: the surface reproduces C++ at K = 100
    for every maturity, where forward > spot = 100 and a spot-based switch
    would price the other option.
    """
    surface = _surface("feller_ok")
    expected = _REF["feller_ok"]["vol"]
    for t_label, t in _TIMES:
        assert surface.atm_level(t) > _SPOT
        tolerance.custom(
            surface.black_vol_at_time(t, 100.0, extrapolate=True),
            float(expected[f"{t_label}_k100"]),
            abs_tol=1e-14,
            rel_tol=1e-12,
            reason="at the money vega is ~20, so a 1.5e-8 price error is ~1e-9 in vol",
        )
