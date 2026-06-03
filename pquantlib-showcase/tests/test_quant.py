"""Regression tests for the quant compute layer.

These pin down that every PQuantLib code path the showcase relies on produces
sane, finite, internally-consistent numbers — so a library or wiring change is
caught here rather than in the browser.
"""

from __future__ import annotations

import math

from pquantlib_showcase.quant import bonds, curves, dates, exotics, heston, options, swaps


def test_dates_daycount_and_schedule() -> None:
    facts = dates.calendar_facts(2026, 6, 15, "TARGET (EUR)")
    assert facts.is_business_day
    table = dates.day_count_table(2026, 6, 15, 12)
    # Actual/365 Fixed over exactly one year is 1.0; 30/360 counts 360 days.
    names = {n: (dc, yf) for n, dc, yf in table}
    assert math.isclose(names["Actual/365 Fixed"][1], 1.0, abs_tol=1e-9)
    assert names["30/360 (Bond Basis)"][0] == 360
    sched = dates.schedule_table(2026, 6, 15, 6, 2, "TARGET (EUR)")
    assert len(sched) == 5  # 2y semiannual -> 5 dates


def test_flat_curve_is_flat() -> None:
    prof = curves.flat_curve_profile(0.04, 10.0)
    assert all(abs(z - 0.04) < 1e-6 for z in prof.zero_rates)
    assert all(0.0 < df <= 1.0 for df in prof.discount_factors)


def test_bootstrap_reprices_inputs() -> None:
    res = curves.bootstrap_deposit_curve([0.02, 0.023, 0.026, 0.028, 0.030])
    assert all(0.0 < df <= 1.0 for df in res.profile.discount_factors)
    assert len(res.pillar_labels) == 5


def test_bond_premium_when_coupon_exceeds_rate() -> None:
    res = bonds.price_bond(coupon=0.06, years=5, curve_rate=0.04)
    assert res.clean_price > 100.0  # high coupon -> premium
    assert res.dirty_price >= res.clean_price
    assert 0.03 < res.ytm < 0.05
    assert len(res.cashflows) > 5


def test_swap_par_rate_zeroes_npv() -> None:
    res = swaps.price_vanilla_swap(1_000_000.0, 5, fixed_rate=0.03, curve_rate=0.03)
    at_par = swaps.price_vanilla_swap(1_000_000.0, 5, fixed_rate=res.fair_rate, curve_rate=0.03)
    assert abs(at_par.npv) < 1.0  # NPV ~ 0 at the par rate
    ois = swaps.price_ois(1_000_000.0, 2, 0.03, 0.03)
    assert 0.0 < ois.fair_rate < 0.10


def test_four_engines_agree() -> None:
    res = options.price_vanilla("Call", 100, 100, 0.05, 0.02, 0.20, 1.0, mc_samples=50_000)
    assert abs(res.binomial - res.analytic) < 0.02
    assert abs(res.fd - res.analytic) < 0.05
    assert abs(res.mc - res.analytic) < 3 * res.mc_error  # within 3 sigma
    # Greeks for an ATM call are in textbook ranges.
    assert 0.5 < res.delta < 0.7
    assert res.gamma > 0 and res.vega > 0


def test_implied_vol_round_trip() -> None:
    px = options.price_vanilla("Put", 95, 100, 0.03, 0.01, 0.30, 0.75).analytic
    iv = options.implied_vol("Put", 95, 100, 0.03, 0.01, 0.75, px)
    assert abs(iv - 0.30) < 1e-3


def test_binomial_converges() -> None:
    ns, prices, target = options.binomial_convergence("Call", 100, 100, 0.05, 0.0, 0.2, 1.0, max_steps=150)
    assert abs(prices[-1] - target) < 0.05


def test_barrier_cheaper_than_vanilla() -> None:
    res = exotics.price_barrier("Call", "Down-and-Out", 100, 100, 90, 0.0, 0.05, 0.0, 0.25, 1.0)
    assert 0 < res.price < res.vanilla_price


def test_double_barrier_and_asian_below_vanilla() -> None:
    db = exotics.price_double_barrier("Call", "Knock-Out", 100, 100, 80, 120, 0.0, 0.05, 0.0, 0.2, 1.0)
    assert 0 < db.price < db.vanilla_price
    asian = exotics.price_asian("Call", 100, 100, 0.05, 0.0, 0.2, 1.0)
    assert 0 < asian.price < asian.vanilla_price


def test_heston_produces_smile() -> None:
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    sm = heston.heston_smile(100, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7, 1.0, strikes)
    assert all(p > 0 for p in sm.prices)
    # Negative rho => downward skew: low-strike vol above high-strike vol.
    assert sm.implied_vols[0] > sm.implied_vols[-1]


def test_heston_calibration_fits_smile() -> None:
    strikes, vols = heston.synthetic_market_smile()
    cal = heston.calibrate_heston(100, 0.05, 0.0, 1.0, strikes, vols)
    assert cal.rmse_bps < 50.0  # fits the smile to better than 50 bps
    assert all(p > 0 for p in (cal.v0, cal.kappa, cal.theta, cal.sigma))
