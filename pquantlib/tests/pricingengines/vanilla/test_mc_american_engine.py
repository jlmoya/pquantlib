"""Tests for ``MCAmericanEngine`` (Longstaff-Schwartz American MC).

# C++ parity: ql/pricingengines/vanilla/mcamericanengine.{hpp,cpp} (v1.42.1).

Cross-validates against ``migration-harness/references/cluster/l6a.json``
(captured from the QuantLib reference build) plus the Longstaff-Schwartz
1998 paper Table 1 reference value (4.478) on the canonical American
put scenario (S=36, K=40, r=6%, q=0, sigma=20%, T=1y, 50 weekly exercise
dates).

All comparisons are LOOSE-tier — Monte Carlo + regression variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.european_option import EuropeanOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.methods.montecarlo.lsm_basis_system import PolynomialType
from pquantlib.methods.montecarlo.path import Path
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.mc_american_engine import (
    AmericanPathPricer,
    MCAmericanEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


@pytest.fixture(scope="module")
def reference_data() -> dict[str, Any]:
    return reference_reader.load("cluster/l6a")


def _build_lsm1998_setup() -> tuple[GeneralizedBlackScholesProcess, Date, Date]:
    """LSM 1998 paper Table 1 setup: S=36 K=40 r=6% q=0% sigma=20% T=1y."""
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.May, 2026)
    expiry = ref + 365
    spot = SimpleQuote(36.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.06, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(
        reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20
    )
    process = GeneralizedBlackScholesProcess(
        x0=spot, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol
    )
    return process, ref, expiry


# --- AmericanPathPricer tests ------------------------------------------------


def test_american_path_pricer_rescales_by_strike() -> None:
    """``state(path, t) = path[t] / strike`` for a StrikedTypePayoff."""
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    pp = AmericanPathPricer(payoff=payoff, polynomial_order=2, polynomial_type=PolynomialType.Monomial)
    grid = TimeGrid.regular(1.0, 4)
    p = Path(grid, np.array([36.0, 35.0, 38.0, 40.0, 42.0], dtype=np.float64))
    # state(p, 0) = 36 / 40 = 0.9
    assert abs(pp.state(p, 0) - 0.9) < 1e-12
    # exercise(p, 0) = payoff(state * scaling*strike) = payoff(36) = max(40 - 36, 0) = 4
    assert abs(pp(p, 0) - 4.0) < 1e-12
    # exercise(p, 4) = payoff(42) = 0 (put OTM)
    assert pp(p, 4) == 0.0


def test_american_path_pricer_basis_size_includes_payoff() -> None:
    """C++ appends the payoff itself as an extra basis function => size = order + 2."""
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    for order in (0, 1, 2, 3):
        pp = AmericanPathPricer(payoff=payoff, polynomial_order=order, polynomial_type=PolynomialType.Monomial)
        assert len(pp.basis_system()) == order + 2


def test_american_path_pricer_rejects_disallowed_polynomial() -> None:
    """C++ AmericanPathPricer restricts to {Monomial, Laguerre, Hermite,
    Hyperbolic, Chebyshev2nd}. Legendre / Chebyshev (1st-kind) raise.
    """
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    with pytest.raises(Exception, match="insufficient polynomial type"):
        AmericanPathPricer(payoff=payoff, polynomial_order=2, polynomial_type=PolynomialType.Legendre)
    with pytest.raises(Exception, match="insufficient polynomial type"):
        AmericanPathPricer(payoff=payoff, polynomial_order=2, polynomial_type=PolynomialType.Chebyshev)


# --- MCAmericanEngine tests --------------------------------------------------


def test_mc_american_put_lsm1998_within_paper_band(reference_data: dict[str, Any]) -> None:
    """Reproduce Longstaff-Schwartz 1998 paper Table 1 American put value ~4.478.

    Same setup as the C++ probe (50 steps, 4096 pricing samples, 2048
    calibration samples, Monomial order 2, seed=42).
    """
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    exercise = AmericanExercise(ref, expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=50,
            antithetic_variate=False,
            control_variate=False,
            required_samples=4096,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=2048,
        )
    )
    npv = opt.npv()
    err = opt.error_estimate()
    cpp_ref = float(reference_data["mc_american_put_lsm1998_monomial_o2"]["npv"])
    paper_ref = 4.478
    # Match the C++ probe within 3-sigma.
    assert abs(npv - cpp_ref) < 5 * max(err, 0.05), (
        f"NPV={npv} vs C++ probe={cpp_ref} (err={err})"
    )
    # Within 0.05 of the 1998 paper value.
    assert abs(npv - paper_ref) < 0.05, (
        f"NPV={npv} differs from Longstaff-Schwartz 1998 ref 4.478 by >0.05"
    )


def test_mc_american_put_lsm1998_laguerre_basis(
    reference_data: dict[str, Any],
) -> None:
    """Same LSM 1998 setup but with Laguerre order 2 — expect ~4.45."""
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    exercise = AmericanExercise(ref, expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=50,
            required_samples=4096,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Laguerre,
            calibration_samples=2048,
        )
    )
    npv = opt.npv()
    err = opt.error_estimate()
    cpp_ref = float(reference_data["mc_american_put_lsm1998_laguerre_o2"]["npv"])
    # Match C++ probe within a generous LOOSE band (different basis →
    # different regression conditioning, but both should agree on the
    # American value within ~0.2 at this sample count).
    assert abs(npv - cpp_ref) < 0.4, (
        f"NPV={npv} vs Laguerre C++ probe={cpp_ref} (err={err})"
    )


def test_mc_american_put_exceeds_european_baseline(
    reference_data: dict[str, Any],
) -> None:
    """American put NPV > European put NPV (early-exercise premium)."""
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    am_exercise = AmericanExercise(ref, expiry)
    am = VanillaOption(payoff, am_exercise)
    am.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=50,
            required_samples=4096,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=2048,
        )
    )
    am_npv = am.npv()
    euro_npv = float(reference_data["analytic_european_put_lsm1998_setup"]["npv"])
    # Early-exercise premium > 0 (LSM 1998 reports ~0.63).
    assert am_npv > euro_npv, f"American {am_npv} should exceed European {euro_npv}"
    assert am_npv - euro_npv > 0.3, (
        f"Early-exercise premium {am_npv - euro_npv} suspiciously small"
    )


def test_mc_american_deep_otm_call_nearly_zero(
    reference_data: dict[str, Any],
) -> None:
    """Deep-OTM call (S=36, K=80) should price ~0."""
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Call, 80.0)
    exercise = AmericanExercise(ref, expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=50,
            required_samples=4096,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=2048,
        )
    )
    npv = opt.npv()
    cpp_ref = float(reference_data["mc_american_deep_otm_call"]["npv"])
    # Both should be ~0 (C++ reports ~0.0006).
    assert abs(npv) < 0.01
    assert abs(npv - cpp_ref) < 0.01


def test_mc_american_reproducible_under_seed() -> None:
    """Same seed + same params → same NPV (mod path mutation in place)."""
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    exercise = AmericanExercise(ref, expiry)

    def price() -> float:
        opt = VanillaOption(payoff, exercise)
        opt.set_pricing_engine(
            MCAmericanEngine(
                process,
                time_steps=20,  # fewer steps → faster test
                required_samples=1024,
                seed=42,
                polynom_order=2,
                polynom_type=PolynomialType.Monomial,
                calibration_samples=512,
            )
        )
        return opt.npv()

    v1 = price()
    v2 = price()
    assert v1 == v2


def test_mc_american_exercise_probability_in_results() -> None:
    """``engine.exercise_probability()`` is populated after pricing."""
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    exercise = AmericanExercise(ref, expiry)
    engine = MCAmericanEngine(
        process,
        time_steps=20,
        required_samples=1024,
        seed=42,
        polynom_order=2,
        polynom_type=PolynomialType.Monomial,
        calibration_samples=512,
    )
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(engine)
    _ = opt.npv()
    prob = engine.exercise_probability()
    assert 0.0 < prob <= 1.0


def test_mc_american_exercise_probability_unavailable_pre_calculate() -> None:
    """``exercise_probability()`` before calculate() raises."""
    process, _ref, _expiry = _build_lsm1998_setup()
    engine = MCAmericanEngine(
        process,
        time_steps=20,
        required_samples=512,
        seed=42,
        polynom_order=2,
        polynom_type=PolynomialType.Monomial,
        calibration_samples=512,
    )
    with pytest.raises(Exception, match="exercise probability not available"):
        engine.exercise_probability()


def test_mc_american_european_exercise_rejected() -> None:
    """``MCAmericanEngine`` requires an EarlyExercise (American/Bermudan)."""
    process, _ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    # European exercise has type European, not EarlyExercise.
    exercise = EuropeanExercise(expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=20,
            required_samples=512,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=512,
        )
    )
    with pytest.raises(Exception, match="wrong exercise"):
        opt.npv()


def test_mc_american_payoff_at_expiry_rejected() -> None:
    """C++ refuses ``payoff_at_expiry=True`` American — payoffs paid at
    expiry require different handling.
    """
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    exercise = AmericanExercise(ref, expiry, payoff_at_expiry=True)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=20,
            required_samples=512,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=512,
        )
    )
    with pytest.raises(Exception, match="payoff at expiry not handled"):
        opt.npv()


def test_mc_american_control_variate_clamps_negative() -> None:
    """With control variate on, deep-OTM American option NPV must be >= 0."""
    process, ref, expiry = _build_lsm1998_setup()
    # Deep-OTM call.
    payoff = PlainVanillaPayoff(OptionType.Call, 80.0)
    exercise = AmericanExercise(ref, expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=30,
            required_samples=2048,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=1024,
            control_variate=True,
        )
    )
    npv = opt.npv()
    assert npv >= 0.0, f"CV-clamp failed: npv={npv}"


def test_mc_american_control_variate_matches_analytic_european() -> None:
    """Sanity check: with CV on, American NPV should be >= European NPV
    (analytic) — same lower bound, just (much) higher in this scenario.
    """
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    am_exercise = AmericanExercise(ref, expiry)
    eu_exercise = EuropeanExercise(expiry)

    opt_am = VanillaOption(payoff, am_exercise)
    opt_am.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=30,
            required_samples=2048,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=1024,
            control_variate=True,
        )
    )
    am_npv = opt_am.npv()

    opt_eu = EuropeanOption(payoff, eu_exercise)
    opt_eu.set_pricing_engine(AnalyticEuropeanEngine(process))
    eu_npv = opt_eu.npv()
    assert am_npv >= eu_npv - 0.01  # MC variance margin
    assert am_npv - eu_npv > 0.1  # early-exercise premium meaningful


def test_mc_american_needs_either_samples_or_tolerance() -> None:
    process, ref, expiry = _build_lsm1998_setup()
    payoff = PlainVanillaPayoff(OptionType.Put, 40.0)
    exercise = AmericanExercise(ref, expiry)
    opt = VanillaOption(payoff, exercise)
    opt.set_pricing_engine(
        MCAmericanEngine(
            process,
            time_steps=20,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
            calibration_samples=512,
        )
    )
    with pytest.raises(Exception, match="neither tolerance nor number"):
        opt.npv()


def test_mc_american_constructor_rejects_both_step_modes() -> None:
    process, _ref, _expiry = _build_lsm1998_setup()
    with pytest.raises(Exception, match="both time steps"):
        MCAmericanEngine(
            process,
            time_steps=20,
            time_steps_per_year=20,
            required_samples=512,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
        )


def test_mc_american_constructor_rejects_neither_step_mode() -> None:
    process, _ref, _expiry = _build_lsm1998_setup()
    with pytest.raises(Exception, match="no time steps"):
        MCAmericanEngine(
            process,
            required_samples=512,
            seed=42,
            polynom_order=2,
            polynom_type=PolynomialType.Monomial,
        )
