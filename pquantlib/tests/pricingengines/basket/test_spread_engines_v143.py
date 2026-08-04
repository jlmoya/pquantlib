"""Cross-validate the two v1.43 spread engines against the C++ probe.

Reference: ``migration-harness/references/v143/spread/engines``.

Covers :class:`PearsonSpreadEngine` (Pearson 1995, adaptive Gauss-Lobatto over
a conditional Black price) and :class:`GaussianCopulaSpreadEngine` (nested
Gauss-Hermite over a Gaussian copula with smile-implied marginals).

Each probe case carries its whole market description — setup, correlation,
option type, strike, quadrature knobs — so the sweeps below reconstruct it
rather than restating it. Five markets are covered: the two upstream flat-vol
ones; a Merton pair with *different* dividend yields, which is the only way a
wrong ``fwd = spot * qDF / rDF`` shows up at all, since a ``BlackProcess`` has
q == r and hides it; and two SVI markets that differ *only* in whether
``AtmSmileSection``'s ATM override is a no-op. That last pair is what proves the
override is reproduced rather than accidentally bypassed.

Tolerances
----------
Both engines are quadrature-based, so agreement is governed by the quadrature
rather than by the last bits of the arithmetic; the relative tier is LOOSE
(1e-8). Pearson needs nothing more — it agrees with C++ to ~7e-15 absolute,
because a Gauss-Lobatto integral of a Black price involves no cancellation.

The copula engine does, and for a specific reason rather than as a concession.
Its marginals come from :class:`SmileSectionRNDCalculator`, whose CDF grid is
built from finite differences of option prices and is therefore
cancellation-limited at ``2 * eps * max(F, K) / gap`` — see that class's test
for the derivation. Inverting the grid turns that CDF floor into a *strike*
floor: since the density in strike space is ``pdf(ln S) / S``, a CDF error
``dC`` becomes a strike error

    dS = dC * S / pdf(ln S)

and the spread payoff is 1-Lipschitz in each of ``S1`` and ``S2``, so the NPV
error is at most ``df * (dS1 + dS2)``. That is :func:`_copula_npv_floor`,
evaluated per case from each leg's own forward and the port's own density
there. It does not shrink for a deep-OTM spread whose NPV is small, which is
exactly why a purely relative bound would tighten where the arithmetic is least
able to deliver.

The floor is evaluated at each leg's forward rather than as a supremum over the
whole distribution: ``dS`` grows without limit in the tails, where the density
vanishes — but there the quadrature weight and the option's exposure vanish with
it, and beyond the grid the quantile clamps onto fixed endpoints that carry no
CDF sensitivity at all. The ATM evaluation is the scale of the error over the
region that actually contributes. Measured across the reference it leaves a
factor of 2.5 in hand at the worst case, and a genuine porting error — wrong
quadrature weights, wrong copula coupling, wrong forward — moves the NPV by
percent, so nothing is given up.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.volatility.svi_smile_section import SviSmileSection
from pquantlib.instruments.basket_option import (
    AverageBasketPayoff,
    BasketOption,
    SpreadBasketPayoff,
)
from pquantlib.methods.finitedifferences.utilities.smile_section_rnd_calculator import (
    SmileSectionRNDCalculator,
)
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.gaussian_copula_spread_engine import (
    GaussianCopulaSpreadEngine,
)
from pquantlib.pricingengines.basket.pearson_spread_engine import PearsonSpreadEngine
from pquantlib.processes.black_process import BlackProcess
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.atm_smile_section import AtmSmileSection
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.volatility.equity_fx.piecewise_black_variance_surface import (
    PiecewiseBlackVarianceSurface,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

TODAY = Date.from_ymd(1, Month.March, 2025)
MATURITY = Date.from_ymd(1, Month.March, 2026)
RISK_FREE = 0.05

# SVI parameters (a, b, sigma, rho, m) — from the v1.43 test-suite case
# testGaussianCopulaSpreadEngineSVI.
SVI_1 = (0.04, 0.10, 0.30, -0.40, 0.0)
SVI_2 = (0.02, 0.08, 0.25, -0.30, 0.0)

_REL_TOL = 1.0e-8
_ABS_TOL = 1.0e-9

# The SmileSection digital-price gap, which sets the CDF floor the copula
# marginals inherit. Same constant as in the SmileSectionRNDCalculator test.
_CDF_GAP = 1.0e-5
_EPS = sys.float_info.epsilon


def _leg_strike_floor(
    process: GeneralizedBlackScholesProcess, maturity: Date
) -> float:
    """The strike-space floor one leg's smile-implied marginal inherits.

    ``dS = dC * S / pdf(ln S)`` with ``dC = 2 eps max(F, S) / gap``, evaluated
    at the leg's forward. Built through the same
    ``AtmSmileSection(section, forward)`` wrapper the engine uses, so the
    density is the one the engine actually samples.
    """
    fwd = (
        process.state_variable().value()
        * process.dividend_yield().discount(maturity)
        / process.risk_free_rate().discount(maturity)
    )
    t = process.black_volatility().time_from_reference(maturity)
    marginal = SmileSectionRNDCalculator(
        AtmSmileSection(base=process.black_volatility().smile_section_at_time(t), atm=fwd)
    )
    cdf_floor = 2.0 * _EPS * fwd / _CDF_GAP
    return cdf_floor * fwd / abs(marginal.pdf(math.log(fwd)))


def _copula_npv_floor(
    p1: GeneralizedBlackScholesProcess,
    p2: GeneralizedBlackScholesProcess,
    discount: float,
) -> float:
    """``df * (dS1 + dS2)`` — the payoff is 1-Lipschitz in each underlying."""
    return discount * (
        _leg_strike_floor(p1, MATURITY) + _leg_strike_floor(p2, MATURITY)
    )


@pytest.fixture(autouse=True)
def _set_eval_date() -> None:  # pyright: ignore[reportUnusedFunction]
    ObservableSettings().evaluation_date = TODAY


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/spread/engines")


def _assert_close(actual: float, expected: float, *, what: str, abs_floor: float) -> None:
    tolerance.custom(
        actual,
        expected,
        abs_tol=max(abs_floor, _REL_TOL * abs(expected)),
        rel_tol=0.0,
        reason=f"{what}: quadrature-limited agreement",
    )


# --- market construction: mirrors the probe exactly --------------------------


def _flat_curve(rate: float) -> YieldTermStructure:
    return FlatForward.from_rate(
        reference_date=TODAY, forward_rate=rate, day_counter=Actual365Fixed()
    )


def _flat_vol(v: float) -> BlackVolTermStructure:
    return BlackConstantVol(
        reference_date=TODAY,
        calendar=NullCalendar(),
        day_counter=Actual365Fixed(),
        volatility=v,
    )


def _svi_vol(
    forward: float, params: tuple[float, float, float, float, float]
) -> BlackVolTermStructure:
    t = Actual365Fixed().year_fraction(TODAY, MATURITY)
    return PiecewiseBlackVarianceSurface.single_tenor(
        reference_date=TODAY,
        date=MATURITY,
        smile_section=SviSmileSection(
            forward=forward, svi_params=params, exercise_time=t
        ),
        day_counter=Actual365Fixed(),
    )


def _legs(
    setup: str, risk_free: YieldTermStructure
) -> tuple[GeneralizedBlackScholesProcess, GeneralizedBlackScholesProcess]:
    """The two legs of one market.

    ``risk_free`` is passed in rather than rebuilt per leg because
    ``GaussianCopulaSpreadEngine`` compares the two processes' curves by
    identity.
    """
    match setup:
        case "a":
            # upstream testPearsonSpreadEngine market; f1 - f2 = -10
            return (
                BlackProcess(
                    x0=SimpleQuote(100.0), risk_free_ts=risk_free, black_vol_ts=_flat_vol(0.25)
                ),
                BlackProcess(
                    x0=SimpleQuote(110.0), risk_free_ts=risk_free, black_vol_ts=_flat_vol(0.35)
                ),
            )
        case "b":
            # upstream testGaussianCopulaSpreadEngineFlatVol market; f1 - f2 = 4
            return (
                BlackProcess(
                    x0=SimpleQuote(100.0), risk_free_ts=risk_free, black_vol_ts=_flat_vol(0.20)
                ),
                BlackProcess(
                    x0=SimpleQuote(96.0), risk_free_ts=risk_free, black_vol_ts=_flat_vol(0.25)
                ),
            )
        case "c":
            # different dividend yields per leg, so fwd = spot * qDF / rDF is
            # genuinely exercised
            return (
                BlackScholesMertonProcess(
                    x0=SimpleQuote(100.0),
                    dividend_ts=_flat_curve(0.02),
                    risk_free_ts=risk_free,
                    black_vol_ts=_flat_vol(0.30),
                ),
                BlackScholesMertonProcess(
                    x0=SimpleQuote(95.0),
                    dividend_ts=_flat_curve(0.06),
                    risk_free_ts=risk_free,
                    black_vol_ts=_flat_vol(0.15),
                ),
            )
        case "d":
            # upstream SVI market verbatim: the sections are built at spot/df,
            # but the legs are BlackProcess so the engines' forwards are the
            # spots — AtmSmileSection then overrides the SVI's own ATM level.
            df = risk_free.discount(MATURITY)
            return (
                BlackProcess(
                    x0=SimpleQuote(100.0),
                    risk_free_ts=risk_free,
                    black_vol_ts=_svi_vol(100.0 / df, SVI_1),
                ),
                BlackProcess(
                    x0=SimpleQuote(96.0),
                    risk_free_ts=risk_free,
                    black_vol_ts=_svi_vol(96.0 / df, SVI_2),
                ),
            )
        case "e":
            # the same SVI shapes anchored at the engines' own forwards, so the
            # override is a no-op
            return (
                BlackProcess(
                    x0=SimpleQuote(100.0),
                    risk_free_ts=risk_free,
                    black_vol_ts=_svi_vol(100.0, SVI_1),
                ),
                BlackProcess(
                    x0=SimpleQuote(96.0),
                    risk_free_ts=risk_free,
                    black_vol_ts=_svi_vol(96.0, SVI_2),
                ),
            )
        case _:
            raise AssertionError(f"unknown setup: {setup}")


def _spread_option(option_type: str, strike: float) -> BasketOption:
    return BasketOption(
        SpreadBasketPayoff(
            PlainVanillaPayoff(
                OptionType.Call if option_type == "Call" else OptionType.Put, strike
            )
        ),
        EuropeanExercise(MATURITY),
    )


def _is_guard_case(name: str) -> bool:
    return (
        "_ctor_" in name
        or "_rejects_" in name
        or name.endswith("_no_extra_results")
        or name.endswith("_derived_inputs")
    )


# --- derived market inputs ---------------------------------------------------


@pytest.mark.parametrize("setup", ["a", "b", "c", "d", "e"])
def test_derived_market_inputs(cpp: dict[str, Any], setup: str) -> None:
    """The market quantities every priced case is built on.

    Pinning them separately turns a whole column of NPV failures into a single
    obvious "the forward is wrong" rather than a search.
    """
    expected = cpp[f"{setup}_derived_inputs"]["expected"]
    risk_free = _flat_curve(RISK_FREE)
    p1, p2 = _legs(setup, risk_free)

    df = risk_free.discount(MATURITY)
    _assert_close(df, float(expected["discount"]), what=f"{setup}: discount", abs_floor=_ABS_TOL)

    fwd1 = (
        p1.state_variable().value()
        * p1.dividend_yield().discount(MATURITY)
        / p1.risk_free_rate().discount(MATURITY)
    )
    fwd2 = (
        p2.state_variable().value()
        * p2.dividend_yield().discount(MATURITY)
        / p2.risk_free_rate().discount(MATURITY)
    )
    _assert_close(fwd1, float(expected["forward1"]), what=f"{setup}: fwd1", abs_floor=_ABS_TOL)
    _assert_close(fwd2, float(expected["forward2"]), what=f"{setup}: fwd2", abs_floor=_ABS_TOL)

    _assert_close(
        p1.black_volatility().time_from_reference(MATURITY),
        float(expected["time1"]),
        what=f"{setup}: time1",
        abs_floor=_ABS_TOL,
    )
    _assert_close(
        p2.black_volatility().time_from_reference(MATURITY),
        float(expected["time2"]),
        what=f"{setup}: time2",
        abs_floor=_ABS_TOL,
    )
    _assert_close(
        p1.black_volatility().black_variance(MATURITY, fwd1, True),
        float(expected["variance1"]),
        what=f"{setup}: var1",
        abs_floor=_ABS_TOL,
    )
    _assert_close(
        p2.black_volatility().black_variance(MATURITY, fwd2, True),
        float(expected["variance2"]),
        what=f"{setup}: var2",
        abs_floor=_ABS_TOL,
    )


# --- PearsonSpreadEngine -----------------------------------------------------


def test_pearson_spread_engine(cpp: dict[str, Any]) -> None:
    """Every Pearson case in the probe.

    Includes the negative strikes that drive the ``effective_strike <= 0``
    branch, and the degenerate correlations +-1 where the conditional standard
    deviation collapses to exactly zero and Black returns the intrinsic.
    """
    checked = 0
    for name, case in cpp.items():
        if not name.startswith("pearson_") or _is_guard_case(name):
            continue
        inputs, expected = case["inputs"], case["expected"]
        risk_free = _flat_curve(RISK_FREE)
        p1, p2 = _legs(str(inputs["setup"]), risk_free)
        option = _spread_option(str(inputs["option_type"]), float(inputs["strike"]))
        option.set_pricing_engine(
            PearsonSpreadEngine(
                p1,
                p2,
                float(inputs["correlation"]),
                float(inputs.get("integration_tolerance", 1.0e-10)),
                int(inputs.get("max_integration_iterations", 10000)),
                float(inputs.get("n_std", 8.0)),
            )
        )
        _assert_close(option.npv(), float(expected["npv"]), what=name, abs_floor=_ABS_TOL)
        checked += 1
    assert checked > 20, f"expected the probe to carry Pearson cases, got {checked}"


def test_pearson_negative_strike_branch_breaks_put_call_parity(cpp: dict[str, Any]) -> None:
    """The ``effective_strike <= 0`` branch returns the CALL intrinsic always.

    It ignores ``option_type``, so wherever it fires a put picks up the call's
    payoff. Upstream's own test strikes at 5 and never reaches the branch, so
    nothing upstream notices. The port reproduces the branch verbatim —
    diverging would make the two libraries disagree — and this case pins the
    consequence so it cannot be "fixed" by accident.

    The consequence is measured as put-call parity, ``C - P = df (F1 - F2 -
    K)``, which is model-independent and therefore a fact about the engine
    rather than about the reference numbers. At strike 5 it holds to round-off;
    at strike -30, where the branch fires over the lower part of the
    integration range, C++ violates it by ~9.8e-3 — and so must this port, to
    the same value.
    """
    derived = cpp["a_derived_inputs"]["expected"]
    df = float(derived["discount"])
    f1, f2 = float(derived["forward1"]), float(derived["forward2"])

    def npv(option_type: str, strike: float) -> float:
        risk_free = _flat_curve(RISK_FREE)
        p1, p2 = _legs("a", risk_free)
        option = _spread_option(option_type, strike)
        option.set_pricing_engine(PearsonSpreadEngine(p1, p2, 0.75))
        return option.npv()

    # Strike 5: the branch never fires and parity holds to round-off.
    violation_at_5 = npv("Call", 5.0) - npv("Put", 5.0) - df * (f1 - f2 - 5.0)
    assert abs(violation_at_5) < 1.0e-10, f"parity must hold off the branch: {violation_at_5}"

    # Strike -30: the branch fires and parity breaks, by exactly as much as in
    # C++ (whose call and put values the reference carries).
    call_ref = float(cpp["pearson_a_call_km30_rho075"]["expected"]["npv"])
    put_ref = float(cpp["pearson_a_put_km30_rho075"]["expected"]["npv"])
    cpp_violation = call_ref - put_ref - df * (f1 - f2 + 30.0)
    assert abs(cpp_violation) > 1.0e-3, "the reference must actually exhibit the bug"

    violation = npv("Call", -30.0) - npv("Put", -30.0) - df * (f1 - f2 + 30.0)
    _assert_close(
        violation,
        cpp_violation,
        what="put-call parity violation at strike -30",
        abs_floor=_ABS_TOL,
    )


# --- GaussianCopulaSpreadEngine ---------------------------------------------


def test_gaussian_copula_spread_engine(cpp: dict[str, Any]) -> None:
    """Every Gaussian-copula case in the probe."""
    checked = 0
    for name, case in cpp.items():
        if not name.startswith("copula_") or _is_guard_case(name):
            continue
        inputs, expected = case["inputs"], case["expected"]
        risk_free = _flat_curve(RISK_FREE)
        p1, p2 = _legs(str(inputs["setup"]), risk_free)
        option = _spread_option(str(inputs["option_type"]), float(inputs["strike"]))
        option.set_pricing_engine(
            GaussianCopulaSpreadEngine(
                p1, p2, float(inputs["correlation"]), int(inputs.get("n_points", 64))
            )
        )
        _assert_close(
            option.npv(),
            float(expected["npv"]),
            what=name,
            abs_floor=_copula_npv_floor(p1, p2, risk_free.discount(MATURITY)),
        )
        checked += 1
    assert checked > 20, f"expected the probe to carry copula cases, got {checked}"


def test_atm_override_is_observable(cpp: dict[str, Any]) -> None:
    """Setups D and E differ only in whether the ATM override is a no-op.

    D builds each SVI section at ``spot / df`` while the legs are
    ``BlackProcess``, so the engine's forward is the spot and
    ``AtmSmileSection`` overrides the section's own ATM level; E anchors the
    same shapes at the engine's forwards, making the override a no-op. If the
    override were skipped, D would price as if its sections were self-anchored
    and the two setups would agree — so their disagreement in the reference is
    what makes the override observable at all.
    """
    d = float(cpp["copula_d_call_k4_rho050"]["expected"]["npv"])
    e = float(cpp["copula_e_call_k4_rho050"]["expected"]["npv"])
    assert abs(d - e) > 1.0e-3, "the probe must separate the two setups"

    risk_free = _flat_curve(RISK_FREE)
    for setup, expected_npv in (("d", d), ("e", e)):
        p1, p2 = _legs(setup, risk_free)
        option = _spread_option("Call", 4.0)
        option.set_pricing_engine(GaussianCopulaSpreadEngine(p1, p2, 0.5))
        _assert_close(
            option.npv(),
            expected_npv,
            what=f"setup {setup}",
            abs_floor=_copula_npv_floor(p1, p2, risk_free.discount(MATURITY)),
        )


# --- results surface ---------------------------------------------------------


def test_engines_report_only_the_value(cpp: dict[str, Any]) -> None:
    """Neither engine populates greeks or additional results.

    C++ leaves every field of the Greeks block at ``Null<Real>()``, so
    ``option.delta()`` throws "delta not provided" and
    ``additionalResults()`` is empty. This port's ``BasketOptionResults``
    carries no Greeks at all — the L5-E carve-out — so the *observable*
    surface is the same: nothing but the value. Asserted here so a future
    "helpful" addition cannot silently diverge from C++.
    """
    risk_free = _flat_curve(RISK_FREE)

    pearson_expected = cpp["pearson_no_extra_results"]["expected"]
    assert int(pearson_expected["additional_results_count"]) == 0
    assert pearson_expected["delta_throws_not_provided"] is True
    p1, p2 = _legs("a", risk_free)
    pearson = _spread_option("Call", 5.0)
    pearson.set_pricing_engine(PearsonSpreadEngine(p1, p2, 0.75))
    _assert_close(
        pearson.npv(),
        float(pearson_expected["npv"]),
        what="pearson_no_extra_results",
        abs_floor=_ABS_TOL,
    )
    assert not hasattr(pearson, "delta")

    copula_expected = cpp["copula_no_extra_results"]["expected"]
    assert int(copula_expected["additional_results_count"]) == 0
    assert copula_expected["delta_throws_not_provided"] is True
    b1, b2 = _legs("b", risk_free)
    copula = _spread_option("Call", 3.0)
    copula.set_pricing_engine(GaussianCopulaSpreadEngine(b1, b2, 0.5))
    _assert_close(
        copula.npv(),
        float(copula_expected["npv"]),
        what="copula_no_extra_results",
        abs_floor=_copula_npv_floor(b1, b2, risk_free.discount(MATURITY)),
    )
    assert not hasattr(copula, "delta")


# --- guards ------------------------------------------------------------------


def _expected_throws(cpp: dict[str, Any], case_name: str) -> bool:
    return bool(cpp[case_name]["expected"]["throws"])


def _throws(f: Callable[[], object]) -> bool:
    try:
        f()
    except LibraryException:
        return True
    return False


def test_copula_correlation_guard(cpp: dict[str, Any]) -> None:
    """Correlation must be in [-1, 1] — inclusive at both ends."""
    risk_free = _flat_curve(RISK_FREE)
    p1, p2 = _legs("b", risk_free)
    for case_name, rho in (
        ("copula_ctor_rejects_rho_above_one", 1.0000001),
        ("copula_ctor_rejects_rho_below_minus_one", -1.0000001),
        ("copula_ctor_accepts_rho_exactly_one", 1.0),
        ("copula_ctor_accepts_rho_exactly_minus_one", -1.0),
    ):
        assert _throws(
            lambda r=rho: GaussianCopulaSpreadEngine(p1, p2, r)
        ) is _expected_throws(cpp, case_name), case_name


def test_copula_requires_the_same_risk_free_object(cpp: dict[str, Any]) -> None:
    """Two *equal but distinct* flat 5% curves are rejected.

    The check compares the curves by identity, so equal numbers are not enough.
    Nothing about the priced result would differ here — which is exactly why
    this needs its own case: a port that compared values would pass every NPV
    assertion and still be wrong.
    """
    assert _expected_throws(cpp, "copula_ctor_rejects_distinct_risk_free_curves")
    shared = _flat_curve(RISK_FREE)
    other = _flat_curve(RISK_FREE)
    assert shared is not other
    tolerance.exact(shared.discount(MATURITY), other.discount(MATURITY))

    p1, _ = _legs("b", shared)
    p2_other = BlackProcess(
        x0=SimpleQuote(96.0), risk_free_ts=other, black_vol_ts=_flat_vol(0.25)
    )
    with pytest.raises(LibraryException, match="must share the risk-free term structure"):
        GaussianCopulaSpreadEngine(p1, p2_other, 0.5)


def test_pearson_does_not_validate_correlation(cpp: dict[str, Any]) -> None:
    """Pearson validates nothing; only ``sqrt(max(1 - rho**2, 0))`` clamps."""
    assert _expected_throws(cpp, "pearson_ctor_does_not_validate_correlation") is False
    risk_free = _flat_curve(RISK_FREE)
    p1, p2 = _legs("a", risk_free)
    assert _throws(lambda: PearsonSpreadEngine(p1, p2, 1.5)) is False


@pytest.mark.parametrize(
    ("case_name", "engine_factory"),
    [
        ("pearson_rejects_non_spread_payoff", PearsonSpreadEngine),
        ("copula_rejects_non_spread_payoff", GaussianCopulaSpreadEngine),
    ],
)
def test_engines_reject_a_non_spread_payoff(
    cpp: dict[str, Any], case_name: str, engine_factory: Any
) -> None:
    assert _expected_throws(cpp, case_name)
    risk_free = _flat_curve(RISK_FREE)
    p1, p2 = _legs("a", risk_free)
    option = BasketOption(
        AverageBasketPayoff(PlainVanillaPayoff(OptionType.Call, 5.0), n=2),
        EuropeanExercise(MATURITY),
    )
    option.set_pricing_engine(engine_factory(p1, p2, 0.75))
    with pytest.raises(LibraryException):
        option.npv()
