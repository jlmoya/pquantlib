"""Cross-validation tests for the dividend-option compat layer.

These exercise pquantlib-helpers' DividendVanillaOption +
BinomialDividendVanillaEngine + BlackScholesDividendLattice against two
references.

CORRECTNESS gate — European converges to the C++ analytic
---------------------------------------------------------
``migration-harness/references/cluster/ws2.json`` holds the v1.42.1
``AnalyticDividendEuropeanEngine`` European NPV (~8.0826). The fixed
``BlackScholesDividendLattice`` now implements the escrowed-spot model: it runs
a plain CRR tree on the escrowed spot ``S0 - D`` (``D`` = PV of dividends in
``(referenceDate, expiry]``), which is exactly the model the C++ analytic engine
prices in closed form. The European binomial NPV therefore *converges* to that
analytic value as the tree refines (O(1/N) discretization). At N=1095 the
relative difference is ~1.8e-5, and the binomial European also agrees with the
FD European (~8.0841) — three mutually-confirming estimates of the same number.
This is the proof the fix is correct.

SAME-ALGORITHM gate — JQuantLib Java same-algorithm output
----------------------------------------------------------
``migration-harness/references/cluster/ws2_java.json`` holds the NPV + greeks
produced by the (now-corrected) JQuantLib
``CRR{European,American}DividendOptionHelper`` on the canonical scenario. The
Python compat engine is a line-for-line port of that Java engine, so it must
reproduce these values. NPV / delta / gamma reach ``tight`` (abs 1e-14 / rel
1e-12); the only JVM<->CPython divergence is libm transcendental last-bit noise
over 1095 tree steps, which stays inside ``tight``. Theta is the one exception:
derived from value/delta/gamma via the BSM PDE, whose ``-0.5*sigma^2*S0^2*gamma``
term amplifies gamma's last-bit rounding, producing a ~1e-11 gap — a documented
LOOSE-tier ``custom`` check, not an algorithm divergence.

American (escrowed approximation; no analytic oracle)
-----------------------------------------------------
The escrowed spot (~30.02) is deep below the strike (40), so the American put
sits at the early-exercise boundary: its value (~9.98) is essentially the
intrinsic value (40 - 30.02), with delta -1 and gamma 0. This is the correct
escrowed-American result; it is checked TIGHT against the regenerated Java
reference (there is no closed-form American oracle).
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.processes.black_scholes_merton_process import (
    BlackScholesMertonProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib_helpers.instruments.dividend_vanilla_option import (
    DividendVanillaOption,
)
from pquantlib_helpers.pricingengines.vanilla.binomial_dividend_vanilla_engine import (
    BinomialDividendVanillaEngine,
    DividendTreeBuilder,
)

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

_JAVA = reference_reader.load("cluster/ws2_java")
_CPP = reference_reader.load("cluster/ws2")
_SCEN = _JAVA["scenario"]


# ---------------------------------------------------------------------------
# Scenario rebuild (identical to the JQuantLib CRRDividendOptionHelper setup)
# ---------------------------------------------------------------------------


def _build_option(exercise_kind: str) -> DividendVanillaOption:
    """Rebuild the canonical scenario as a DividendVanillaOption + engine.

    Mirrors CRRDividendOptionHelper: NullCalendar/Actual* are folded into the
    Target/Actual365Fixed used by the test; term structures anchored at the
    settlement date; timeSteps = (maturity - settlement) * 3.
    """
    underlying = float(_SCEN["underlying"])
    strike = float(_SCEN["strike"])
    r = float(_SCEN["risk_free_rate"])
    q = float(_SCEN["dividend_yield"])
    vol = float(_SCEN["volatility"])
    dc = Actual365Fixed()
    calendar = TARGET()

    settlement = Date(int(_SCEN["settlement_serial"]))
    maturity = Date(int(_SCEN["maturity_serial"]))
    div_dates = [Date(int(s)) for s in _SCEN["dividend_date_serials"]]
    div_amounts = [float(_SCEN["dividend_amount"])] * len(div_dates)
    time_steps = int(_SCEN["time_steps"])

    spot = SimpleQuote(underlying)
    r_ts = FlatForward.from_rate(settlement, r, dc)
    q_ts = FlatForward.from_rate(settlement, q, dc)
    vol_ts = BlackConstantVol(
        reference_date=settlement, calendar=calendar, day_counter=dc, volatility=vol
    )
    process = BlackScholesMertonProcess(
        x0=spot, dividend_ts=q_ts, risk_free_ts=r_ts, black_vol_ts=vol_ts
    )

    if exercise_kind == "european":
        exercise = EuropeanExercise(maturity)
    else:
        exercise = AmericanExercise(settlement, maturity)

    payoff = PlainVanillaPayoff(OptionType.Put, strike)
    option = DividendVanillaOption(payoff, exercise, div_dates, div_amounts)
    engine = BinomialDividendVanillaEngine(
        process, time_steps, DividendTreeBuilder.CoxRossRubinstein
    )
    option.set_pricing_engine(engine)
    return option


# ---------------------------------------------------------------------------
# CORRECTNESS gate — European converges to the C++ analytic escrowed value
# ---------------------------------------------------------------------------


class TestEuropeanConvergesToAnalytic:
    """The fixed escrowed-spot engine converges to AnalyticDividendEuropeanEngine.

    This is the proof of correctness. The escrowed-spot binomial prices the same
    model the C++ ``AnalyticDividendEuropeanEngine`` solves in closed form, so
    the European NPV must converge to the C++ analytic value (~8.0826) as the
    tree refines (O(1/N) discretization). It must also agree with the FD
    European estimate (~8.0841) — three independent estimates of one number.
    """

    def test_european_converges_to_cpp_analytic(self) -> None:
        option = _build_option("european")
        analytic = float(_CPP["analytic_dividend_european"]["npv"])
        tolerance.custom(
            option.npv(),
            analytic,
            abs_tol=0.0,
            rel_tol=5e-5,
            reason=(
                "CRR tree vs C++ escrowed analytic; O(1/N) discretization at "
                "N=1095 (observed rel diff ~1.8e-5). The escrowed-spot binomial "
                "prices the exact model AnalyticDividendEuropeanEngine solves in "
                "closed form, so it converges to ~8.0826"
            ),
        )

    def test_european_agrees_with_fd(self) -> None:
        """CRR European ≈ FD European — two discretizations of the same model."""
        option = _build_option("european")
        fd_european = float(_JAVA["fd_european"]["npv"])
        tolerance.custom(
            option.npv(),
            fd_european,
            abs_tol=0.0,
            rel_tol=5e-4,
            reason=(
                "CRR vs FD discretization of the same escrowed-dividend European "
                "option; both converge to the C++ analytic ~8.0826, and agree "
                "with each other to O(1/N) tree + FD grid error"
            ),
        )


# ---------------------------------------------------------------------------
# SAME-ALGORITHM gate — Java same-algorithm
# ---------------------------------------------------------------------------


class TestPrimaryJavaCrossValidation:
    """The compat engine reproduces JQuantLib's CRR dividend engine."""

    def test_european_npv(self) -> None:
        option = _build_option("european")
        tolerance.tight(option.npv(), float(_JAVA["crr_european"]["npv"]))

    def test_european_delta(self) -> None:
        option = _build_option("european")
        tolerance.tight(option.delta(), float(_JAVA["crr_european"]["delta"]))

    def test_european_gamma(self) -> None:
        option = _build_option("european")
        tolerance.tight(option.gamma(), float(_JAVA["crr_european"]["gamma"]))

    def test_european_theta(self) -> None:
        option = _build_option("european")
        # NPV / delta / gamma all reach TIGHT (the algorithm is bit-faithful to
        # Java); theta cannot, because it is derived from value/delta/gamma via
        # the BSM PDE ``theta = r*V - (r-q)*S*delta - 0.5*sigma^2*S^2*gamma``,
        # and the last term AMPLIFIES gamma's sub-TIGHT libm-rounding by
        # 0.5*sigma^2*S0^2 ~= 25.9. The observed gap is ~1e-11 absolute — i.e.
        # exactly 25.9x the ~4e-13 gamma rounding noise — not an algorithm
        # divergence. LOOSE-tier check with this written justification.
        tolerance.custom(
            option.theta(),
            float(_JAVA["crr_european"]["theta"]),
            abs_tol=1e-8,
            rel_tol=0.0,
            reason=(
                "theta = BSM-PDE(value, delta, gamma); the -0.5*sigma^2*S0^2*gamma "
                "term amplifies gamma's last-bit libm rounding by ~25.9x, "
                "yielding a ~1e-11 JVM<->CPython gap. NPV/delta/gamma reach TIGHT"
            ),
        )

    def test_american_npv(self) -> None:
        option = _build_option("american")
        tolerance.tight(option.npv(), float(_JAVA["crr_american"]["npv"]))

    def test_american_delta(self) -> None:
        option = _build_option("american")
        tolerance.tight(option.delta(), float(_JAVA["crr_american"]["delta"]))

    def test_american_gamma(self) -> None:
        option = _build_option("american")
        tolerance.tight(option.gamma(), float(_JAVA["crr_american"]["gamma"]))

    def test_american_theta(self) -> None:
        option = _build_option("american")
        # Same amplification as test_european_theta — see its comment.
        tolerance.custom(
            option.theta(),
            float(_JAVA["crr_american"]["theta"]),
            abs_tol=1e-8,
            rel_tol=0.0,
            reason=(
                "theta = BSM-PDE(value, delta, gamma); the -0.5*sigma^2*S0^2*gamma "
                "term amplifies gamma's last-bit libm rounding by ~25.9x, "
                "yielding a ~1e-11 JVM<->CPython gap. NPV/delta/gamma reach TIGHT"
            ),
        )


# ---------------------------------------------------------------------------
# Degenerate-limit sanity — empty dividend schedule reduces to plain CRR
# ---------------------------------------------------------------------------


class TestNoDividendLimit:
    """With zero dividends the escrow scale is 1.0 → plain CRR European."""

    def test_no_dividend_limit_matches_analytic(self) -> None:
        """With zero dividends the lattice should converge to the plain CRR.

        This isolates the tree discretization error from the dividend-model
        divergence: a dividend option with an empty schedule must price like a
        plain European on the same CRR tree, hence close to the C++ analytic
        no-dividend value (LOOSE, tree-vs-analytic O(1/N)).
        """
        underlying = float(_SCEN["underlying"])
        strike = float(_SCEN["strike"])
        r = float(_SCEN["risk_free_rate"])
        q = float(_SCEN["dividend_yield"])
        vol = float(_SCEN["volatility"])
        dc = Actual365Fixed()
        calendar = TARGET()
        settlement = Date(int(_SCEN["settlement_serial"]))
        maturity = Date(int(_SCEN["maturity_serial"]))
        time_steps = int(_SCEN["time_steps"])

        spot = SimpleQuote(underlying)
        process = BlackScholesMertonProcess(
            x0=spot,
            dividend_ts=FlatForward.from_rate(settlement, q, dc),
            risk_free_ts=FlatForward.from_rate(settlement, r, dc),
            black_vol_ts=BlackConstantVol(
                reference_date=settlement,
                calendar=calendar,
                day_counter=dc,
                volatility=vol,
            ),
        )
        payoff = PlainVanillaPayoff(OptionType.Put, strike)
        # Empty dividend schedule — DividendVanillaOption builds an empty
        # Dividend vector, so the lattice escrow is zero everywhere.
        option = DividendVanillaOption(payoff, EuropeanExercise(maturity), [], [])
        option.set_pricing_engine(
            BinomialDividendVanillaEngine(
                process, time_steps, DividendTreeBuilder.CoxRossRubinstein
            )
        )
        no_div_analytic = float(_CPP["analytic_european_no_dividend"]["npv"])
        tolerance.custom(
            option.npv(),
            no_div_analytic,
            abs_tol=1e-3,
            rel_tol=0.0,
            reason=(
                "CRR tree discretization vs C++ closed-form European at "
                "N=1095; O(1/N) convergence — economic sanity, not a "
                "correctness gate (the correctness gate is the Java same-"
                "algorithm comparison)"
            ),
        )


# ---------------------------------------------------------------------------
# Guard test — time_steps < 2 must raise cleanly
# ---------------------------------------------------------------------------


class TestTimeStepsGuard:
    """BinomialDividendVanillaEngine rejects time_steps < 2 at construction time.

    calculate() reads grid.at(2), lattice.underlying(2, 2), and grid.at(1)
    for the Odegaard three-point greek extraction, so time_steps == 1 would
    crash with an IndexError deep in the rollback.  The guard must surface a
    clean LibraryException before calculate() is ever called.
    """

    def _make_process(self) -> BlackScholesMertonProcess:
        """Minimal BSM process for guard-only tests (no pricing performed)."""
        dc = Actual365Fixed()
        calendar = TARGET()
        settlement = Date(int(_SCEN["settlement_serial"]))
        spot = SimpleQuote(float(_SCEN["underlying"]))
        return BlackScholesMertonProcess(
            x0=spot,
            dividend_ts=FlatForward.from_rate(settlement, float(_SCEN["dividend_yield"]), dc),
            risk_free_ts=FlatForward.from_rate(settlement, float(_SCEN["risk_free_rate"]), dc),
            black_vol_ts=BlackConstantVol(
                reference_date=settlement,
                calendar=calendar,
                day_counter=dc,
                volatility=float(_SCEN["volatility"]),
            ),
        )

    def test_time_steps_zero_raises(self) -> None:
        process = self._make_process()
        with pytest.raises(LibraryException, match="at least 2 time steps required"):
            BinomialDividendVanillaEngine(process, 0)

    def test_time_steps_one_raises(self) -> None:
        process = self._make_process()
        with pytest.raises(LibraryException, match="at least 2 time steps required"):
            BinomialDividendVanillaEngine(process, 1)

    def test_time_steps_two_is_accepted(self) -> None:
        """time_steps == 2 is the minimum accepted value — construction must succeed."""
        process = self._make_process()
        # Just assert no exception is raised; we do not price here.
        BinomialDividendVanillaEngine(process, 2)
