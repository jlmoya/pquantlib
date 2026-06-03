"""Cross-validation tests for the retired-API dividend-option compat layer.

These exercise pquantlib-helpers' DividendVanillaOption +
BinomialDividendVanillaEngine + BlackScholesDividendLattice against two
references.

PRIMARY gate (correctness) — JQuantLib Java same-algorithm output
-----------------------------------------------------------------
``migration-harness/references/cluster/ws2_java.json`` holds the NPV + greeks
produced by the JQuantLib ``CRR{European,American}DividendOptionHelper`` (which
drive ``BinomialDividendVanillaEngine<CoxRossRubinstein>``) on the canonical
test scenario. The Python compat engine is a line-for-line port of that Java
engine, so it must reproduce these values.

Tolerance: NPV / delta / gamma all reach ``tight`` (abs 1e-14 / rel 1e-12) for
both European and American — the algorithm is bit-faithful to Java; the only
JVM<->CPython divergence is libm transcendental last-bit noise (``exp`` /
``log`` / ``sqrt``) accumulated over 1095 tree steps, which stays inside
``tight``. Theta is the one exception: it is derived from value/delta/gamma via
the BSM PDE, whose ``-0.5*sigma^2*S0^2*gamma`` term amplifies gamma's last-bit
rounding by ~25.9x, producing a ~1e-11 gap — a documented LOOSE-tier
``custom`` check, not an algorithm divergence. The ``*_bits`` fields are
retained in the reference for forensic comparison, not asserted.

SECONDARY gate (economic sanity) — C++ v1.42.1 analytic, European only
----------------------------------------------------------------------
``migration-harness/references/cluster/ws2.json`` holds the v1.42.1
``AnalyticDividendEuropeanEngine`` (escrowed-dividend closed form) NPV and the
no-dividend ``AnalyticEuropeanEngine`` NPV for the European scenario.

NOTE (notable finding): the original plan expected the European compat NPV to
match the C++ analytic value within ~1e-3. It does NOT, and CANNOT, because the
retired JQuantLib ``BlackScholesDividendLattice`` uses a structurally different
dividend model: its escrow accumulator sums the *bare discount factor*
``exp(-r t)`` per dividend and drops the dividend cash amount
(BlackScholesDividendLattice.java:67). With a 2.06 dividend amount this
under-subtracts the escrow, so the lattice NPV (~4.45) sits strictly between
the no-dividend analytic (~3.84) and the full-escrow C++ analytic (~8.08).
Reproducing this Java behaviour verbatim is the whole point of a compat layer,
so the meaningful economic check here is a *bracketing* one: the dividend
lattice NPV must exceed the no-dividend analytic value (the discrete dividends
do raise a put), and must stay below the full-escrow analytic value. This is a
sanity bracket, not a 1e-3 convergence gate.
"""

from __future__ import annotations

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
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
# PRIMARY gate — Java same-algorithm
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
# SECONDARY gate — C++ analytic economic sanity (European bracket)
# ---------------------------------------------------------------------------


class TestSecondaryCppEconomicSanity:
    """European compat NPV brackets between no-dividend and full-escrow analytic.

    See the module docstring: the retired JQuantLib lattice's escrow model
    differs structurally from the C++ analytic engine, so this is a bracket
    sanity check, not a convergence gate. The bracket itself uses LOOSE-tier
    margins (>1e-8) BY DESIGN — tree-vs-analytic + model divergence.
    """

    def test_european_npv_brackets_analytic(self) -> None:
        option = _build_option("european")
        npv = option.npv()
        no_div = float(_CPP["analytic_european_no_dividend"]["npv"])
        full_escrow = float(_CPP["analytic_dividend_european"]["npv"])
        # Discrete dividends raise a put above the no-dividend value.
        assert npv > no_div, (
            f"dividend lattice NPV {npv} should exceed no-dividend "
            f"analytic {no_div}"
        )
        # ...but the JQuantLib escrow under-subtracts vs full escrow.
        assert npv < full_escrow, (
            f"dividend lattice NPV {npv} should stay below full-escrow "
            f"analytic {full_escrow}"
        )

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
