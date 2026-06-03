"""End-to-end cross-validation for the legacy FD dividend engines (FD-beta).

PRIMARY gate (correctness) — JQuantLib Java same-algorithm output
-----------------------------------------------------------------
``migration-harness/references/cluster/ws2_java.json`` holds the NPV produced by
the JQuantLib ``FD{European,American}DividendOptionHelper`` (which drive
``FDDividend{European,American}Engine``) on the canonical W-S2 scenario
(Put, S=36, K=40, r=0.06, q=0, vol=0.20, 3 dividends of 2.06 at interior dates,
expiry ~1y, timeSteps = (maturity - settlement) * 3 = 1095). The Python compat
engines are a faithful port of the same FD algorithm with the same grid
parameters, so they must reproduce the Java NPV.

Tolerance
---------
Both engines reach ``tight`` (abs 1e-14 / rel 1e-12) against the Java reference:

* **American** (``FDDividendAmericanEngine`` = ``FDAmericanCondition`` over the
  2-component control-variate ``FDStepConditionEngine``) — a single ``timeSteps``
  (=1095) system rollback. ``FDStepConditionEngine`` does NOT swap its ctor
  parameters, so the American path genuinely uses 1095 time steps.

* **European** (``FDDividendEuropeanEngine`` = Merton-73 escrowed-dividend
  ``FDMultiPeriodEngine``) — rolls back in FOUR segments between the 3 dividend
  dates, ``timeStepPerPeriod * (dividends + 1)`` Crank-Nicolson Thomas solves,
  rebuilding the log-grid + BSM operator at every dividend. The Java
  ``FDMultiPeriodEngine`` constructor declares its middle two parameters as
  ``(gridPoints, timeSteps)`` — the reverse of ``FDVanillaEngine`` — and is
  reached from ``FDDividendEngineBase`` via ``super(process, timeSteps,
  gridPoints, ...)``; the net positional effect is ``timeStepPerPeriod :=
  gridPoints`` (= 100 here), so the Java European result is bit-INVARIANT to the
  ``timeSteps`` argument. ``FDMultiPeriodEngine.__init__`` reproduces that swap
  (``_time_step_per_period = grid_points``); with it, the Python European NPV /
  delta / gamma match the Java reference to ~8e-14 (TIGHT) at any ``timeSteps``.
  (Before the swap fix the European path drifted with ``timeSteps`` to a ~7e-8
  relative bias — a real algorithm divergence, NOT a cross-runtime libm floor.)

Greeks: theta is emitted as 0.0 by the Java emitter (the helper's
date-perturbation theta path is not exercised in the reference), so it is not
asserted.
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
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_dividend_american_engine import (
    FDDividendAmericanEngine,
)
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_dividend_european_engine import (
    FDDividendEuropeanEngine,
)

_JAVA = reference_reader.load("cluster/ws2_java")
_SCEN = _JAVA["scenario"]


def _build_option(exercise_kind: str) -> DividendVanillaOption:
    """Rebuild the canonical W-S2 scenario wired to an FD dividend engine.

    Mirrors FDDividendOptionHelper: BlackScholesMertonProcess anchored at the
    settlement (reference) date; timeSteps = (maturity - settlement) * 3.
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

    payoff = PlainVanillaPayoff(OptionType.Put, strike)
    if exercise_kind == "european":
        exercise = EuropeanExercise(maturity)
        option = DividendVanillaOption(payoff, exercise, div_dates, div_amounts)
        option.set_pricing_engine(FDDividendEuropeanEngine(process, time_steps))
    else:
        exercise = AmericanExercise(settlement, maturity)
        option = DividendVanillaOption(payoff, exercise, div_dates, div_amounts)
        option.set_pricing_engine(FDDividendAmericanEngine(process, time_steps))
    return option


class TestFDDividendEuropeanEngine:
    """FDDividendEuropeanEngine reproduces JQuantLib's FD European NPV + greeks."""

    def test_npv(self) -> None:
        option = _build_option("european")
        tolerance.tight(option.npv(), float(_JAVA["fd_european"]["npv"]))

    def test_delta(self) -> None:
        option = _build_option("european")
        tolerance.tight(option.delta(), float(_JAVA["fd_european"]["delta"]))

    def test_gamma(self) -> None:
        option = _build_option("european")
        tolerance.tight(option.gamma(), float(_JAVA["fd_european"]["gamma"]))


class TestFDDividendEuropeanBitInvariance:
    """FDDividendEuropeanEngine NPV is bit-invariant to ``time_steps``.

    The ``FDMultiPeriodEngine`` param-swap fix sets ``_time_step_per_period :=
    grid_points`` (= 100), making ``time_steps`` irrelevant to the European
    rollback grid.  This test re-prices at a completely different ``time_steps``
    (200 instead of 1095) and asserts the NPV still matches the Java reference
    to TIGHT tolerance.  A future revert of the swap fix would make the two
    engines diverge because the per-period step count would track ``time_steps``
    instead of ``grid_points``, causing a measurable NPV drift (~7e-8 relative)
    that this test would catch.
    """

    def test_npv_invariant_to_time_steps(self) -> None:
        """European NPV matches Java reference regardless of time_steps value."""
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

        spot = SimpleQuote(underlying)
        r_ts = FlatForward.from_rate(settlement, r, dc)
        q_ts = FlatForward.from_rate(settlement, q, dc)
        vol_ts = BlackConstantVol(
            reference_date=settlement, calendar=calendar, day_counter=dc, volatility=vol
        )
        process = BlackScholesMertonProcess(
            x0=spot, dividend_ts=q_ts, risk_free_ts=r_ts, black_vol_ts=vol_ts
        )

        payoff = PlainVanillaPayoff(OptionType.Put, strike)
        exercise = EuropeanExercise(maturity)
        option = DividendVanillaOption(payoff, exercise, div_dates, div_amounts)
        # Use a completely different time_steps (200 vs the reference 1095).
        # grid_points stays at its default (100), which is what actually controls
        # the FDMultiPeriodEngine per-period step count after the param-swap fix.
        option.set_pricing_engine(FDDividendEuropeanEngine(process, time_steps=200))

        tolerance.tight(option.npv(), float(_JAVA["fd_european"]["npv"]))


class TestFDDividendAmericanEngine:
    """FDDividendAmericanEngine reproduces JQuantLib's FD American NPV."""

    def test_npv(self) -> None:
        option = _build_option("american")
        tolerance.tight(option.npv(), float(_JAVA["fd_american"]["npv"]))

    def test_delta(self) -> None:
        option = _build_option("american")
        tolerance.tight(option.delta(), float(_JAVA["fd_american"]["delta"]))

    def test_gamma(self) -> None:
        option = _build_option("american")
        tolerance.tight(option.gamma(), float(_JAVA["fd_american"]["gamma"]))
