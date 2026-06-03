"""Cross-validation tests for CRRDividendOptionHelper / European / American.

# Retired-API compat layer — W-S3a.

Java parity: ``org.jquantlib.testsuite.helpers.CRRDividendOptionTest`` —
ported 1:1 from the JQuantLib test class.

PRIMARY gate (correctness) — same-algorithm vs JQuantLib Java
-------------------------------------------------------------
``migration-harness/references/cluster/ws2_java.json`` holds the NPV +
greeks produced by the JQuantLib ``CRR{European,American}DividendOptionHelper``
on the canonical scenario (Put, K=40, S=36, r=6%, q=0%, vol=20%,
settlement=1998-05-17, maturity=1999-05-17, 3x annual dividends of 2.06).

The Python helpers are built on the same
:class:`~pquantlib_helpers.pricingengines.vanilla.binomial_dividend_vanilla_engine.BinomialDividendVanillaEngine`
that W-S2 proved bit-faithful to Java, so:

* NPV / delta / gamma — ``tight`` (abs 1e-14 / rel 1e-12).
* theta — ``custom(abs_tol=1e-8)`` — BSM PDE amplifies gamma's last-bit
  rounding by ~25.9x; same documented W-S2 precedent.

vega / rho / implied_vol — sanity gates only (bump-and-revalue; economic sign
and magnitude are checked, not exact bit reproduction).
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit
from pquantlib_helpers.helpers.crr_american_dividend_option_helper import (
    CRRAmericanDividendOptionHelper,
)
from pquantlib_helpers.helpers.crr_european_dividend_option_helper import (
    CRREuropeanDividendOptionHelper,
)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_JAVA = reference_reader.load("cluster/ws2_java")
_SCEN = _JAVA["scenario"]

# ---------------------------------------------------------------------------
# Canonical scenario — mirrors CRRDividendOptionTest.__init__
# ---------------------------------------------------------------------------

_CALENDAR = TARGET()
_TODAY = Date.from_ymd(15, Month.May, 1998)
_SETTLEMENT = Date.from_ymd(17, Month.May, 1998)
_MATURITY = Date.from_ymd(17, Month.May, 1999)
_DC = Actual365Fixed()

_OPTION_TYPE = OptionType.Put
_STRIKE = 40.0
_UNDERLYING = 36.0
_RISK_FREE_RATE = 0.06
_VOLATILITY = 0.2
_DIVIDEND_YIELD = 0.00
_DIVIDEND_AMOUNT = 2.06


def _build_dividend_schedule() -> tuple[list[Date], list[float]]:
    """Reproduce Java: 3 dividends at today+i*3M+15D for i in 1..3."""
    dates: list[Date] = []
    amounts: list[float] = []
    for i in range(1, 4):
        div_date = _TODAY + Period(i * 3, TimeUnit.Months) + Period(15, TimeUnit.Days)
        dates.append(div_date)
        amounts.append(_DIVIDEND_AMOUNT)
    return dates, amounts


def _make_european() -> CRREuropeanDividendOptionHelper:
    ObservableSettings().evaluation_date = _TODAY
    div_dates, div_amounts = _build_dividend_schedule()
    return CRREuropeanDividendOptionHelper(
        _OPTION_TYPE,
        _UNDERLYING,
        _STRIKE,
        _RISK_FREE_RATE,
        _DIVIDEND_YIELD,
        _VOLATILITY,
        _SETTLEMENT,
        _MATURITY,
        div_dates,
        div_amounts,
        _CALENDAR,
        _DC,
    )


def _make_american() -> CRRAmericanDividendOptionHelper:
    ObservableSettings().evaluation_date = _TODAY
    div_dates, div_amounts = _build_dividend_schedule()
    return CRRAmericanDividendOptionHelper(
        _OPTION_TYPE,
        _UNDERLYING,
        _STRIKE,
        _RISK_FREE_RATE,
        _DIVIDEND_YIELD,
        _VOLATILITY,
        _SETTLEMENT,
        _MATURITY,
        div_dates,
        div_amounts,
        _CALENDAR,
        _DC,
    )


# ---------------------------------------------------------------------------
# PRIMARY gate — exact same-algorithm reproduction vs JQuantLib Java
# ---------------------------------------------------------------------------


class TestCRREuropeanDividendOption:
    """European CRR dividend-option helper — Java same-algorithm gate."""

    def test_npv(self) -> None:
        option = _make_european()
        tolerance.tight(
            option.npv(),
            float(_JAVA["crr_european"]["npv"]),
        )

    def test_delta(self) -> None:
        option = _make_european()
        tolerance.tight(
            option.delta(),
            float(_JAVA["crr_european"]["delta"]),
        )

    def test_gamma(self) -> None:
        option = _make_european()
        tolerance.tight(
            option.gamma(),
            float(_JAVA["crr_european"]["gamma"]),
        )

    def test_theta(self) -> None:
        # BSM PDE amplifies gamma's last-bit libm rounding by ~25.9x;
        # same documented W-S2 precedent.  See W-S2 test for full justification.
        option = _make_european()
        tolerance.custom(
            option.theta(),
            float(_JAVA["crr_european"]["theta"]),
            abs_tol=1e-8,
            rel_tol=0.0,
            reason=(
                "theta = BSM-PDE(value, delta, gamma); "
                "the -0.5*sigma^2*S0^2*gamma term amplifies gamma's last-bit "
                "libm rounding by ~25.9x, yielding a ~1e-11 JVM<->CPython gap. "
                "NPV/delta/gamma reach TIGHT."
            ),
        )


class TestCRRAmericanDividendOption:
    """American CRR dividend-option helper — Java same-algorithm gate."""

    def test_npv(self) -> None:
        option = _make_american()
        tolerance.tight(
            option.npv(),
            float(_JAVA["crr_american"]["npv"]),
        )

    def test_delta(self) -> None:
        option = _make_american()
        tolerance.tight(
            option.delta(),
            float(_JAVA["crr_american"]["delta"]),
        )

    def test_gamma(self) -> None:
        option = _make_american()
        tolerance.tight(
            option.gamma(),
            float(_JAVA["crr_american"]["gamma"]),
        )

    def test_theta(self) -> None:
        option = _make_american()
        tolerance.custom(
            option.theta(),
            float(_JAVA["crr_american"]["theta"]),
            abs_tol=1e-8,
            rel_tol=0.0,
            reason=(
                "theta = BSM-PDE(value, delta, gamma); "
                "the -0.5*sigma^2*S0^2*gamma term amplifies gamma's last-bit "
                "libm rounding by ~25.9x, yielding a ~1e-11 JVM<->CPython gap. "
                "NPV/delta/gamma reach TIGHT."
            ),
        )


# ---------------------------------------------------------------------------
# SECONDARY gate — vega / rho / implied_vol economic sanity
# ---------------------------------------------------------------------------


class TestCRREuropeanGreeksSanity:
    """Sanity checks for bump-and-revalue greeks (European).

    vega / rho are not in the Java reference (the test class prints them but
    does not assert them — ``//TODO`` comments in the Java source). We verify
    economic sign and order of magnitude.
    """

    def test_vega_positive(self) -> None:
        """Higher vol → higher put price → vega > 0."""
        option = _make_european()
        vega = option.vega()
        assert vega > 0.0, f"vega expected positive; got {vega}"

    def test_vega_order_of_magnitude(self) -> None:
        """For a ~1Y ATM-ish put, vega should be in (0, 100)."""
        option = _make_european()
        vega = option.vega()
        assert 0.0 < vega < 100.0, f"vega out of expected range: {vega}"

    def test_rho_negative(self) -> None:
        """Higher r → lower put price → rho < 0 for a put."""
        option = _make_european()
        rho = option.rho()
        assert rho < 0.0, f"rho expected negative for a put; got {rho}"

    def test_rho_order_of_magnitude(self) -> None:
        """For a ~1Y OTM put, |rho| should be in (0, 100)."""
        option = _make_european()
        rho = option.rho()
        assert -100.0 < rho < 0.0, f"rho out of expected range: {rho}"

    def test_implied_vol_round_trips(self) -> None:
        """impliedVolatility(NPV) should reproduce the original vol ≈ 20%."""
        option = _make_european()
        target = option.npv()
        # A fresh option instance is used for the implied-vol solve so that
        # the solve's internal bump does not pollute the reference helper.
        option2 = _make_european()
        iv = option2.implied_volatility(target)
        tolerance.custom(
            iv,
            _VOLATILITY,
            abs_tol=1e-3,
            rel_tol=0.0,
            reason=(
                "round-trip implied vol at accuracy=1e-4; LOOSE-tier because "
                "the CRR engine converges only to O(1/N) vs Black-Scholes closed "
                "form, so the Brent solve is limited by tree discretization error."
            ),
        )


class TestCRRAmericanGreeksSanity:
    """Sanity checks for bump-and-revalue greeks (American).

    Under the corrected escrowed-spot model the escrowed spot (~30.02) is deep
    below the strike (40), so the American put sits exactly at the early-exercise
    boundary: its value is the intrinsic value (~9.98), independent of
    volatility. Hence vega is ~0 here — a correct economic consequence of the
    escrowed model, not a bug.
    """

    def test_vega_is_zero_at_exercise_boundary(self) -> None:
        """Deep-ITM escrowed American put = intrinsic ⇒ vega ≈ 0."""
        option = _make_american()
        vega = option.vega()
        assert abs(vega) < 1e-6, (
            f"deep-ITM escrowed American put is at the early-exercise boundary "
            f"(value = intrinsic), so vega should be ~0; got {vega}"
        )

    def test_value_is_intrinsic(self) -> None:
        """The escrowed American put prices at its intrinsic value."""
        option = _make_american()
        # escrowed spot = S0 - PV(dividends) ≈ 30.018; intrinsic = K - escrowed.
        npv = option.npv()
        assert 9.9 < npv < 10.1, (
            f"escrowed American put expected at intrinsic ~9.98; got {npv}"
        )

    def test_rho_negative(self) -> None:
        """Higher r → lower American put price → rho < 0."""
        option = _make_american()
        rho = option.rho()
        assert rho < 0.0, f"rho expected negative for an American put; got {rho}"

    def test_rho_order_of_magnitude(self) -> None:
        option = _make_american()
        rho = option.rho()
        assert -100.0 < rho < 0.0, f"rho out of expected range: {rho}"


# ---------------------------------------------------------------------------
# DEFAULT-ARGS check — NullCalendar / Actual360 defaults wired correctly
# ---------------------------------------------------------------------------


class TestCRRDefaultArgs:
    """Construct helpers with default cal/dc (NullCalendar / Actual360).

    The helpers must accept the 10-argument form (without cal/dc) and price
    without error.  We don't assert a specific NPV because the day counter
    changes the year fraction and hence the time_steps count.
    """

    def test_european_default_args_prices(self) -> None:
        ObservableSettings().evaluation_date = _TODAY
        div_dates, div_amounts = _build_dividend_schedule()
        option = CRREuropeanDividendOptionHelper(
            _OPTION_TYPE,
            _UNDERLYING,
            _STRIKE,
            _RISK_FREE_RATE,
            _DIVIDEND_YIELD,
            _VOLATILITY,
            _SETTLEMENT,
            _MATURITY,
            div_dates,
            div_amounts,
        )
        npv = option.npv()
        assert npv > 0.0, f"expected positive NPV; got {npv}"

    def test_american_default_args_prices(self) -> None:
        ObservableSettings().evaluation_date = _TODAY
        div_dates, div_amounts = _build_dividend_schedule()
        option = CRRAmericanDividendOptionHelper(
            _OPTION_TYPE,
            _UNDERLYING,
            _STRIKE,
            _RISK_FREE_RATE,
            _DIVIDEND_YIELD,
            _VOLATILITY,
            _SETTLEMENT,
            _MATURITY,
            div_dates,
            div_amounts,
        )
        npv = option.npv()
        assert npv > 0.0, f"expected positive NPV; got {npv}"


# ---------------------------------------------------------------------------
# GUARD tests — zero-vol / zero-rate degenerate cases
# ---------------------------------------------------------------------------


class TestZeroGreekGuards:
    """Verify vega/rho raise LibraryException at zero vol/rate (review fix)."""

    def test_vega_raises_at_zero_vol(self) -> None:
        """vega() must raise LibraryException when vol=0 (relative bump = 0)."""
        ObservableSettings().evaluation_date = _TODAY
        div_dates, div_amounts = _build_dividend_schedule()
        option = CRREuropeanDividendOptionHelper(
            _OPTION_TYPE,
            _UNDERLYING,
            _STRIKE,
            _RISK_FREE_RATE,
            _DIVIDEND_YIELD,
            0.0,  # zero vol — degenerate relative bump
            _SETTLEMENT,
            _MATURITY,
            div_dates,
            div_amounts,
            _CALENDAR,
            _DC,
        )
        with pytest.raises(LibraryException, match="zero volatility"):
            option.vega()

    def test_rho_raises_at_zero_rate(self) -> None:
        """rho() must raise LibraryException when r=0 (relative bump = 0)."""
        ObservableSettings().evaluation_date = _TODAY
        div_dates, div_amounts = _build_dividend_schedule()
        option = CRREuropeanDividendOptionHelper(
            _OPTION_TYPE,
            _UNDERLYING,
            _STRIKE,
            0.0,  # zero rate — degenerate relative bump
            _DIVIDEND_YIELD,
            _VOLATILITY,
            _SETTLEMENT,
            _MATURITY,
            div_dates,
            div_amounts,
            _CALENDAR,
            _DC,
        )
        with pytest.raises(LibraryException, match="zero rate"):
            option.rho()
