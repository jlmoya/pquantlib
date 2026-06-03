"""CRRDividendOptionHelper — abstract base for CRR binomial dividend-option helpers.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

Java parity:
``org.jquantlib.helpers.CRRDividendOptionHelper`` (jquantlib-helpers).

Abstract base class for European and American dividend options using the
Cox-Ross-Rubinstein binomial method.  Builds a
:class:`~pquantlib.processes.black_scholes_merton_process.BlackScholesMertonProcess`
from flat r/q/vol :class:`~pquantlib.quotes.simple_quote.SimpleQuote` quotes
backed by :class:`~pquantlib.termstructures.yield_.flat_forward.FlatForward` and
:class:`~pquantlib.termstructures.volatility.equity_fx.black_constant_vol.BlackConstantVol`;
wires a
:class:`~pquantlib_helpers.pricingengines.vanilla.binomial_dividend_vanilla_engine.BinomialDividendVanillaEngine`
with ``time_steps = int((exercise.last_date() - reference_date) * 3)``.

``vega()`` and ``rho()`` are implemented by bump-and-revalue (central difference
on the vol or r quote by ±v*1e-4 / ±r*1e-4 respectively), exactly matching the
Java helper's finite-difference approach.

``implied_volatility(target_price)`` delegates to the parent
:class:`~pquantlib.instruments.vanilla_option.VanillaOption` solver, passing the
internally-constructed process and standard bounds (accuracy=1e-4, max_eval=100,
min_vol=1e-7, max_vol=4.0) — matching
``DividendVanillaOption.impliedVolatility(price, stochProcess, ...)`` in Java.

Default calendar: :class:`~pquantlib.time.calendars.null_calendar.NullCalendar`.
Default day counter: :class:`~pquantlib.daycounters.actual_360.Actual360`.
"""

from __future__ import annotations

from abc import ABC

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import Exercise
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.calendar import Calendar
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib_helpers.instruments.dividend_vanilla_option import DividendVanillaOption
from pquantlib_helpers.pricingengines.vanilla.binomial_dividend_vanilla_engine import (
    BinomialDividendVanillaEngine,
    DividendTreeBuilder,
)


class CRRDividendOptionHelper(DividendVanillaOption, ABC):
    """Abstract base: CRR binomial dividend-option helper.

    Java parity: ``org.jquantlib.helpers.CRRDividendOptionHelper``.

    Subclasses supply the concrete :class:`~pquantlib.exercise.Exercise`
    (European or American); this class wires everything else.

    Parameters
    ----------
    option_type:
        Call or Put (:class:`~pquantlib.payoffs.OptionType`).
    underlying:
        Spot price of the underlying asset.
    strike:
        Option strike price.
    r:
        Risk-free rate (continuously compounded).
    q:
        Dividend yield (continuously compounded).
    vol:
        Implied volatility.
    reference_date:
        Pricing reference / settlement date; anchors the flat curves.
    exercise:
        Concrete exercise type provided by subclasses.
    dividend_dates:
        Dates when discrete dividends are paid.
    dividend_amounts:
        Corresponding dividend cash amounts.
    cal:
        Calendar. Default: :class:`~pquantlib.time.calendars.null_calendar.NullCalendar`.
    dc:
        Day counter. Default: :class:`~pquantlib.daycounters.actual_360.Actual360`.
    """

    def __init__(
        self,
        option_type: OptionType,
        underlying: float,
        strike: float,
        r: float,
        q: float,
        vol: float,
        reference_date: Date,
        exercise: Exercise,
        dividend_dates: list[Date],
        dividend_amounts: list[float],
        cal: Calendar | None = None,
        dc: DayCounter | None = None,
    ) -> None:
        # Mutable-default-argument pitfall: instantiate defaults inside.
        _cal: Calendar = cal if cal is not None else NullCalendar()
        _dc: DayCounter = dc if dc is not None else Actual360()

        payoff = PlainVanillaPayoff(option_type, strike)
        super().__init__(payoff, exercise, dividend_dates, dividend_amounts)

        # Build observable SimpleQuote handles so vega/rho can bump them.
        # Java parity: ``rRate``, ``qRate``, ``vol`` are Handle<SimpleQuote>.
        self._r_quote = SimpleQuote(r)
        self._q_quote = SimpleQuote(q)
        self._vol_quote = SimpleQuote(vol)

        # Flat term structures anchored at reference_date.
        r_ts = FlatForward(reference_date, self._r_quote, _dc)
        q_ts = FlatForward(reference_date, self._q_quote, _dc)
        vol_ts = BlackConstantVol(
            reference_date=reference_date,
            calendar=_cal,
            day_counter=_dc,
            volatility=self._vol_quote,
        )

        # Spot quote (set after process construction so observers fire).
        spot = SimpleQuote(0.0)
        self._process = BlackScholesMertonProcess(
            x0=spot,
            dividend_ts=q_ts,
            risk_free_ts=r_ts,
            black_vol_ts=vol_ts,
        )

        # time_steps = (lastDate - referenceDate) * 3, matching Java int cast.
        # Java parity: ``(int)(exercise.lastDate().sub(referenceDate) * 3)``.
        # Date subtraction in pquantlib returns integer days; multiply by 3 to
        # get "3 steps per calendar day" as in JQuantLib.
        time_steps = int((exercise.last_date() - reference_date) * 3)
        engine = BinomialDividendVanillaEngine(
            self._process,
            time_steps,
            DividendTreeBuilder.CoxRossRubinstein,
        )
        self.set_pricing_engine(engine)

        # Notify observers: set spot last so the pricing chain is live.
        spot.set_value(underlying)

    # ---- overridden Greeks --------------------------------------------------

    def vega(self) -> float:
        """Vega via central-difference bump on the vol quote.

        Java parity: ``CRRDividendOptionHelper.vega()`` — perturbs ``vol`` by
        ±v*1e-4, calls ``super.NPV()`` at each perturbation, returns the central
        finite difference, then restores the original value.
        """
        v = self._vol_quote.value()
        dv = v * 1.0e-4
        self._vol_quote.set_value(v + dv)
        value_p = self.npv()
        self._vol_quote.set_value(v - dv)
        value_m = self.npv()
        self._vol_quote.set_value(v)
        return (value_p - value_m) / (2.0 * dv)

    def rho(self) -> float:
        """Rho via central-difference bump on the risk-free rate quote.

        Java parity: ``CRRDividendOptionHelper.rho()`` — perturbs ``rRate`` by
        ±r*1e-4, calls ``super.NPV()`` at each perturbation, returns the central
        finite difference, then restores the original value.
        """
        r = self._r_quote.value()
        dr = r * 1.0e-4
        self._r_quote.set_value(r + dr)
        value_p = self.npv()
        self._r_quote.set_value(r - dr)
        value_m = self.npv()
        self._r_quote.set_value(r)
        return (value_p - value_m) / (2.0 * dr)

    def implied_volatility(  # type: ignore[override]
        self,
        target_price: float,
        accuracy: float = 1.0e-4,
        max_evaluations: int = 100,
        min_vol: float = 1.0e-7,
        max_vol: float = 4.0,
    ) -> float:
        """Implied volatility by Brent-solving the CRR engine price.

        Java parity: ``CRRDividendOptionHelper.impliedVolatility(price)`` —
        delegates to ``impliedVolatility(price, stochProcess, 1e-4, 100, 1e-7, 4.0)``.

        # C++ parity note: ``VanillaOption::impliedVolatility`` clones the
        # process with a fresh ``BlackConstantVol``-backed quote and solves.
        # That codepath hits ``DividendVanillaOption.setup_arguments``, which
        # requires ``DividendVanillaOptionArguments`` — but the analytic/FD
        # engines the parent creates use plain ``OptionArguments``.  The Java
        # codepath avoids this because ``DividendVanillaOption.impliedVolatility``
        # in jquantlib delegates straight to the instrument's own engine (which
        # already carries ``DividendVanillaOptionArguments``).  We replicate
        # that semantics by bumping our internal vol quote directly and
        # re-pricing through our own wired engine (same observable chain).
        """
        saved_vol = self._vol_quote.value()

        from pquantlib.exceptions import LibraryException  # noqa: PLC0415

        def price_error(v: float) -> float:
            self._vol_quote.set_value(v)
            try:
                return self.npv() - target_price
            except LibraryException:
                # Near-zero vol can make the CRR tree degenerate
                # (negative probability). Option value at near-zero vol
                # approaches discounted intrinsic, which is typically below any
                # realistic target price — so return a clearly negative residual
                # to form a proper bracket with the positive high-vol residual.
                return -(target_price + 1.0)

        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        guess = 0.5 * (min_vol + max_vol)
        solver.set_lower_bound(min_vol)
        solver.set_upper_bound(max_vol)
        try:
            result = solver.solve(price_error, accuracy, guess, min_vol, max_vol)
        finally:
            # Always restore the original vol quote, even on solver failure.
            self._vol_quote.set_value(saved_vol)
        return result


__all__ = ["CRRDividendOptionHelper"]
