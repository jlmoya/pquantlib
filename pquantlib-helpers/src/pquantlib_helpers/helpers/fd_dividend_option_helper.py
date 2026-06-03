"""FDDividendOptionHelper — abstract base for FD dividend-option helpers.

# Retired-API compat layer — NOT a port of C++ QuantLib v1.42.1.

Java parity:
``org.jquantlib.helpers.FDDividendOptionHelper`` (jquantlib-helpers).

Abstract base class for European and American dividend options using the
finite-difference engines.  Builds a
:class:`~pquantlib.processes.black_scholes_merton_process.BlackScholesMertonProcess`
from flat r/q/vol :class:`~pquantlib.quotes.simple_quote.SimpleQuote` quotes
backed by :class:`~pquantlib.termstructures.yield_.flat_forward.FlatForward` and
:class:`~pquantlib.termstructures.volatility.equity_fx.black_constant_vol.BlackConstantVol`;
wires an FD dividend engine (European or American) with
``time_steps = int((exercise.last_date() - reference_date) * 3)``.

``theta()`` is implemented by date-perturbation bump (yesterday / tomorrow),
exactly matching the Java helper's finite-difference approach.

``vega()`` and ``rho()`` are implemented by bump-and-revalue (central difference
on the vol or r quote by ±v*1e-4 / ±r*1e-4 respectively), exactly matching the
Java helper's finite-difference approach.

``implied_volatility(target_price)`` is solved through the already-wired FD
engine by bumping the internal ``_vol_quote`` directly (Brent 1D solve, bounds
[1e-7, 4.0]).

Default calendar: :class:`~pquantlib.time.calendars.null_calendar.NullCalendar`.
Default day counter: :class:`~pquantlib.daycounters.actual_360.Actual360`.
"""

from __future__ import annotations

from abc import ABC

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import Exercise
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.patterns.observable_settings import ObservableSettings
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
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_engine_adapter import (
    FDEngineAdapter,
)


class FDDividendOptionHelper(DividendVanillaOption, ABC):
    """Abstract base: FD dividend-option helper.

    Java parity: ``org.jquantlib.helpers.FDDividendOptionHelper<T extends FDEngineAdapter>``.

    Subclasses supply the concrete :class:`~pquantlib.exercise.Exercise`
    (European or American) and instantiate the correct
    :class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_engine_adapter.FDEngineAdapter`
    subclass; this class wires everything else.

    Parameters
    ----------
    engine:
        A fully-constructed FD engine adapter (FDDividendEuropeanEngine or
        FDDividendAmericanEngine), already bound to the stochastic process.
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

        # Build observable SimpleQuote handles so theta/vega/rho can bump them.
        # Java parity: ``rRate``, ``qRate``, ``vol`` are Handle<SimpleQuote>.
        self._r_quote = SimpleQuote(r)
        self._q_quote = SimpleQuote(q)
        self._vol_quote = SimpleQuote(vol)
        # Store dc for theta date-perturbation year-fraction calculation.
        self._dc = _dc

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

        # Engine is built by the subclass hook.
        engine = self._build_engine(self._process, time_steps)
        self.set_pricing_engine(engine)

        # Notify observers: set spot last so the pricing chain is live.
        spot.set_value(underlying)

    # ---- abstract hook for subclasses to supply the engine ------------------

    def _build_engine(
        self, process: BlackScholesMertonProcess, time_steps: int
    ) -> FDEngineAdapter:
        """Build the concrete FD engine; overridden by European/American subclasses."""
        raise NotImplementedError  # pragma: no cover

    # ---- overridden Greeks --------------------------------------------------

    def theta(self) -> float:
        """Theta via date-perturbation bump (yesterday / tomorrow).

        Java parity: ``FDDividendOptionHelper.theta()`` — perturbs the
        evaluation date by ±1 day, calls ``super.NPV()`` at each perturbation,
        returns the central finite difference scaled by the day-fraction, then
        restores the original evaluation date.
        """
        settings = ObservableSettings()
        eval_date = settings.evaluation_date
        yesterday = eval_date - 1
        tomorrow = eval_date + 1
        dt = self._dc.year_fraction(yesterday, tomorrow)
        settings.evaluation_date = yesterday
        value_m = self.npv()
        settings.evaluation_date = tomorrow
        value_p = self.npv()
        settings.evaluation_date = eval_date
        return (value_p - value_m) / dt

    def vega(self) -> float:
        """Vega via central-difference bump on the vol quote.

        Java parity: ``FDDividendOptionHelper.vega()`` — perturbs ``vol`` by
        ±v*1e-4, calls ``super.NPV()`` at each perturbation, returns the central
        finite difference, then restores the original value.
        """
        from pquantlib.exceptions import LibraryException  # noqa: PLC0415

        v = self._vol_quote.value()
        if v == 0.0:
            # Java parity: the relative bump v*1e-4 degenerates to 0 at v==0,
            # producing NaN (0/0) in JQuantLib.  We surface a clear exception
            # instead of an opaque ZeroDivisionError or silent NaN.
            raise LibraryException(
                "vega is undefined at zero volatility (relative bump degenerates)"
            )
        dv = v * 1.0e-4
        self._vol_quote.set_value(v + dv)
        value_p = self.npv()
        self._vol_quote.set_value(v - dv)
        value_m = self.npv()
        self._vol_quote.set_value(v)
        return (value_p - value_m) / (2.0 * dv)

    def rho(self) -> float:
        """Rho via central-difference bump on the risk-free rate quote.

        Java parity: ``FDDividendOptionHelper.rho()`` — perturbs ``rRate`` by
        ±r*1e-4, calls ``super.NPV()`` at each perturbation, returns the central
        finite difference, then restores the original value.
        """
        from pquantlib.exceptions import LibraryException  # noqa: PLC0415

        r = self._r_quote.value()
        if r == 0.0:
            # Java parity: the relative bump r*1e-4 degenerates to 0 at r==0,
            # producing NaN (0/0) in JQuantLib.  We surface a clear exception
            # instead of an opaque ZeroDivisionError or silent NaN.
            raise LibraryException(
                "rho is undefined at zero rate (relative bump degenerates)"
            )
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
        """Implied volatility by Brent-solving the FD engine price.

        Java parity: ``FDDividendOptionHelper.impliedVolatility(price)`` —
        delegates to ``impliedVolatility(price, stochProcess, 1e-4, 100, 1e-7, 4.0)``.

        # Java parity note: JQuantLib's DividendVanillaOption.impliedVolatility
        # uses ImpliedVolatilityHelper.clone to wire a FRESH FD engine —
        # unavailable here.  The parent VanillaOption.implied_volatility path is
        # also unavailable (incompatible OptionArguments).  This override solves
        # through the already-wired FD engine by bumping _vol_quote directly.
        """
        saved_vol = self._vol_quote.value()

        from pquantlib.exceptions import LibraryException  # noqa: PLC0415

        def price_error(v: float) -> float:
            self._vol_quote.set_value(v)
            try:
                return self.npv() - target_price
            except LibraryException:
                # Near-zero vol can cause degenerate FD grids.
                # Return a clearly negative residual to form a proper bracket.
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


__all__ = ["FDDividendOptionHelper"]
