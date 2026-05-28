"""VanillaOption — single-asset option with no discrete dividends or barriers.

# C++ parity: ql/instruments/vanillaoption.{hpp,cpp} (v1.42.1) —
# ``class VanillaOption : public OneAssetOption``.

A vanilla option is a OneAssetOption whose payoff is a
``StrikedTypePayoff`` (typically ``PlainVanillaPayoff``) and whose
exercise is European / American / Bermudan.

C++ exposes an ``impliedVolatility`` helper that constructs an engine
(``AnalyticEuropeanEngine`` for European; ``FdBlackScholesVanillaEngine``
for American/Bermudan) and runs a Brent solver on
``engine_value(vol) - target = 0``. The Python port mirrors this
behaviour, cloning the process with a fresh BlackConstantVol backed
by a mutable ``SimpleQuote`` so each Brent iteration just updates
the quote and reprices.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine import (
    FdBlackScholesVanillaEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)


class VanillaOption(OneAssetOption):
    """Vanilla single-asset option (Call/Put with optional early exercise).

    # C++ parity: trivial constructor that forwards to
    # ``OneAssetOption(payoff, exercise)``.
    """

    def __init__(self, payoff: StrikedTypePayoff, exercise: Exercise) -> None:
        super().__init__(payoff, exercise)

    def is_expired(self) -> bool:
        """Return ``False`` by default.

        # C++ parity: ``Instrument::isExpired`` for VanillaOption defers
        # to the ``Settings::evaluationDate``. Until evaluation_date is
        # wired into pquantlib (deferred per L1 carve-out), this method
        # returns ``False`` so the engine always runs. Tests that need
        # an expired option can subclass and override.
        """
        return False

    def implied_volatility(
        self,
        target_value: float,
        process: GeneralizedBlackScholesProcess,
        accuracy: float = 1e-4,
        max_evaluations: int = 100,
        min_vol: float = 1.0e-7,
        max_vol: float = 4.0,
    ) -> float:
        """Solve for the constant Black vol that prices the option at ``target_value``.

        # C++ parity: ``VanillaOption::impliedVolatility``.

        Algorithm:

        1. Clone ``process`` with a fresh ``BlackConstantVol`` backed
           by a mutable ``SimpleQuote`` (the *vol quote*).
        2. Build the appropriate engine: ``AnalyticEuropeanEngine``
           for European exercises; ``FdBlackScholesVanillaEngine``
           for American / Bermudan.
        3. Brent-solve ``f(x) = engine_value(x) - target_value``
           on ``[min_vol, max_vol]``.

        The accuracy default matches C++ at 1e-4 — the FD engine
        only converges to LOOSE tolerance, so finer thresholds would
        not be meaningful for American options.
        """
        qassert.require(not self.is_expired(), "option expired")

        vol_quote = SimpleQuote(0.0)

        # Clone the process replacing the BlackVol curve with a constant-vol
        # curve backed by the mutable quote.
        original_vol_ts = process.black_volatility()
        new_vol_ts = BlackConstantVol(
            reference_date=original_vol_ts.reference_date(),
            calendar=original_vol_ts.calendar(),
            day_counter=original_vol_ts.day_counter(),
            volatility=vol_quote,
        )
        new_process: GeneralizedBlackScholesProcess = GeneralizedBlackScholesProcess(
            x0=process.state_variable(),
            dividend_ts=process.dividend_yield(),
            risk_free_ts=process.risk_free_rate(),
            black_vol_ts=new_vol_ts,
        )

        # Select the engine matching the exercise type.
        engine: PricingEngine
        if self._exercise.type() == Exercise.Type.European:
            engine = AnalyticEuropeanEngine(new_process)
        elif self._exercise.type() in (
            Exercise.Type.American,
            Exercise.Type.Bermudan,
        ):
            engine = FdBlackScholesVanillaEngine(new_process)
        else:
            raise LibraryException(
                f"VanillaOption.implied_volatility: unknown exercise type {self._exercise.type()}"
            )

        # Wire the engine arguments.
        engine.reset()
        args = engine.get_arguments()
        self.setup_arguments(args)
        args.validate()

        # Brent on f(x) = engine_value(x) - target_value.
        def price_error(x: float) -> float:
            vol_quote.set_value(x)
            engine.calculate()
            results = engine.get_results()
            assert results.value is not None
            return float(results.value) - target_value

        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        guess = 0.5 * (min_vol + max_vol)
        return solver.solve(price_error, accuracy, guess, min_vol, max_vol)


__all__ = ["VanillaOption"]
