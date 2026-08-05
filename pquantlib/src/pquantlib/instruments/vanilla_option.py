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
behaviour, delegating the process clone and the Brent solve to
``ImpliedVolatilityHelper`` — which is what C++'s own
``VanillaOption::impliedVolatility`` does with
``detail::ImpliedVolatilityHelper``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.implied_volatility import ImpliedVolatilityHelper
from pquantlib.instruments.one_asset_option import OneAssetOption
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
        # Steps 1 and 3 are `ImpliedVolatilityHelper.clone` / `.calculate`
        # (C++ `detail::ImpliedVolatilityHelper`, which C++'s own
        # `VanillaOption::impliedVolatility` calls for exactly this).
        new_process = ImpliedVolatilityHelper.clone(process, vol_quote)

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

        return ImpliedVolatilityHelper.calculate(
            self,
            engine,
            vol_quote,
            target_value,
            accuracy,
            max_evaluations,
            min_vol,
            max_vol,
        )


__all__ = ["VanillaOption"]
