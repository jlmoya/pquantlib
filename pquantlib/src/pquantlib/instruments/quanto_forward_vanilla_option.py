"""QuantoForwardVanillaOption — quanto version of a forward-starting vanilla.

# C++ parity: ql/instruments/quantoforwardvanillaoption.{hpp,cpp} (v1.43).

Combines the two decorations: the strike resets to ``moneyness *
S(reset_date)`` (from :class:`ForwardVanillaOption`) *and* the payout is
converted at a fixed exchange rate (the three quanto greeks).

Its C++ ``results`` typedef is ``QuantoOptionResults<ForwardVanillaOption::
results>``, and ``ForwardVanillaOption::results`` is itself a typedef for
``OneAssetOption::results`` (forwardvanillaoption.hpp:50) — so the results
carrier is the same :class:`QuantoOptionResults` the plain quanto vanilla
uses.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.forward_vanilla_option import ForwardVanillaOption
from pquantlib.instruments.quanto_vanilla_option import QuantoOptionResults
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineResults
from pquantlib.time.date import Date


class QuantoForwardVanillaOption(ForwardVanillaOption):
    """Quanto version of a forward vanilla option.

    # C++ parity: ``class QuantoForwardVanillaOption : public
    # ForwardVanillaOption``.

    Args:
        moneyness: strike-reset multiplier (strike = moneyness * S(reset)).
        reset_date: date at which the strike is fixed.
        payoff: the payoff carried to maturity.
        exercise: the European exercise.
    """

    def __init__(
        self,
        moneyness: float,
        reset_date: Date,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(moneyness, reset_date, payoff, exercise)
        self._qvega: float | None = None
        self._qrho: float | None = None
        self._qlambda: float | None = None

    # --- quanto greeks --------------------------------------------------------

    def qvega(self) -> float:
        """Sensitivity to the exchange-rate volatility."""
        self.calculate()
        qassert.require(self._qvega is not None, "exchange rate vega calculation failed")
        assert self._qvega is not None
        return self._qvega

    def qrho(self) -> float:
        """Sensitivity to the foreign risk-free rate."""
        self.calculate()
        qassert.require(self._qrho is not None, "foreign interest rate rho calculation failed")
        assert self._qrho is not None
        return self._qrho

    def qlambda(self) -> float:
        """Sensitivity to the underlying/exchange-rate correlation."""
        self.calculate()
        qassert.require(self._qlambda is not None, "quanto correlation sensitivity calculation failed")
        assert self._qlambda is not None
        return self._qlambda

    # --- Instrument plumbing --------------------------------------------------

    def fetch_results(self, results: PricingEngineResults) -> None:
        """# C++ parity: ``QuantoForwardVanillaOption::fetchResults``."""
        super().fetch_results(results)
        qassert.require(
            isinstance(results, QuantoOptionResults),
            "no quanto results returned from pricing engine",
        )
        assert isinstance(results, QuantoOptionResults)
        self._qrho = results.qrho
        self._qvega = results.qvega
        self._qlambda = results.qlambda

    def setup_expired(self) -> None:
        """# C++ parity: ``QuantoForwardVanillaOption::setupExpired`` — 0.0, not null."""
        super().setup_expired()
        self._qvega = 0.0
        self._qrho = 0.0
        self._qlambda = 0.0


__all__ = ["QuantoForwardVanillaOption"]
