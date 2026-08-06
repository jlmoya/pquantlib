"""QuantoBarrierOption — quanto version of a single-asset barrier option.

# C++ parity: ql/instruments/quantobarrieroption.{hpp,cpp} (v1.43).

Same barrier mechanics as :class:`BarrierOption`, with the payout
converted at a fixed exchange rate.

Its C++ ``results`` typedef is ``QuantoOptionResults<BarrierOption::
results>``; ``BarrierOption`` declares no results class of its own
(barrieroption.hpp:43), so that resolves to ``OneAssetOption::results``
and the carrier is the same :class:`QuantoOptionResults`.

Practical consequence worth knowing: the engine C++ pairs with this
instrument, ``QuantoEngine<BarrierOption, AnalyticBarrierEngine>``, fills
only the NPV — ``AnalyticBarrierEngine`` computes no greeks at all. So
every greek accessor on a quanto barrier option, standard *and* quanto,
raises. That is C++ behaviour, not a gap in this port.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import BarrierOption, BarrierType
from pquantlib.instruments.quanto_vanilla_option import QuantoOptionResults
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineResults


class QuantoBarrierOption(BarrierOption):
    """Quanto version of a barrier option.

    # C++ parity: ``class QuantoBarrierOption : public BarrierOption``.

    Args:
        barrier_type: knock-in / knock-out discriminant.
        barrier: the barrier level.
        rebate: amount paid when the barrier condition ends the option.
        payoff: the payoff.
        exercise: the European exercise.
    """

    def __init__(
        self,
        barrier_type: BarrierType,
        barrier: float,
        rebate: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(barrier_type, barrier, rebate, payoff, exercise)
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
        """# C++ parity: ``QuantoBarrierOption::fetchResults``."""
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
        """# C++ parity: ``QuantoBarrierOption::setupExpired`` — 0.0, not null."""
        super().setup_expired()
        self._qvega = 0.0
        self._qrho = 0.0
        self._qlambda = 0.0


__all__ = ["QuantoBarrierOption"]
