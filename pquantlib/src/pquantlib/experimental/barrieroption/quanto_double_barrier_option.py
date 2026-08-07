"""QuantoDoubleBarrierOption — quanto version of a double-barrier option.

# C++ parity: ql/experimental/barrieroption/quantodoublebarrieroption.{hpp,cpp}
# (v1.43).

A :class:`~pquantlib.instruments.double_barrier_option.DoubleBarrierOption`
that additionally publishes the three quanto sensitivities (``qvega``,
``qrho``, ``qlambda``) fetched from a
:class:`~pquantlib.pricingengines.quanto.quanto_engine.QuantoEngine`.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOption,
    DoubleBarrierType,
)
from pquantlib.instruments.quanto_option_results import QuantoOptionResults
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineResults


class QuantoDoubleBarrierOption(DoubleBarrierOption):
    """Quanto double-barrier option.

    # C++ parity: ``class QuantoDoubleBarrierOption : public DoubleBarrierOption``
    # (quantodoublebarrieroption.hpp:34-58 + .cpp:24-67).
    """

    def __init__(
        self,
        barrier_type: DoubleBarrierType,
        barrier_lo: float,
        barrier_hi: float,
        rebate: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(barrier_type, barrier_lo, barrier_hi, rebate, payoff, exercise)
        self._qvega: float | None = None
        self._qrho: float | None = None
        self._qlambda: float | None = None

    # --- greeks -----------------------------------------------------------

    def qvega(self) -> float:
        """# C++ parity: ``qvega()`` (quantodoublebarrieroption.cpp:33-38)."""
        self.calculate()
        qassert.require(
            self._qvega is not None, "exchange rate vega calculation failed"
        )
        assert self._qvega is not None
        return self._qvega

    def qrho(self) -> float:
        """# C++ parity: ``qrho()`` (quantodoublebarrieroption.cpp:40-45)."""
        self.calculate()
        qassert.require(
            self._qrho is not None, "foreign interest rate rho calculation failed"
        )
        assert self._qrho is not None
        return self._qrho

    def qlambda(self) -> float:
        """# C++ parity: ``qlambda()`` (quantodoublebarrieroption.cpp:47-52)."""
        self.calculate()
        qassert.require(
            self._qlambda is not None, "quanto correlation sensitivity calculation failed"
        )
        assert self._qlambda is not None
        return self._qlambda

    # --- lifecycle --------------------------------------------------------

    def setup_expired(self) -> None:
        """# C++ parity: ``setupExpired`` (quantodoublebarrieroption.cpp:54-57)."""
        super().setup_expired()
        self._qvega = 0.0
        self._qrho = 0.0
        self._qlambda = 0.0

    def fetch_results(self, results: PricingEngineResults) -> None:
        """# C++ parity: ``fetchResults`` (quantodoublebarrieroption.cpp:59-67)."""
        super().fetch_results(results)
        qassert.require(
            isinstance(results, QuantoOptionResults),
            "no quanto results returned from pricing engine",
        )
        assert isinstance(results, QuantoOptionResults)
        self._qrho = results.qrho
        self._qvega = results.qvega
        self._qlambda = results.qlambda


__all__ = ["QuantoDoubleBarrierOption"]
