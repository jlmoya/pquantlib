"""NonstandardSwaption — European/Bermudan option on a NonstandardSwap.

# C++ parity: ql/instruments/nonstandardswaption.{hpp,cpp} @ v1.42.1.

Wraps a :class:`NonstandardSwap` with an :class:`Exercise` schedule.
The ``calibrationBasket`` helper (which ties into ``BlackCalibrationHelper``)
is **carved out** for Phase 11 W1-B; ``calibrationBasket`` requires
``SwapIndex.clone(Period)`` and the calibration-basket-fitting
infrastructure that's scheduled for later waves.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.instruments.nonstandard_swap import (
    NonstandardSwap,
    NonstandardSwapArguments,
    NonstandardSwapResults,
)
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    check_settlement_type_and_method_consistency,
)
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class _NullSwaptionPayoff(Payoff):
    """Placeholder payoff for nonstandard swaptions."""

    def name(self) -> str:
        return "NullNonstandardSwaptionPayoff"

    def description(self) -> str:
        return "Null payoff (used by NonstandardSwaption)"

    def __call__(self, price: float) -> float:
        _ = price
        raise NotImplementedError("NonstandardSwaption payoff is never evaluated")


_NULL_PAYOFF = _NullSwaptionPayoff()


class NonstandardSwaptionArguments(NonstandardSwapArguments, OptionArguments):
    """Engine-arguments carrier for NonstandardSwaption.

    # C++ parity: ``NonstandardSwaption::arguments`` (nonstandardswaption.hpp:82-90).
    """

    def __init__(self) -> None:
        NonstandardSwapArguments.__init__(self)
        OptionArguments.__init__(self)

    def validate(self) -> None:
        NonstandardSwapArguments.validate(self)
        qassert.require(self.swap is not None, "underlying nonstandard swap not set")
        qassert.require(self.exercise is not None, "exercise not set")
        check_settlement_type_and_method_consistency(
            SettlementType(self.settlement_type),
            SettlementMethod(self.settlement_method),
        )


class NonstandardSwaptionResults(NonstandardSwapResults, InstrumentResults):
    """Engine-results carrier."""


class NonstandardSwaption(Option):
    """Option on a NonstandardSwap.

    # C++ parity: ``class NonstandardSwaption`` (nonstandardswaption.hpp:41-79).
    """

    def __init__(
        self,
        swap: NonstandardSwap,
        exercise: Exercise,
        settlement_type: SettlementType = SettlementType.Physical,
        settlement_method: SettlementMethod = SettlementMethod.PhysicalOTC,
    ) -> None:
        super().__init__(_NULL_PAYOFF, exercise)
        self._swap: NonstandardSwap = swap
        self._settlement_type: SettlementType = settlement_type
        self._settlement_method: SettlementMethod = settlement_method
        check_settlement_type_and_method_consistency(
            settlement_type, settlement_method
        )
        swap.register_with(self)

    def is_expired(self) -> bool:
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        self._swap.setup_arguments(args)
        if not isinstance(args, NonstandardSwaptionArguments):
            return
        args.swap = self._swap
        args.exercise = self._exercise
        args.settlement_type = int(self._settlement_type)
        args.settlement_method = int(self._settlement_method)

    def type(self) -> object:
        return self._swap.type()

    def underlying_swap(self) -> NonstandardSwap:
        return self._swap

    def settlement_type(self) -> SettlementType:
        return self._settlement_type

    def settlement_method(self) -> SettlementMethod:
        return self._settlement_method


__all__ = [
    "NonstandardSwaption",
    "NonstandardSwaptionArguments",
    "NonstandardSwaptionResults",
]
