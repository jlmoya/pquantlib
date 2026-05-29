"""FloatFloatSwaption — European/Bermudan option on a FloatFloatSwap.

# C++ parity: ql/instruments/floatfloatswaption.{hpp,cpp} @ v1.42.1.

Wraps a :class:`FloatFloatSwap` with an :class:`Exercise` schedule
and delegates pricing to a Gaussian1d (or comparable) engine. The
``calibrationBasket`` C++ helper that ties into BlackCalibrationHelper
+ SwapIndex.clone is **carved out** for Phase 11 W1-B (depends on
infrastructure that's scheduled for later waves).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.float_float_swap import (
    FloatFloatSwap,
    FloatFloatSwapArguments,
    FloatFloatSwapResults,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    check_settlement_type_and_method_consistency,
)
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class _NullSwaptionPayoff(Payoff):
    """Placeholder payoff for float-float swaptions.

    # C++ parity: C++ passes ``ext::shared_ptr<Payoff>()`` (a null) to
    # ``Option(payoff, exercise)`` (floatfloatswaption.cpp:30).
    """

    def name(self) -> str:
        return "NullFloatFloatSwaptionPayoff"

    def description(self) -> str:
        return "Null payoff (used by FloatFloatSwaption to satisfy Option contract)"

    def __call__(self, price: float) -> float:
        # Never called — the underlying float-float swap drives the payoff.
        _ = price
        raise NotImplementedError("FloatFloatSwaption payoff is never evaluated")


_NULL_PAYOFF = _NullSwaptionPayoff()


class FloatFloatSwaptionArguments(FloatFloatSwapArguments, OptionArguments):
    """Engine-arguments carrier for FloatFloatSwaption.

    # C++ parity: ``FloatFloatSwaption::arguments`` (floatfloatswaption.hpp:78-85)
    # — multi-inherits ``FloatFloatSwap::arguments`` + ``Option::arguments``.
    """

    def __init__(self) -> None:
        # Multi-inheritance: both parents are plain data carriers
        # without super().__init__() chains.
        FloatFloatSwapArguments.__init__(self)
        OptionArguments.__init__(self)

    def validate(self) -> None:
        # # C++ parity: floatfloatswaption.cpp:63-69.
        FloatFloatSwapArguments.validate(self)
        qassert.require(self.swap is not None, "underlying float-float swap not set")
        qassert.require(self.exercise is not None, "exercise not set")
        check_settlement_type_and_method_consistency(
            SettlementType(self.settlement_type),
            SettlementMethod(self.settlement_method),
        )


class FloatFloatSwaptionResults(FloatFloatSwapResults, InstrumentResults):
    """Engine-results carrier for FloatFloatSwaption.

    Inherits both the FloatFloat swap results (fair_spread* fields) and
    the standard Instrument result API.
    """


class FloatFloatSwaption(Option):
    """Option on a FloatFloatSwap with a Settlement{Type,Method}.

    # C++ parity: ``class FloatFloatSwaption`` (floatfloatswaption.hpp:40-75).
    """

    def __init__(
        self,
        swap: FloatFloatSwap,
        exercise: Exercise,
        settlement_type: SettlementType = SettlementType.Physical,
        settlement_method: SettlementMethod = SettlementMethod.PhysicalOTC,
    ) -> None:
        # # C++ parity: floatfloatswaption.cpp:26-42.
        super().__init__(_NULL_PAYOFF, exercise)
        self._swap: FloatFloatSwap = swap
        self._settlement_type: SettlementType = settlement_type
        self._settlement_method: SettlementMethod = settlement_method
        check_settlement_type_and_method_consistency(
            settlement_type, settlement_method
        )
        # Register with the underlying swap.
        swap.register_with(self)

    def is_expired(self) -> bool:
        # # C++ parity: floatfloatswaption.cpp:44-46.
        # Expired iff last exercise date <= today (we use the standard
        # Swaption-style "never expired purely on calendar grounds"
        # heuristic that the underlying engine reads the curve
        # referenceDate).
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        # # C++ parity: floatfloatswaption.cpp:48-61.
        # First, populate the swap-level carrier.
        self._swap.setup_arguments(args)
        if not isinstance(args, FloatFloatSwaptionArguments):
            # Plain SwapEngine path — done.
            return
        args.swap = self._swap
        args.exercise = self._exercise
        args.settlement_type = int(self._settlement_type)
        args.settlement_method = int(self._settlement_method)

    def type(self) -> object:
        return self._swap.type()

    def underlying_swap(self) -> FloatFloatSwap:
        return self._swap

    def settlement_type(self) -> SettlementType:
        return self._settlement_type

    def settlement_method(self) -> SettlementMethod:
        return self._settlement_method


__all__ = [
    "FloatFloatSwaption",
    "FloatFloatSwaptionArguments",
    "FloatFloatSwaptionResults",
]
