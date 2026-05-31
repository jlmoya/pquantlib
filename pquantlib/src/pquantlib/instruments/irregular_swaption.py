"""IrregularSwaption — option on an IrregularSwap.

# C++ parity: ql/experimental/swaptions/irregularswaption.hpp + .cpp (v1.42.1,
# 099987f0).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.irregular_swap import IrregularSwap, IrregularSwapArguments
from pquantlib.instruments.swap import SwapType
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class _NullIrregularSwaptionPayoff(Payoff):
    """Placeholder payoff — C++ passes a null ``shared_ptr<Payoff>``.

    # C++ parity: irregularswaption.cpp:102 — the engines never read the
    # swaption payoff; the irregular swap defines the cashflow structure.
    # PQuantLib uses a sentinel to satisfy ``Option.__init__``.
    """

    def name(self) -> str:
        return "NullIrregularSwaptionPayoff"

    def description(self) -> str:
        return "Null payoff (used by IrregularSwaption to satisfy Option contract)"

    def __call__(self, price: float) -> float:
        del price
        return 0.0


_NULL_IRREGULAR_PAYOFF: _NullIrregularSwaptionPayoff = _NullIrregularSwaptionPayoff()


class IrregularSettlement:
    """Settlement information for an irregular swaption.

    # C++ parity: ``struct IrregularSettlement { enum Type { Physical, Cash }; }``
    # in irregularswaption.hpp:38-43.
    """

    class Type(IntEnum):
        Physical = 0
        Cash = 1

    @staticmethod
    def to_string(t: IrregularSettlement.Type) -> str:
        """C++ parity: ``operator<<`` (irregularswaption.cpp:87-97)."""
        if t == IrregularSettlement.Type.Physical:
            return "Delivery"
        if t == IrregularSettlement.Type.Cash:
            return "Cash"
        qassert.fail(f"unknown IrregularSettlement::Type({int(t)})")


class IrregularSwaptionArguments(IrregularSwapArguments, OptionArguments):
    """Engine argument carrier for IrregularSwaption.

    # C++ parity: ``IrregularSwaption::arguments`` (irregularswaption.hpp:82-90)
    # multi-inherits ``IrregularSwap::arguments`` and ``Option::arguments``.
    """

    def __init__(self) -> None:
        IrregularSwapArguments.__init__(self)
        OptionArguments.__init__(self)
        self.swap: IrregularSwap | None = None
        self.settlement_type: IrregularSettlement.Type = IrregularSettlement.Type.Physical

    def validate(self) -> None:
        # # C++ parity: irregularswaption.cpp:124-128.
        IrregularSwapArguments.validate(self)
        qassert.require(self.swap is not None, "Irregular swap not set")
        qassert.require(self.exercise is not None, "exercise not set")


class IrregularSwaption(Option):
    """Option on an :class:`IrregularSwap`."""

    def __init__(
        self,
        swap: IrregularSwap,
        exercise: Exercise,
        delivery: IrregularSettlement.Type = IrregularSettlement.Type.Physical,
    ) -> None:
        # # C++ parity: irregularswaption.cpp:99-105 — null payoff, registerWith.
        super().__init__(_NULL_IRREGULAR_PAYOFF, exercise)
        self._swap: IrregularSwap = swap
        self._settlement_type: IrregularSettlement.Type = delivery
        self._swap.register_with(self)

    # --- Instrument interface --------------------------------------------------

    def is_expired(self) -> bool:
        """# C++ parity: irregularswaption.cpp:107-109.

        C++ consults ``Settings::evaluationDate`` via ``simple_event(...)``;
        PQuantLib defers to ``False`` (the engine computes the expired-day
        NPV), matching the regular Swaption.
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """# C++ parity: irregularswaption.cpp:111-122."""
        self._swap.setup_arguments(args)
        qassert.require(
            isinstance(args, IrregularSwaptionArguments), "wrong argument type"
        )
        assert isinstance(args, IrregularSwaptionArguments)
        args.swap = self._swap
        args.settlement_type = self._settlement_type
        args.exercise = self._exercise
        args.payoff = None

    # --- inspectors ------------------------------------------------------------

    def settlement_type(self) -> IrregularSettlement.Type:
        return self._settlement_type

    def type(self) -> SwapType:
        return self._swap.type()

    def underlying_swap(self) -> IrregularSwap:
        return self._swap


__all__ = [
    "IrregularSettlement",
    "IrregularSwaption",
    "IrregularSwaptionArguments",
]
