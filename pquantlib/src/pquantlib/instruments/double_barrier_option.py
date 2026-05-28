"""Double-barrier option (one asset).

# C++ parity: ql/instruments/doublebarrieroption.{hpp,cpp} +
# ql/instruments/doublebarriertype.hpp (v1.42.1).

C++ design:

* ``DoubleBarrier::Type`` — IntEnum (KnockIn / KnockOut / KIKO /
  KOKI). KIKO = "lower barrier KI, upper KO"; KOKI = "lower barrier
  KO, upper KI" (i.e. one barrier knocks in, the other knocks out).
* ``DoubleBarrierOption`` — one-asset option with a lower and an upper
  barrier. ``rebate`` is paid at expiry if/when the option is knocked
  out (or never knocked in, for KI variants).

C++'s ``impliedVolatility`` helper is deferred — it requires the
``ImpliedVolatilityHelper`` plumbing carved out in Phase 3 for
``VanillaOption``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class DoubleBarrierType(IntEnum):
    """Double-barrier knock-in / knock-out discriminant.

    # C++ parity: ``struct DoubleBarrier { enum Type { KnockIn,
    # KnockOut, KIKO, KOKI }; };`` (ql/instruments/doublebarriertype.hpp).
    # Integer values match C++ declaration order: KnockIn=0, KnockOut=1,
    # KIKO=2, KOKI=3.
    """

    KnockIn = 0
    KnockOut = 1
    KIKO = 2
    KOKI = 3


class DoubleBarrierOptionArguments(OptionArguments):
    """Engine arguments for double-barrier options.

    # C++ parity: ``DoubleBarrierOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.barrier_type: DoubleBarrierType | None = None
        self.barrier_lo: float | None = None
        self.barrier_hi: float | None = None
        self.rebate: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(
            self.barrier_type
            in (
                DoubleBarrierType.KnockIn,
                DoubleBarrierType.KnockOut,
                DoubleBarrierType.KIKO,
                DoubleBarrierType.KOKI,
            ),
            "Invalid barrier type",
        )
        qassert.require(self.barrier_lo is not None, "no low barrier given")
        qassert.require(self.barrier_hi is not None, "no high barrier given")
        qassert.require(self.rebate is not None, "no rebate given")


class DoubleBarrierOption(OneAssetOption):
    """Double-barrier option (one asset) with optional rebate.

    # C++ parity: ``DoubleBarrierOption(barrierType, barrier_lo,
    # barrier_hi, rebate, payoff, exercise)``. The ``impliedVolatility``
    # helper is deferred (same Phase 3 carve-out as VanillaOption).
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
        super().__init__(payoff, exercise)
        self._barrier_type: DoubleBarrierType = barrier_type
        self._barrier_lo: float = barrier_lo
        self._barrier_hi: float = barrier_hi
        self._rebate: float = rebate

    def barrier_type(self) -> DoubleBarrierType:
        return self._barrier_type

    def barrier_lo(self) -> float:
        return self._barrier_lo

    def barrier_hi(self) -> float:
        return self._barrier_hi

    def rebate(self) -> float:
        return self._rebate

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, DoubleBarrierOptionArguments),
            "wrong argument type (expected DoubleBarrierOptionArguments)",
        )
        assert isinstance(args, DoubleBarrierOptionArguments)
        args.barrier_type = self._barrier_type
        args.barrier_lo = self._barrier_lo
        args.barrier_hi = self._barrier_hi
        args.rebate = self._rebate


__all__ = [
    "DoubleBarrierOption",
    "DoubleBarrierOptionArguments",
    "DoubleBarrierType",
]
