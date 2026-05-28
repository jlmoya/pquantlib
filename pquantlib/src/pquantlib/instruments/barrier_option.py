"""Barrier option (one asset).

# C++ parity: ql/instruments/barrieroption.{hpp,cpp} +
# ql/instruments/barriertype.hpp (v1.42.1).

C++ design:

* ``Barrier::Type`` — IntEnum (DownIn / UpIn / DownOut / UpOut).
* ``BarrierOption`` — one-asset option that knocks in/out when the
  underlying first touches the barrier. ``rebate`` is paid at expiry
  if the option is knocked out (or never knocked in for KI variants).

C++'s ``impliedVolatility`` helper is deferred — it requires an FD
engine for the inverse-solve (FdBlackScholesBarrierEngine).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class BarrierType(IntEnum):
    """Barrier knock-in / knock-out discriminant.

    # C++ parity: ``struct Barrier { enum Type { DownIn, UpIn, DownOut,
    # UpOut }; };`` (ql/instruments/barriertype.hpp). Integer values
    # match C++ declaration order: DownIn=0, UpIn=1, DownOut=2, UpOut=3.
    """

    DownIn = 0
    UpIn = 1
    DownOut = 2
    UpOut = 3


class BarrierOptionArguments(OptionArguments):
    """Engine arguments for barrier options.

    # C++ parity: ``BarrierOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.barrier_type: BarrierType | None = None
        self.barrier: float | None = None
        self.rebate: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.barrier_type is not None, "no barrier type given")
        qassert.require(self.barrier is not None, "no barrier value given")
        qassert.require(self.rebate is not None, "no rebate given")
        assert self.barrier is not None
        assert self.rebate is not None
        qassert.require(self.barrier > 0.0, f"barrier ({self.barrier}) must be positive")
        qassert.require(self.rebate >= 0.0, f"negative rebate ({self.rebate}) not allowed")


class BarrierOption(OneAssetOption):
    """Barrier option (one asset) with optional rebate.

    # C++ parity: ``BarrierOption(barrierType, barrier, rebate, payoff,
    # exercise)``. The ``impliedVolatility`` helper is deferred (needs
    # an FD engine for the inverse-solve).
    """

    def __init__(
        self,
        barrier_type: BarrierType,
        barrier: float,
        rebate: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._barrier_type: BarrierType = barrier_type
        self._barrier: float = barrier
        self._rebate: float = rebate

    def barrier_type(self) -> BarrierType:
        return self._barrier_type

    def barrier(self) -> float:
        return self._barrier

    def rebate(self) -> float:
        return self._rebate

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, BarrierOptionArguments),
            "wrong argument type (expected BarrierOptionArguments)",
        )
        assert isinstance(args, BarrierOptionArguments)
        args.barrier_type = self._barrier_type
        args.barrier = self._barrier
        args.rebate = self._rebate


__all__ = [
    "BarrierOption",
    "BarrierOptionArguments",
    "BarrierType",
]
