"""Partial-time barrier option (one asset).

# C++ parity: ql/instruments/partialtimebarrieroption.{hpp,cpp}
# (v1.42.1).

A partial-time barrier option is a barrier option in which the barrier
is only monitored over a sub-window of the option's lifetime:

* ``PartialBarrierRange.Start`` — monitor from inception until the
  *cover event date*; after that the option becomes vanilla.
* ``PartialBarrierRange.EndB1`` — monitor from the cover event date
  to expiry; trigger knock-out only if the barrier is hit or crossed
  from either side during the monitoring window, regardless of the
  underlying value at the start of monitoring.
* ``PartialBarrierRange.EndB2`` — monitor from the cover event date
  to expiry; immediately trigger knock-out if the underlying is on
  the wrong side of the barrier at the *start* of monitoring.

Reference: Heynen & Kat 1994, "Partial barrier options", Journal of
Financial Engineering 3(3), 253-274. Closed-form via the bivariate
normal CDF (Drezner 1978 / We04).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import BarrierType
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class PartialBarrierRange(IntEnum):
    """Time-range discriminant for partial-time barrier monitoring.

    # C++ parity: ``struct PartialBarrier { enum Range { Start = 0,
    # EndB1 = 2, EndB2 = 3 }; };`` (partialtimebarrieroption.hpp).
    # C++ assigns Start=0, EndB1=2, EndB2=3 (note the skip at 1, which
    # historically held an unimplemented range). We preserve the exact
    # integer mapping to stay parity-safe against C++ probe values.
    """

    Start = 0
    EndB1 = 2
    EndB2 = 3


class PartialTimeBarrierOptionArguments(OptionArguments):
    """Engine arguments for partial-time barrier options.

    # C++ parity: ``PartialTimeBarrierOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.barrier_type: BarrierType | None = None
        self.barrier_range: PartialBarrierRange | None = None
        self.barrier: float | None = None
        self.rebate: float | None = None
        self.cover_event_date: Date | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.barrier_type is not None, "no barrier type given")
        qassert.require(self.barrier_range is not None, "no barrier range given")
        qassert.require(self.barrier is not None, "no barrier value given")
        qassert.require(self.rebate is not None, "no rebate given")
        qassert.require(
            self.cover_event_date is not None, "no cover event date given"
        )
        assert self.barrier is not None
        assert self.rebate is not None
        qassert.require(
            self.barrier > 0.0, f"barrier ({self.barrier}) must be positive"
        )
        qassert.require(
            self.rebate >= 0.0, f"negative rebate ({self.rebate}) not allowed"
        )


class PartialTimeBarrierOption(OneAssetOption):
    """Partial-time barrier option (one asset).

    # C++ parity: ``PartialTimeBarrierOption(barrierType, barrierRange,
    # barrier, rebate, coverEventDate, payoff, exercise)``.
    """

    def __init__(
        self,
        barrier_type: BarrierType,
        barrier_range: PartialBarrierRange,
        barrier: float,
        rebate: float,
        cover_event_date: Date,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._barrier_type: BarrierType = barrier_type
        self._barrier_range: PartialBarrierRange = barrier_range
        self._barrier: float = barrier
        self._rebate: float = rebate
        self._cover_event_date: Date = cover_event_date

    def barrier_type(self) -> BarrierType:
        return self._barrier_type

    def barrier_range(self) -> PartialBarrierRange:
        return self._barrier_range

    def barrier(self) -> float:
        return self._barrier

    def rebate(self) -> float:
        return self._rebate

    def cover_event_date(self) -> Date:
        return self._cover_event_date

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, PartialTimeBarrierOptionArguments),
            "wrong argument type (expected PartialTimeBarrierOptionArguments)",
        )
        assert isinstance(args, PartialTimeBarrierOptionArguments)
        args.barrier_type = self._barrier_type
        args.barrier_range = self._barrier_range
        args.barrier = self._barrier
        args.rebate = self._rebate
        args.cover_event_date = self._cover_event_date


__all__ = [
    "PartialBarrierRange",
    "PartialTimeBarrierOption",
    "PartialTimeBarrierOptionArguments",
]
