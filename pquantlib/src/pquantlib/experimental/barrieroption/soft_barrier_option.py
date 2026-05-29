"""Soft-barrier option (one asset).

# C++ parity: ql/instruments/softbarrieroption.{hpp,cpp} (v1.42.1).

A soft-barrier option gets knocked in/out *proportionally* over a
barrier range, instead of being knocked in/out in full at a hard
barrier. Equivalent to a continuum of standard barriers averaged over
the band ``[barrier_lo, barrier_hi]``.

C++ also exposes an ``impliedVolatility`` helper that does an inverse
solve via the analytic engine; we don't port that yet because the
companion ``ImpliedVolatilityHelper`` plumbing was carved out for
``VanillaOption`` in Phase 3.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import BarrierType
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class SoftBarrierOptionArguments(OptionArguments):
    """Engine arguments for soft-barrier options.

    # C++ parity: ``SoftBarrierOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.barrier_type: BarrierType | None = None
        self.barrier_lo: float | None = None
        self.barrier_hi: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.barrier_type is not None, "no barrier type given")
        qassert.require(self.barrier_lo is not None, "no low barrier given")
        qassert.require(self.barrier_hi is not None, "no high barrier given")
        assert self.barrier_lo is not None
        assert self.barrier_hi is not None
        qassert.require(
            self.barrier_lo > 0.0,
            f"low barrier ({self.barrier_lo}) must be positive",
        )
        qassert.require(
            self.barrier_hi > 0.0,
            f"high barrier ({self.barrier_hi}) must be positive",
        )
        qassert.require(
            self.barrier_hi >= self.barrier_lo,
            "upper barrier must be >= lower barrier",
        )


class SoftBarrierOption(OneAssetOption):
    """Soft-barrier european option on a single asset.

    # C++ parity: ``SoftBarrierOption(barrierType, barrier_lo,
    # barrier_hi, payoff, exercise)``. Only European exercise is
    # supported (C++ ``SoftBarrierOption`` is documented as such).
    """

    def __init__(
        self,
        barrier_type: BarrierType,
        barrier_lo: float,
        barrier_hi: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._barrier_type: BarrierType = barrier_type
        self._barrier_lo: float = barrier_lo
        self._barrier_hi: float = barrier_hi

    def barrier_type(self) -> BarrierType:
        return self._barrier_type

    def barrier_lo(self) -> float:
        return self._barrier_lo

    def barrier_hi(self) -> float:
        return self._barrier_hi

    def is_expired(self) -> bool:
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, SoftBarrierOptionArguments),
            "wrong argument type (expected SoftBarrierOptionArguments)",
        )
        assert isinstance(args, SoftBarrierOptionArguments)
        args.barrier_type = self._barrier_type
        args.barrier_lo = self._barrier_lo
        args.barrier_hi = self._barrier_hi


__all__ = ["SoftBarrierOption", "SoftBarrierOptionArguments"]
