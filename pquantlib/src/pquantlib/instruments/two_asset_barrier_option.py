"""TwoAssetBarrierOption — barrier option on two assets.

# C++ parity: ql/instruments/twoassetbarrieroption.{hpp,cpp} (v1.42.1)
# (moved out of ``experimental/exoticoptions`` in QL 1.38, kept as a
# deprecated header there).

The value of the first asset (S1) is compared to the strike to
determine the (call/put) payoff; the value of the second asset (S2)
is monitored to check whether the barrier is hit (up-in / up-out /
down-in / down-out).

Closed-form pricing is by Heynen & Kat (1994), implemented in
``AnalyticTwoAssetBarrierEngine``.  The Python port uses the existing
``BarrierType`` IntEnum from ``instruments.barrier_option``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import BarrierType
from pquantlib.instruments.instrument import Instrument
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class TwoAssetBarrierOptionArguments(OptionArguments):
    """Engine arguments for ``TwoAssetBarrierOption``.

    # C++ parity: ``TwoAssetBarrierOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.barrier_type: BarrierType | None = None
        self.barrier: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.barrier_type is not None, "no barrier type given")
        qassert.require(self.barrier is not None, "no barrier given")


class TwoAssetBarrierOption(Option):
    """Two-asset barrier option (payoff on S1, barrier on S2).

    # C++ parity: ``TwoAssetBarrierOption(Barrier::Type, Real barrier,
    #               const ext::shared_ptr<StrikedTypePayoff>&,
    #               const ext::shared_ptr<Exercise>&)``.

    Args:
        barrier_type: One of ``BarrierType.{DownIn,UpIn,DownOut,UpOut}``.
        barrier: Barrier level on the second asset (S2).
        payoff: ``StrikedTypePayoff`` (call/put) on the first asset (S1).
        exercise: Exercise (typically ``EuropeanExercise``).
    """

    def __init__(
        self,
        barrier_type: BarrierType,
        barrier: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._barrier_type: BarrierType = barrier_type
        self._barrier: float = barrier

    # --- inspectors ------------------------------------------------------

    def barrier_type(self) -> BarrierType:
        return self._barrier_type

    def barrier(self) -> float:
        return self._barrier

    # --- Instrument interface --------------------------------------------

    def is_expired(self) -> bool:
        """Defers to engine — ``Settings.evaluation_date`` is a Phase 1
        carve-out so we never set the instrument to expired here."""
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Inject barrier_type + barrier into engine arguments.

        # C++ parity: ``TwoAssetBarrierOption::setupArguments``.
        """
        Option.setup_arguments(self, args)
        qassert.require(
            isinstance(args, TwoAssetBarrierOptionArguments),
            "wrong argument type (expected TwoAssetBarrierOptionArguments)",
        )
        assert isinstance(args, TwoAssetBarrierOptionArguments)
        args.barrier_type = self._barrier_type
        args.barrier = self._barrier

    def fetch_results(self, results) -> None:  # type: ignore[no-untyped-def]
        """Pull value out of the engine results (no Greeks for this engine).

        # C++ parity: inherits ``Option::fetchResults`` via Instrument.
        """
        Instrument.fetch_results(self, results)


__all__ = ["TwoAssetBarrierOption", "TwoAssetBarrierOptionArguments"]
