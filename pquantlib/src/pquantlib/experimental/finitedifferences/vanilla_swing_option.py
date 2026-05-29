"""VanillaSwingOption — multi-exercise option with min/max-rights cap.

# C++ parity: ql/instruments/vanillaswingoption.{hpp,cpp} (v1.42.1).

A swing option is a Bermudan-style instrument granting the holder a
range ``[min_exercise_rights, max_exercise_rights]`` of exercise
events across the swing-exercise date list. At each event the holder
collects the underlying payoff (a :class:`StrikedTypePayoff`).

Used by :class:`FdSimpleExtOUJumpSwingEngine` (W5-B scaffold).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.swing_exercise import SwingExercise
from pquantlib.instruments.instrument import Instrument
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class VanillaSwingOptionArguments(OptionArguments):
    """Engine arguments for :class:`VanillaSwingOption`.

    # C++ parity: ``VanillaSwingOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.min_exercise_rights: int | None = None
        self.max_exercise_rights: int | None = None

    def validate(self) -> None:
        # C++ parity: ``VanillaSwingOption::arguments::validate()``.
        qassert.require(self.payoff is not None, "no payoff given")
        qassert.require(self.exercise is not None, "no exercise given")
        qassert.require(
            self.min_exercise_rights is not None
            and self.max_exercise_rights is not None
            and self.min_exercise_rights <= self.max_exercise_rights,
            "minExerciseRights <= maxExerciseRights",
        )
        assert self.max_exercise_rights is not None
        assert self.exercise is not None
        qassert.require(
            len(self.exercise.dates()) >= self.max_exercise_rights,
            "number of exercise rights exceeds number of exercise dates",
        )


class VanillaSwingOption(Option):
    """Vanilla swing option.

    # C++ parity: ``class VanillaSwingOption : public OneAssetOption``.
    """

    def __init__(
        self,
        payoff: StrikedTypePayoff,
        exercise: SwingExercise,
        min_exercise_rights: int,
        max_exercise_rights: int,
    ) -> None:
        super().__init__(payoff, exercise)
        self._min_exercise_rights: int = int(min_exercise_rights)
        self._max_exercise_rights: int = int(max_exercise_rights)

    def min_exercise_rights(self) -> int:
        return self._min_exercise_rights

    def max_exercise_rights(self) -> int:
        return self._max_exercise_rights

    def is_expired(self) -> bool:
        """Always returns ``False`` — deferred to engine."""
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy swing-option parameters into engine arguments.

        # C++ parity: ``VanillaSwingOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, VanillaSwingOptionArguments),
            "wrong argument type (expected VanillaSwingOptionArguments)",
        )
        assert isinstance(args, VanillaSwingOptionArguments)
        args.min_exercise_rights = self._min_exercise_rights
        args.max_exercise_rights = self._max_exercise_rights

    def fetch_results(self, results: PricingEngineResults) -> None:
        Instrument.fetch_results(self, results)


__all__ = ["VanillaSwingOption", "VanillaSwingOptionArguments"]
