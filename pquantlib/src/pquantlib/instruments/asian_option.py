"""Asian options — continuous + discrete averaging.

# C++ parity: ql/instruments/asianoption.{hpp,cpp} +
# ql/instruments/averagetype.hpp (v1.42.1).

C++ design:

* ``Average::Type`` — IntEnum (Arithmetic / Geometric).
* ``ContinuousAveragingAsianOption`` — one-asset option whose payoff
  is realized against the continuous-time geometric or arithmetic
  average of the underlying over [start_date, exercise].
* ``DiscreteAveragingAsianOption`` — same idea but with discrete
  fixings (running accumulator + past fixings + future fixing dates).

The Python port mirrors the C++ ``setupArguments`` plumbing — the
nested ``arguments`` classes (one per instrument) carry the extra
fields the analytic engines read.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class AverageType(IntEnum):
    """Averaging convention.

    # C++ parity: ``struct Average { enum Type { Arithmetic, Geometric }; };``
    # (ql/instruments/averagetype.hpp). Integer values match C++:
    # Arithmetic=0, Geometric=1.
    """

    Arithmetic = 0
    Geometric = 1


class ContinuousAveragingAsianOptionArguments(OptionArguments):
    """Engine arguments for continuous-averaging Asian options.

    # C++ parity: ``ContinuousAveragingAsianOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.average_type: AverageType | None = None
        # Default to ``Date()`` (C++ uses Null Date for unseasoned).
        self.start_date: Date = Date()

    def validate(self) -> None:
        super().validate()
        qassert.require(self.average_type is not None, "no average type given")


class DiscreteAveragingAsianOptionArguments(OptionArguments):
    """Engine arguments for discrete-averaging Asian options.

    # C++ parity: ``DiscreteAveragingAsianOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.average_type: AverageType | None = None
        self.running_accumulator: float | None = None
        self.past_fixings: int | None = None
        self.fixing_dates: list[Date] = []

    def validate(self) -> None:
        super().validate()
        qassert.require(self.average_type is not None, "no average type given")
        qassert.require(
            self.running_accumulator is not None, "no running accumulator given"
        )
        qassert.require(self.past_fixings is not None, "no past fixings given")
        qassert.require(len(self.fixing_dates) > 0, "no fixing dates given")


class ContinuousAveragingAsianOption(OneAssetOption):
    """Continuous-averaging Asian option (one asset).

    # C++ parity: ``ContinuousAveragingAsianOption(averageType, payoff,
    # exercise)`` (unseasoned) and ``(averageType, startDate, payoff,
    # exercise)`` (seasoned). The Python port collapses both forms via
    # an optional ``start_date`` — pass it for seasoned options.
    """

    def __init__(
        self,
        average_type: AverageType,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
        *,
        start_date: Date | None = None,
    ) -> None:
        super().__init__(payoff, exercise)
        self._average_type: AverageType = average_type
        self._start_date: Date = start_date if start_date is not None else Date()

    def average_type(self) -> AverageType:
        return self._average_type

    def start_date(self) -> Date:
        return self._start_date

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ContinuousAveragingAsianOptionArguments),
            "wrong argument type (expected ContinuousAveragingAsianOptionArguments)",
        )
        assert isinstance(args, ContinuousAveragingAsianOptionArguments)
        args.average_type = self._average_type
        args.start_date = self._start_date


class DiscreteAveragingAsianOption(OneAssetOption):
    """Discrete-averaging Asian option (one asset).

    # C++ parity: ``DiscreteAveragingAsianOption(averageType,
    # runningAccumulator, pastFixings, fixingDates, payoff, exercise)``.
    # The second C++ constructor (allPastFixings vector) is deferred —
    # callers pass ``running_accumulator`` + ``past_fixings`` directly.
    """

    def __init__(
        self,
        average_type: AverageType,
        running_accumulator: float,
        past_fixings: int,
        fixing_dates: Sequence[Date],
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._average_type: AverageType = average_type
        self._running_accumulator: float = running_accumulator
        self._past_fixings: int = past_fixings
        self._fixing_dates: list[Date] = list(fixing_dates)

    def average_type(self) -> AverageType:
        return self._average_type

    def running_accumulator(self) -> float:
        return self._running_accumulator

    def past_fixings(self) -> int:
        return self._past_fixings

    def fixing_dates(self) -> list[Date]:
        return self._fixing_dates

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, DiscreteAveragingAsianOptionArguments),
            "wrong argument type (expected DiscreteAveragingAsianOptionArguments)",
        )
        assert isinstance(args, DiscreteAveragingAsianOptionArguments)
        args.average_type = self._average_type
        args.running_accumulator = self._running_accumulator
        args.past_fixings = self._past_fixings
        args.fixing_dates = list(self._fixing_dates)


__all__ = [
    "AverageType",
    "ContinuousAveragingAsianOption",
    "ContinuousAveragingAsianOptionArguments",
    "DiscreteAveragingAsianOption",
    "DiscreteAveragingAsianOptionArguments",
]
