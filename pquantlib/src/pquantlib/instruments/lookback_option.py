"""Lookback options (continuous monitoring).

# C++ parity: ql/instruments/lookbackoption.{hpp,cpp} (v1.42.1).

Two concrete variants:

* ``ContinuousFloatingLookbackOption(minmax, FloatingTypePayoff,
  exercise)`` — payoff at exercise = ``S_T - min`` (Call) or
  ``max - S_T`` (Put). ``minmax`` is the realized extremum to date
  (= spot at issuance for an unseasoned option).
* ``ContinuousFixedLookbackOption(minmax, StrikedTypePayoff, exercise)``
  — payoff = ``max(max - K, 0)`` (Call on running max) or
  ``max(K - min, 0)`` (Put on running min).

Partial-time variants and discrete-monitoring lookbacks are deferred
to Phase 6.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import FloatingTypePayoff, StrikedTypePayoff, TypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class ContinuousFloatingLookbackOptionArguments(OptionArguments):
    """Engine arguments for floating-strike continuous lookback.

    # C++ parity: ``ContinuousFloatingLookbackOption::arguments`` —
    # holds the running extremum (min for calls, max for puts).
    """

    def __init__(self) -> None:
        super().__init__()
        self.minmax: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.minmax is not None, "null prior extremum")
        assert self.minmax is not None
        qassert.require(
            self.minmax >= 0.0,
            f"nonnegative prior extremum required: {self.minmax} not allowed",
        )


class ContinuousFixedLookbackOptionArguments(OptionArguments):
    """Engine arguments for fixed-strike continuous lookback.

    # C++ parity: ``ContinuousFixedLookbackOption::arguments`` — same
    # ``minmax`` field as floating but a striked payoff.
    """

    def __init__(self) -> None:
        super().__init__()
        self.minmax: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.minmax is not None, "null prior extremum")
        assert self.minmax is not None
        qassert.require(
            self.minmax >= 0.0,
            f"nonnegative prior extremum required: {self.minmax} not allowed",
        )


class ContinuousFloatingLookbackOption(OneAssetOption):
    """Continuous-monitoring floating-strike lookback option.

    # C++ parity: ``ContinuousFloatingLookbackOption(minmax, payoff,
    # exercise)``. The payoff is a ``FloatingTypePayoff``; the strike is
    # fixed by the realized extremum at exercise.
    """

    def __init__(
        self,
        minmax: float,
        payoff: FloatingTypePayoff,
        exercise: Exercise,
    ) -> None:
        # C++ accepts ``TypePayoff``; floating-type is the only useful
        # subclass for a lookback. Constructor narrows to
        # ``FloatingTypePayoff`` for type safety.
        super().__init__(payoff, exercise)
        self._minmax: float = minmax

    def minmax(self) -> float:
        return self._minmax

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ContinuousFloatingLookbackOptionArguments),
            "wrong argument type (expected ContinuousFloatingLookbackOptionArguments)",
        )
        assert isinstance(args, ContinuousFloatingLookbackOptionArguments)
        args.minmax = self._minmax


class ContinuousFixedLookbackOption(OneAssetOption):
    """Continuous-monitoring fixed-strike lookback option.

    # C++ parity: ``ContinuousFixedLookbackOption(minmax, payoff,
    # exercise)``. The payoff is a ``StrikedTypePayoff`` (typically
    # PlainVanilla); engine valuates against ``max`` (Call) or ``min``
    # (Put) over the lookback window.
    """

    def __init__(
        self,
        minmax: float,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._minmax: float = minmax

    def minmax(self) -> float:
        return self._minmax

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ContinuousFixedLookbackOptionArguments),
            "wrong argument type (expected ContinuousFixedLookbackOptionArguments)",
        )
        assert isinstance(args, ContinuousFixedLookbackOptionArguments)
        args.minmax = self._minmax


# Re-export TypePayoff for callers that want to type the C++-style
# generic exercise + payoff.
_ = TypePayoff


__all__ = [
    "ContinuousFixedLookbackOption",
    "ContinuousFixedLookbackOptionArguments",
    "ContinuousFloatingLookbackOption",
    "ContinuousFloatingLookbackOptionArguments",
]
