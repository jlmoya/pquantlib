"""EverestOption — exotic multi-asset option.

# C++ parity: ql/experimental/exoticoptions/everestoption.{hpp,cpp} (v1.42.1).

Payoff at exercise (under each MC path):

    payoff = (1 + min_i (S_i(T) / S_i(0) - 1) + guarantee) * notional

The pricer also exposes ``yield()`` =
``value / (notional * end_discount) - 1``.

C++ instantiates with ``NullPayoff`` and a user-supplied exercise.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.instrument import Instrument
from pquantlib.instruments.multi_asset_option import (
    MultiAssetOption,
    MultiAssetOptionResults,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class EverestOptionArguments(OptionArguments):
    """Engine arguments for ``EverestOption``.

    # C++ parity: ``EverestOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.notional: float | None = None
        self.guarantee: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.notional is not None, "no notional given")
        qassert.require(self.guarantee is not None, "no guarantee given")
        assert self.notional is not None
        qassert.require(self.notional != 0.0, "null notional given")


class EverestOptionResults(MultiAssetOptionResults):
    """Results carrier for ``EverestOption``.

    # C++ parity: ``EverestOption::results : MultiAssetOption::results +
    # yield``. The engine fills ``value`` and ``yield``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.yield_: float | None = None

    def reset(self) -> None:
        super().reset()
        self.yield_ = None


class EverestOption(MultiAssetOption):
    """Everest option on N underlying assets.

    # C++ parity: ``EverestOption(Real notional, Rate guarantee,
    #               const ext::shared_ptr<Exercise>&)``.

    Args:
        notional: Face notional ``N``.
        guarantee: Yield guarantee ``g`` (additive to ``min_yield``).
        exercise: Exercise (typically ``EuropeanExercise``).
    """

    def __init__(
        self,
        notional: float,
        guarantee: float,
        exercise: Exercise,
    ) -> None:
        super().__init__(NullPayoff(), exercise)
        self._notional: float = notional
        self._guarantee: float = guarantee
        self._yield: float | None = None

    def notional(self) -> float:
        return self._notional

    def guarantee(self) -> float:
        return self._guarantee

    def yield_(self) -> float:
        """Cached yield from the most recent ``calculate()``.

        # C++ parity: ``EverestOption::yield`` — also calls ``calculate``.
        Calling this triggers ``performCalculations`` if necessary.
        """
        self.calculate()
        qassert.require(self._yield is not None, "yield not provided")
        assert self._yield is not None
        return self._yield

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy payoff + exercise + notional + guarantee.

        # C++ parity: ``EverestOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, EverestOptionArguments),
            "wrong argument type (expected EverestOptionArguments)",
        )
        assert isinstance(args, EverestOptionArguments)
        args.notional = self._notional
        args.guarantee = self._guarantee

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull value + yield from the engine results.

        # C++ parity: ``EverestOption::fetchResults``.
        """
        Instrument.fetch_results(self, results)
        qassert.require(
            isinstance(results, EverestOptionResults),
            "no results returned from pricing engine",
        )
        assert isinstance(results, EverestOptionResults)
        self._yield = results.yield_


__all__ = [
    "EverestOption",
    "EverestOptionArguments",
    "EverestOptionResults",
]
