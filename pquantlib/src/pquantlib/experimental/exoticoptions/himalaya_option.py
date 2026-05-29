"""HimalayaOption — exotic basket option.

# C++ parity: ql/experimental/exoticoptions/himalayaoption.{hpp,cpp} (v1.42.1).

Description (from the C++ header):

    The payoff of a Himalaya option is computed in the following way:
    Given a basket of N assets, and N time periods, at the end of each
    period the option who performed the best is added to the average
    and then discarded from the basket.  At the end of the N periods
    the option pays the max between the strike and the average of the
    best performers.

C++ instantiates the underlying payoff as ``PlainVanillaPayoff(Call,
strike)`` and the exercise as ``EuropeanExercise(fixingDates.back())``.

Warning: the C++ implementation does not manage seasoned options;
the Python port keeps the same restriction.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.multi_asset_option import MultiAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class HimalayaOptionArguments(OptionArguments):
    """Engine arguments for ``HimalayaOption``.

    # C++ parity: ``HimalayaOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fixing_dates: list[Date] = []

    def validate(self) -> None:
        super().validate()
        qassert.require(len(self.fixing_dates) > 0, "no fixing dates given")


class HimalayaOption(MultiAssetOption):
    """Himalaya basket option.

    # C++ parity: ``HimalayaOption(const std::vector<Date>&, Real strike)``.

    Args:
        fixing_dates: One date per asset-elimination round. The basket
            size N typically equals ``len(fixing_dates)``; at the i-th
            fixing the best remaining performer is added to the running
            average and removed from the pool.
        strike: Strike of the embedded ``PlainVanillaPayoff(Call,
            strike)``.
    """

    def __init__(self, fixing_dates: list[Date], strike: float) -> None:
        qassert.require(len(fixing_dates) > 0, "no fixing dates given")
        payoff = PlainVanillaPayoff(OptionType.Call, strike)
        exercise = EuropeanExercise(fixing_dates[-1])
        super().__init__(payoff, exercise)
        self._fixing_dates: list[Date] = list(fixing_dates)

    def fixing_dates(self) -> list[Date]:
        """Sequence of fixing/elimination dates."""
        return list(self._fixing_dates)

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy payoff + exercise + fixing dates into the engine arguments.

        # C++ parity: ``HimalayaOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, HimalayaOptionArguments),
            "wrong argument type (expected HimalayaOptionArguments)",
        )
        assert isinstance(args, HimalayaOptionArguments)
        args.fixing_dates = list(self._fixing_dates)


__all__ = ["HimalayaOption", "HimalayaOptionArguments"]
