"""PagodaOption — roofed Asian on a portfolio of N assets.

# C++ parity: ql/experimental/exoticoptions/pagodaoption.{hpp,cpp} (v1.42.1).

Payoff at exercise (under each MC path):

    payoff = fraction * max(0, min(roof, sum_{i, j} S_j(t_i)/S_j(t_{i-1}) - 1))

(C++ ``PagodaMultiPathPricer`` accumulates over fixings + assets).

C++ instantiates the underlying payoff as ``NullPayoff`` and the
exercise as ``EuropeanExercise(fixingDates.back())``.

Warning: seasoned options are not handled (matches C++).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.multi_asset_option import MultiAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class PagodaOptionArguments(OptionArguments):
    """Engine arguments for ``PagodaOption``.

    # C++ parity: ``PagodaOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fixing_dates: list[Date] = []
        self.roof: float | None = None
        self.fraction: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(len(self.fixing_dates) > 0, "no fixingDates given")
        qassert.require(self.roof is not None, "no roof given")
        qassert.require(self.fraction is not None, "no fraction given")


class PagodaOption(MultiAssetOption):
    """Roofed Asian on N assets.

    # C++ parity: ``PagodaOption(const std::vector<Date>&, Real roof,
    #               Real fraction)``.

    Args:
        fixing_dates: Performance-fixing dates (length ``M``); the
            engine accumulates ``S_j(t_i)/S_j(t_{i-1}) - 1`` over
            ``i = 1..M-1`` (and over all assets ``j``), averages by
            asset count, and applies the roof + fraction.
        roof: Roof cap on the cumulative performance.
        fraction: Multiplicative payout fraction.
    """

    def __init__(
        self,
        fixing_dates: list[Date],
        roof: float,
        fraction: float,
    ) -> None:
        qassert.require(len(fixing_dates) > 0, "no fixing dates given")
        exercise = EuropeanExercise(fixing_dates[-1])
        super().__init__(NullPayoff(), exercise)
        self._fixing_dates: list[Date] = list(fixing_dates)
        self._roof: float = roof
        self._fraction: float = fraction

    def fixing_dates(self) -> list[Date]:
        return list(self._fixing_dates)

    def roof(self) -> float:
        return self._roof

    def fraction(self) -> float:
        return self._fraction

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy payoff + exercise + fixing dates + roof + fraction.

        # C++ parity: ``PagodaOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, PagodaOptionArguments),
            "wrong argument type (expected PagodaOptionArguments)",
        )
        assert isinstance(args, PagodaOptionArguments)
        args.fixing_dates = list(self._fixing_dates)
        args.roof = self._roof
        args.fraction = self._fraction


__all__ = ["PagodaOption", "PagodaOptionArguments"]
