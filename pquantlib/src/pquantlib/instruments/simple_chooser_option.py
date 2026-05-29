"""SimpleChooserOption — Rubinstein simple chooser option.

# C++ parity: ql/instruments/simplechooseroption.{hpp,cpp} (v1.42.1).

A simple chooser option gives the holder the right, at a future
choosing date prior to exercise, to choose whether the option should
be a call or a put. The exercise date and strike are the same for
both legs.

Internally the instrument is a ``OneAssetOption`` whose payoff is a
``PlainVanillaPayoff(Call, strike)``; the engine treats it specifically
as a chooser by reading the ``choosing_date`` from the dedicated
arguments bundle.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class SimpleChooserOptionArguments(OptionArguments):
    """Engine arguments for the simple chooser option.

    # C++ parity: ``SimpleChooserOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.choosing_date: Date = Date()

    def validate(self) -> None:
        super().validate()
        qassert.require(self.choosing_date != Date(), " no choosing date given")
        assert self.exercise is not None
        qassert.require(
            self.choosing_date < self.exercise.last_date(),
            "choosing date later than or equal to maturity date",
        )


class SimpleChooserOption(OneAssetOption):
    """Simple chooser option (Rubinstein 1991).

    # C++ parity: ``SimpleChooserOption(choosingDate, strike, exercise)``.
    """

    def __init__(
        self,
        choosing_date: Date,
        strike: float,
        exercise: Exercise,
    ) -> None:
        # Per C++: the base OneAssetOption holds a Call payoff with the
        # chooser strike; the chooser nature is fully captured by the
        # engine + choosing_date argument.
        super().__init__(PlainVanillaPayoff(OptionType.Call, strike), exercise)
        self._choosing_date: Date = choosing_date

    def choosing_date(self) -> Date:
        return self._choosing_date

    def is_expired(self) -> bool:
        """Return ``False`` — defers to engine.

        # C++ parity: ``OneAssetOption::isExpired`` defers to
        # ``Settings.evaluationDate``. Until Settings is wired,
        # we return False so the engine always runs (same as
        # VanillaOption.is_expired in this codebase).
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Populate the chooser-specific arguments bundle.

        # C++ parity: ``SimpleChooserOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, SimpleChooserOptionArguments),
            "wrong argument type",
        )
        assert isinstance(args, SimpleChooserOptionArguments)
        args.choosing_date = self._choosing_date


__all__ = ["SimpleChooserOption", "SimpleChooserOptionArguments"]
