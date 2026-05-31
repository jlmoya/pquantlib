"""ForwardVanillaOption — forward-starting (strike-resetting) vanilla.

# C++ parity: ql/instruments/forwardvanillaoption.{hpp,cpp} (v1.42.1).

A forward-start option fixes its strike to ``moneyness * S(reset_date)``
at a future ``reset_date``, then pays a plain-vanilla payoff at maturity.
The C++ ``ForwardOptionArguments<ArgumentsType>`` template is rendered as
a single ``ForwardOptionArguments`` carrier extending ``OptionArguments``
with ``moneyness`` + ``reset_date``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class ForwardOptionArguments(OptionArguments):
    """Engine arguments for a forward-starting option.

    # C++ parity: ``ForwardOptionArguments<ArgumentsType>``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.moneyness: float | None = None
        self.reset_date: Date | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.moneyness is not None, "null moneyness given")
        assert self.moneyness is not None
        qassert.require(self.moneyness > 0.0, "negative or zero moneyness given")
        qassert.require(self.reset_date is not None, "null reset date given")
        assert self.reset_date is not None
        qassert.require(self.reset_date != Date(), "null reset date given")
        today = ObservableSettings().evaluation_date
        if today is not None:
            qassert.require(self.reset_date >= today, "reset date in the past")
        assert self.exercise is not None
        qassert.require(
            self.exercise.last_date() > self.reset_date,
            "reset date later or equal to maturity",
        )


class ForwardVanillaOption(OneAssetOption):
    """Forward-starting vanilla option.

    # C++ parity: ``ForwardVanillaOption``.

    Args:
        moneyness: strike-reset multiplier (strike = moneyness * S(reset)).
        reset_date: date at which the strike is fixed.
        payoff: the (plain-vanilla) payoff carried to maturity.
        exercise: the European exercise.
    """

    def __init__(
        self,
        moneyness: float,
        reset_date: Date,
        payoff: StrikedTypePayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)
        self._moneyness: float = moneyness
        self._reset_date: Date = reset_date

    def is_expired(self) -> bool:
        # See VanillaOption.is_expired — Settings.evaluation_date is a
        # Phase 1 carve-out, so engines always run.
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ForwardOptionArguments), "wrong argument type"
        )
        assert isinstance(args, ForwardOptionArguments)
        args.moneyness = self._moneyness
        args.reset_date = self._reset_date


__all__ = ["ForwardOptionArguments", "ForwardVanillaOption"]
