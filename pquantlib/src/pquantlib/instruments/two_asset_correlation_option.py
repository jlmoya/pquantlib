"""TwoAssetCorrelationOption — two-asset correlation option.

# C++ parity: ql/instruments/twoassetcorrelationoption.{hpp,cpp} (v1.42.1).

This option pays a payoff based on the value at exercise of the
second asset and its corresponding strike (``X2``), but only if the
first instrument is also in the money with respect to its own strike
(``X1``); if not, the payoff is 0.

Closed-form: Zhang (from Haug, "Option Pricing Formulas").
Implementation in ``AnalyticTwoAssetCorrelationEngine``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.multi_asset_option import MultiAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class TwoAssetCorrelationOptionArguments(OptionArguments):
    """Engine arguments for ``TwoAssetCorrelationOption``.

    # C++ parity: ``TwoAssetCorrelationOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.x2: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.x2 is not None, "no X2 given")


class TwoAssetCorrelationOption(MultiAssetOption):
    """Two-asset correlation option.

    # C++ parity: ``TwoAssetCorrelationOption(Option::Type, Real strike1,
    #               Real strike2, const ext::shared_ptr<Exercise>&)``.

    Args:
        option_type: ``OptionType.Call`` or ``OptionType.Put`` — applied
            to *both* the trigger condition on S1 (vs strike1) and the
            payoff on S2 (vs strike2).
        strike1: Strike for the in-the-money condition on the first asset.
        strike2: Strike for the payoff on the second asset.
        exercise: Exercise (typically ``EuropeanExercise``).
    """

    def __init__(
        self,
        option_type: OptionType,
        strike1: float,
        strike2: float,
        exercise: Exercise,
    ) -> None:
        # C++ instantiates the parent ``MultiAssetOption`` with a
        # ``PlainVanillaPayoff(type, strike1)``.
        payoff = PlainVanillaPayoff(option_type, strike1)
        super().__init__(payoff, exercise)
        self._x2: float = strike2

    # --- inspectors ------------------------------------------------------

    def strike2(self) -> float:
        """Strike on the second asset (X2)."""
        return self._x2

    # --- Instrument interface --------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Inject X2 into engine arguments.

        # C++ parity: ``TwoAssetCorrelationOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, TwoAssetCorrelationOptionArguments),
            "wrong argument type (expected TwoAssetCorrelationOptionArguments)",
        )
        assert isinstance(args, TwoAssetCorrelationOptionArguments)
        args.x2 = self._x2


__all__ = [
    "TwoAssetCorrelationOption",
    "TwoAssetCorrelationOptionArguments",
]
