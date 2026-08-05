"""ComplexChooserOption — chooser with distinct call and put terms.

# C++ parity: ql/instruments/complexchooseroption.{hpp,cpp} (v1.43).

The holder chooses, at ``choosing_date``, whether the contract becomes a
call or a put.  Unlike the simple chooser
(:class:`~pquantlib.instruments.simple_chooser_option.SimpleChooserOption`),
the two legs have **independent** strikes and **independent** expiries.

Internally the instrument is a ``OneAssetOption`` whose payoff is
``PlainVanillaPayoff(Call, strike_call)`` and whose exercise is the
*call* exercise; the put strike / put exercise and the choosing date
travel to the engine through :class:`ComplexChooserOptionArguments` —
exactly as in C++.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
from pquantlib.time.date import Date


class ComplexChooserOptionArguments(OptionArguments):
    """Engine arguments for :class:`ComplexChooserOption`.

    # C++ parity: ``ComplexChooserOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.choosing_date: Date = Date()
        self.strike_call: float | None = None
        self.strike_put: float | None = None
        self.exercise_call: Exercise | None = None
        self.exercise_put: Exercise | None = None

    def validate(self) -> None:
        """# C++ parity: ``ComplexChooserOption::arguments::validate``."""
        super().validate()
        qassert.require(self.choosing_date != Date(), " no choosing date given")
        assert self.exercise_call is not None
        assert self.exercise_put is not None
        qassert.require(
            self.choosing_date < self.exercise_call.last_date(),
            "choosing date later than or equal to Call maturity date",
        )
        qassert.require(
            self.choosing_date < self.exercise_put.last_date(),
            "choosing date later than or equal to Put maturity date",
        )


class ComplexChooserOption(OneAssetOption):
    """Complex chooser option.

    # C++ parity: ``ComplexChooserOption(Date choosingDate, Real strikeCall,
    #               Real strikePut, const ext::shared_ptr<Exercise>& exerciseCall,
    #               ext::shared_ptr<Exercise> exercisePut)``.

    Args:
        choosing_date: Date at which the holder picks call or put. Must be
            strictly before *both* maturities.
        strike_call: Strike of the call leg.
        strike_put: Strike of the put leg.
        exercise_call: Exercise of the call leg (also the base
            ``OneAssetOption`` exercise, per C++).
        exercise_put: Exercise of the put leg.
    """

    def __init__(
        self,
        choosing_date: Date,
        strike_call: float,
        strike_put: float,
        exercise_call: Exercise,
        exercise_put: Exercise,
    ) -> None:
        # C++ builds the base OneAssetOption from the CALL leg only.
        super().__init__(PlainVanillaPayoff(OptionType.Call, strike_call), exercise_call)
        self._choosing_date: Date = choosing_date
        self._strike_call: float = strike_call
        self._strike_put: float = strike_put
        self._exercise_call: Exercise = exercise_call
        self._exercise_put: Exercise = exercise_put

    def is_expired(self) -> bool:
        """Return ``False`` — defers to engine.

        # C++ parity: ``OneAssetOption::isExpired`` consults
        # ``Settings::evaluationDate``. Until Settings is wired, we return
        # False so the engine always runs (same as
        # ``SimpleChooserOption.is_expired`` in this codebase).
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Populate the chooser-specific arguments bundle.

        # C++ parity: ``ComplexChooserOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(isinstance(args, ComplexChooserOptionArguments), "wrong argument type")
        assert isinstance(args, ComplexChooserOptionArguments)
        args.choosing_date = self._choosing_date
        args.strike_call = self._strike_call
        args.strike_put = self._strike_put
        args.exercise_call = self._exercise_call
        args.exercise_put = self._exercise_put


__all__ = ["ComplexChooserOption", "ComplexChooserOptionArguments"]
