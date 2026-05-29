"""CompoundOption — option on an option.

# C++ parity: ql/instruments/compoundoption.{hpp,cpp} (v1.42.1).

A compound option is an option whose underlying is itself an option:
``CompoundOption(mother_payoff, mother_exercise, daughter_payoff,
daughter_exercise)``.

* The **mother** option is the compound option proper. Its payoff is a
  ``StrikedTypePayoff`` on the *value* of the daughter option at the
  mother's expiry.
* The **daughter** option is the underlying option whose value drives
  the mother payoff.

Closed-form pricing (Wystup 2002) requires
``mother_exercise.last_date() <= daughter_exercise.last_date()``.

The Python port mirrors the C++ ``OneAssetOption`` inheritance: the
mother payoff + exercise live on the base; the daughter payoff +
exercise are extra fields added to a dedicated arguments bundle
``CompoundOptionArguments`` (engine-side) and ferried via
``setup_arguments``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class CompoundOptionArguments(OptionArguments):
    """Engine argument bundle for compound options.

    # C++ parity: ``CompoundOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.daughter_payoff: StrikedTypePayoff | None = None
        self.daughter_exercise: Exercise | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(
            self.daughter_payoff is not None,
            "no payoff given for underlying option",
        )
        qassert.require(
            self.daughter_exercise is not None,
            "no exercise given for underlying option",
        )
        assert self.daughter_exercise is not None
        assert self.exercise is not None
        qassert.require(
            self.exercise.last_date() <= self.daughter_exercise.last_date(),
            "maturity of compound option exceeds maturity of underlying option",
        )


class CompoundOption(OneAssetOption):
    """Compound option (option on option) on a single asset.

    # C++ parity: ``CompoundOption`` (compoundoption.hpp:35-51 v1.42.1).
    """

    def __init__(
        self,
        mother_payoff: StrikedTypePayoff,
        mother_exercise: Exercise,
        daughter_payoff: StrikedTypePayoff,
        daughter_exercise: Exercise,
    ) -> None:
        super().__init__(mother_payoff, mother_exercise)
        self._daughter_payoff: StrikedTypePayoff = daughter_payoff
        self._daughter_exercise: Exercise = daughter_exercise

    # --- inspectors -----------------------------------------------------

    def daughter_payoff(self) -> StrikedTypePayoff:
        return self._daughter_payoff

    def daughter_exercise(self) -> Exercise:
        return self._daughter_exercise

    def is_expired(self) -> bool:
        """Return ``False`` — defers to engine.

        # C++ parity: ``OneAssetOption::isExpired`` defers to
        # ``Settings.evaluationDate``. Until Settings is wired,
        # we return False so the engine always runs (same as
        # VanillaOption.is_expired in this codebase).
        """
        return False

    # --- argument setup -------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Populate the engine arguments bundle with daughter info.

        # C++ parity: ``CompoundOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, CompoundOptionArguments),
            "wrong argument type",
        )
        assert isinstance(args, CompoundOptionArguments)
        args.daughter_payoff = self._daughter_payoff
        args.daughter_exercise = self._daughter_exercise


__all__ = ["CompoundOption", "CompoundOptionArguments"]
