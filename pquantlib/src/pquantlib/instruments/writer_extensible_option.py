"""WriterExtensibleOption — writer extends maturity if OTM at t1.

# C++ parity: ql/instruments/writerextensibleoption.{hpp,cpp} (v1.42.1).

A writer-extensible option pays:

* The standard PlainVanilla payoff at ``t1`` (using ``payoff1``)
  if that payoff is ITM.
* Otherwise the *writer* extends to ``t2`` and the new payoff is
  ``payoff2`` (typically with the same option type but a different
  strike).

Interpretation: the writer "saves" a bad outcome at t1 by extending
the option, while keeping the original premium received.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOption
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class WriterExtensibleOptionArguments(OptionArguments):
    """Engine arguments for the writer-extensible option.

    # C++ parity: ``WriterExtensibleOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.payoff2: PlainVanillaPayoff | None = None
        self.exercise2: Exercise | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.payoff2 is not None, "no second payoff given")
        qassert.require(self.exercise2 is not None, "no second exercise given")
        assert self.exercise is not None
        assert self.exercise2 is not None
        qassert.require(
            self.exercise2.last_date() > self.exercise.last_date(),
            "second exercise date is not later than the first",
        )


class WriterExtensibleOption(OneAssetOption):
    """Writer-extensible option (Haug 2007).

    # C++ parity: ``WriterExtensibleOption``.
    """

    def __init__(
        self,
        payoff1: PlainVanillaPayoff,
        exercise1: Exercise,
        payoff2: PlainVanillaPayoff,
        exercise2: Exercise,
    ) -> None:
        super().__init__(payoff1, exercise1)
        self._payoff2: PlainVanillaPayoff = payoff2
        self._exercise2: Exercise = exercise2

    def payoff2(self) -> PlainVanillaPayoff:
        return self._payoff2

    def exercise2(self) -> Exercise:
        return self._exercise2

    def is_expired(self) -> bool:
        """Return ``False`` until Settings.evaluationDate is wired."""
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Propagate the second leg into the arguments bundle.

        # C++ parity: ``WriterExtensibleOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, WriterExtensibleOptionArguments),
            "wrong arguments type",
        )
        assert isinstance(args, WriterExtensibleOptionArguments)
        args.payoff2 = self._payoff2
        args.exercise2 = self._exercise2


__all__ = ["WriterExtensibleOption", "WriterExtensibleOptionArguments"]
