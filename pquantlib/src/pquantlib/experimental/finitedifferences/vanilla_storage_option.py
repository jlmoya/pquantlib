"""VanillaStorageOption — simple gas-storage option.

# C++ parity: ql/instruments/vanillastorageoption.{hpp,cpp} (v1.42.1).

A storage option lets the holder inject or withdraw gas (the
"working gas") between a minimum and maximum total capacity at a
fixed maximum change-rate per exercise instant, paying the
spot-gas price at each exercise. The instrument is parameterised
by:

* ``capacity`` — maximum total stored amount.
* ``load`` — initial fill level at issue.
* ``change_rate`` — max amount that can be injected/withdrawn per
  exercise event.

The C++ embedded payoff is a :class:`NullPayoff` — the engine
computes the spot value from the underlying process directly. The
exercise is required to be Bermudan (a fixed list of dates).

Used by :class:`FdSimpleExtOUStorageEngine` (W5-B scaffold).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import BermudanExercise
from pquantlib.instruments.instrument import Instrument
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class VanillaStorageOptionArguments(OptionArguments):
    """Engine arguments for :class:`VanillaStorageOption`.

    # C++ parity: ``VanillaStorageOption::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.capacity: float | None = None
        self.load: float | None = None
        self.change_rate: float | None = None

    def validate(self) -> None:
        # C++ parity: ``VanillaStorageOption::arguments::validate()``.
        qassert.require(self.payoff is not None, "no payoff given")
        qassert.require(self.exercise is not None, "no exercise given")
        qassert.require(
            self.capacity is not None
            and self.capacity > 0.0
            and self.change_rate is not None
            and self.change_rate > 0.0
            and self.load is not None
            and self.load >= 0.0,
            "positive capacity, load and change rate required",
        )
        assert self.capacity is not None
        assert self.load is not None
        assert self.change_rate is not None
        qassert.require(
            self.load <= self.capacity and self.change_rate <= self.capacity,
            "illegal values load of changeRate",
        )


class VanillaStorageOption(Option):
    """Vanilla gas-storage option.

    # C++ parity: ``class VanillaStorageOption : public OneAssetOption``.

    The Python port subclasses :class:`Option` directly (no
    ``OneAssetOption`` wrapper); the structural fields and the
    setup_arguments / fetch_results plumbing match.
    """

    def __init__(
        self,
        exercise: BermudanExercise,
        capacity: float,
        load: float,
        change_rate: float,
    ) -> None:
        super().__init__(NullPayoff(), exercise)
        self._capacity: float = float(capacity)
        self._load: float = float(load)
        self._change_rate: float = float(change_rate)

    def capacity(self) -> float:
        return self._capacity

    def load(self) -> float:
        return self._load

    def change_rate(self) -> float:
        return self._change_rate

    def is_expired(self) -> bool:
        """Always returns ``False`` — deferred to engine."""
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy storage parameters into engine arguments.

        # C++ parity: ``VanillaStorageOption::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, VanillaStorageOptionArguments),
            "wrong argument type (expected VanillaStorageOptionArguments)",
        )
        assert isinstance(args, VanillaStorageOptionArguments)
        args.capacity = self._capacity
        args.load = self._load
        args.change_rate = self._change_rate

    def fetch_results(self, results: PricingEngineResults) -> None:
        Instrument.fetch_results(self, results)


__all__ = ["VanillaStorageOption", "VanillaStorageOptionArguments"]
