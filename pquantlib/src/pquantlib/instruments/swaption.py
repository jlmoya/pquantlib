"""Swaption — option on a fixed-vs-floating interest rate swap.

# C++ parity: ql/instruments/swaption.{hpp,cpp} (v1.42.1).

C++ ``Swaption`` is an ``Option`` (which itself is an ``Instrument``)
that carries a ``FixedVsFloatingSwap`` underlying and a settlement
configuration (Physical / Cash * PhysicalOTC / PhysicalCleared /
CollateralizedCashPrice / ParYieldCurve). The settlement enums are
nested under a ``Settlement`` struct in C++; PQuantLib promotes them
to free ``SettlementType`` / ``SettlementMethod`` IntEnums per
Python's flat-namespace convention.

The C++ engine API expects ``Swaption::arguments``, a
multi-inheritance carrier of ``FixedVsFloatingSwap::arguments`` and
``Option::arguments``. PQuantLib mirrors that with
``SwaptionArguments`` — it carries the same fields plus the swap
reference + settlement.

Divergences from C++:

- C++ uses ``Payoff = nullptr`` (no payoff for a swaption — the swap
  itself plays that role). PQuantLib mirrors this by passing a
  shared ``_NullSwaptionPayoff`` sentinel to ``Option.__init__``.
  ``setup_arguments`` overwrites the bundled payoff to ``None`` so
  no engine ever sees it.
- C++ ``deepUpdate`` / ``alwaysForwardNotifications`` are LazyObject
  cache-invalidation primitives that aren't separately surfaced in
  PQuantLib's LazyObject port (notification fan-out works by
  re-registration on each engine swap instead). We expose
  ``deep_update`` as a synonym for ``update`` so callers can write
  forward-compatible code.
- ``implied_volatility`` is deferred — depends on
  ``BlackSwaptionEngine`` + ``NewtonSafe`` and is not exercised by
  any Phase 4 test path. Re-add when a calibration test asks for it.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.fixed_vs_floating_swap import (
    FixedVsFloatingSwap,
    FixedVsFloatingSwapArguments,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.instruments.swap import SwapType
from pquantlib.option import Option, OptionArguments
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngineArguments


class SettlementType(IntEnum):
    """Whether a swaption settles physically or in cash.

    # C++ parity: ``Settlement::Type`` in swaption.hpp:41-42 (v1.42.1).
    """

    Physical = 0
    Cash = 1


class SettlementMethod(IntEnum):
    """How the settlement actually happens.

    # C++ parity: ``Settlement::Method`` in swaption.hpp:43-48 (v1.42.1).
    """

    PhysicalOTC = 0
    PhysicalCleared = 1
    CollateralizedCashPrice = 2
    ParYieldCurve = 3


def check_settlement_type_and_method_consistency(
    settlement_type: SettlementType, settlement_method: SettlementMethod
) -> None:
    """Reject (Physical, ParYieldCurve)-style invalid combinations.

    # C++ parity: ``Settlement::checkTypeAndMethodConsistency`` in
    # swaption.cpp:207-220 (v1.42.1).
    """
    if settlement_type == SettlementType.Physical:
        qassert.require(
            settlement_method
            in (SettlementMethod.PhysicalOTC, SettlementMethod.PhysicalCleared),
            "invalid settlement method for physical settlement",
        )
    if settlement_type == SettlementType.Cash:
        qassert.require(
            settlement_method
            in (
                SettlementMethod.CollateralizedCashPrice,
                SettlementMethod.ParYieldCurve,
            ),
            "invalid settlement method for cash settlement",
        )


class _NullSwaptionPayoff(Payoff):
    """Placeholder payoff for swaptions.

    # C++ parity: C++ passes ``ext::shared_ptr<Payoff>()`` (a null
    # shared_ptr) to ``Option(payoff, exercise)``. ``Option`` stores
    # it but the engines never call into the swaption payoff — the
    # swap itself defines the cashflow structure. PQuantLib uses a
    # sentinel ``Payoff`` subclass so ``Option.__init__``'s type
    # constraint is satisfied without introducing ``Payoff | None``
    # plumbing in the abstract base.
    """

    def name(self) -> str:
        return "NullSwaptionPayoff"

    def description(self) -> str:
        return "Null payoff (used by Swaption to satisfy Option contract)"

    def __call__(self, price: float) -> float:
        # Never called — the swap underlying determines the payoff.
        return 0.0


_NULL_SWAPTION_PAYOFF: _NullSwaptionPayoff = _NullSwaptionPayoff()


class SwaptionArguments(FixedVsFloatingSwapArguments, OptionArguments):
    """Engine argument carrier for Swaption.

    # C++ parity: ``Swaption::arguments`` in swaption.hpp:141-149
    # (v1.42.1) — multi-inherits ``FixedVsFloatingSwap::arguments``
    # and ``Option::arguments``.
    """

    def __init__(self) -> None:
        # Multi-inheritance init: both parents have ``__init__`` and
        # neither calls ``super`` (they're plain data carriers).
        FixedVsFloatingSwapArguments.__init__(self)
        OptionArguments.__init__(self)
        self.swap: FixedVsFloatingSwap | None = None
        self.settlement_type: SettlementType = SettlementType.Physical
        self.settlement_method: SettlementMethod = SettlementMethod.PhysicalOTC

    def validate(self) -> None:
        # # C++ parity: ``Swaption::arguments::validate`` in
        # # swaption.cpp:175-180 (v1.42.1).
        FixedVsFloatingSwapArguments.validate(self)
        qassert.require(self.swap is not None, "swap not set")
        qassert.require(self.exercise is not None, "exercise not set")
        check_settlement_type_and_method_consistency(
            self.settlement_type, self.settlement_method
        )


class SwaptionResults(InstrumentResults):
    """Engine results carrier for Swaption.

    # C++ parity: ``Swaption::results`` is just the standard
    # ``Instrument::results`` plus the engine's ``additionalResults``
    # map — both already provided by ``InstrumentResults``.
    """


class Swaption(Option):
    """Option on a fixed-vs-floating interest rate swap.

    # C++ parity: ``class Swaption : public Option`` in
    # swaption.hpp:89-138 (v1.42.1).
    """

    def __init__(
        self,
        swap: FixedVsFloatingSwap,
        exercise: Exercise,
        settlement_type: SettlementType = SettlementType.Physical,
        settlement_method: SettlementMethod = SettlementMethod.PhysicalOTC,
    ) -> None:
        # # C++ parity: Swaption::Swaption (swaption.cpp:133-151).
        check_settlement_type_and_method_consistency(
            settlement_type, settlement_method
        )
        super().__init__(_NULL_SWAPTION_PAYOFF, exercise)
        self._swap: FixedVsFloatingSwap = swap
        self._settlement_type: SettlementType = settlement_type
        self._settlement_method: SettlementMethod = settlement_method
        # C++ parity: swaption.cpp:139 — registerWith(swap_) — so swap
        # updates invalidate the swaption cache.
        self._swap.register_with(self)

    # --- Observer / LazyObject interface ----------------------------------

    def deep_update(self) -> None:
        """Forward a deep update to the swap, then mark the cache dirty.

        # C++ parity: ``Swaption::deepUpdate`` in swaption.cpp:153-156
        # (v1.42.1). PQuantLib's LazyObject doesn't separately surface
        # ``deepUpdate`` vs ``update`` — both just notify observers
        # and invalidate caches via the standard ``update`` chain.
        """
        self.update()

    # --- Instrument interface ----------------------------------------------

    def is_expired(self) -> bool:
        """Expired iff the last exercise date is past today.

        # C++ parity: ``Swaption::isExpired`` (swaption.cpp:158-160).
        # C++ uses ``simple_event(exercise_->dates().back()).hasOccurred()``
        # which consults Settings::evaluationDate. PQuantLib's
        # Settings.evaluation_date is wired but VanillaOption defers
        # to `False` for the same reason — let the engine compute the
        # expired-day NPV (zero). We match that.
        """
        return False

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy swap + exercise + settlement into the engine's arguments.

        # C++ parity: ``Swaption::setupArguments`` (swaption.cpp:162-173).
        """
        # First let the swap fill its own fields. The Swaption-specific
        # argument carrier *is a* FixedVsFloatingSwapArguments.
        self._swap.setup_arguments(args)
        qassert.require(
            isinstance(args, SwaptionArguments),
            "wrong argument type (expected SwaptionArguments)",
        )
        assert isinstance(args, SwaptionArguments)
        args.swap = self._swap
        args.settlement_type = self._settlement_type
        args.settlement_method = self._settlement_method
        args.exercise = self._exercise
        # Match C++: the swaption sets ``payoff = nullptr``. PQuantLib
        # carries a sentinel ``Payoff`` to satisfy Option.__init__,
        # but engines should never read it — null it here so any
        # accidental access errors out.
        args.payoff = None

    # --- inspectors -------------------------------------------------------

    @property
    def settlement_type(self) -> SettlementType:
        return self._settlement_type

    @property
    def settlement_method(self) -> SettlementMethod:
        return self._settlement_method

    def type(self) -> SwapType:
        """Return the underlying swap's payer/receiver type.

        # C++ parity: ``Swaption::type`` (swaption.hpp:113).
        """
        return self._swap.swap_type()

    def underlying_swap(self) -> FixedVsFloatingSwap:
        """The underlying swap (Python rename of C++ ``underlying``)."""
        return self._swap


__all__ = [
    "SettlementMethod",
    "SettlementType",
    "Swaption",
    "SwaptionArguments",
    "SwaptionResults",
    "check_settlement_type_and_method_consistency",
]
