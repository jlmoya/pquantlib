"""CDSOption — option on a CreditDefaultSwap.

# C++ parity: ql/experimental/credit/cdsoption.{hpp,cpp} (v1.42.1).

A CDS option grants the holder the right (but not the obligation) to
enter into an underlying CDS at the option's exercise date. The
direction of the option is set by the underlying's protection side:
a Buyer-side underlying gives a payer CDS option; a Seller-side
underlying gives a receiver CDS option.

By convention:

* All receiver CDS options must knock-out on a credit event before
  the option expiry.
* Payer CDS options may be either knock-out or non-knock-out; the
  non-knock-out variant adds a front-end-protection contribution
  paid up-front (see ``BlackCDSOptionEngine``).

The underlying must be a running-spread-only CDS — upfront-style
underlyings are unsupported (matches C++ check).
"""

from __future__ import annotations

from typing import cast

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.credit_default_swap import (
    CreditDefaultSwap,
    CreditDefaultSwapArguments,
    ProtectionSide,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.option import Option, OptionArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class _NullPayoff(Payoff):
    """Sentinel payoff for CDSOption.

    # C++ parity: ql/instruments/payoffs.hpp ``NullPayoff`` — the C++
    # CDS option constructs ``Option(ext::make_shared<NullPayoff>, exercise)``
    # because the option's payoff is encoded in the underlying CDS, not
    # in a strike. The Python port defines a minimal local sentinel
    # (NullPayoff was deferred at L3-A); it is never evaluated.
    """

    def name(self) -> str:
        return "Null"

    def description(self) -> str:
        return "Null"

    def __call__(self, price: float) -> float:
        qassert.fail("null payoff not handled")
        return 0.0


class CDSOptionArguments(CreditDefaultSwapArguments, OptionArguments):
    """Engine-arguments carrier for CDSOption.

    # C++ parity: ``CdsOption::arguments`` (multiple inheritance from
    # ``CreditDefaultSwap::arguments`` + ``Option::arguments``).
    """

    def __init__(self) -> None:
        CreditDefaultSwapArguments.__init__(self)
        OptionArguments.__init__(self)
        self.swap: CreditDefaultSwap | None = None
        self.knocks_out: bool = True

    def validate(self) -> None:
        # Inherits CDS-arg validation + Option-arg validation.
        CreditDefaultSwapArguments.validate(self)
        OptionArguments.validate(self)
        qassert.require(self.swap is not None, "CDS not set")
        qassert.require(self.exercise is not None, "exercise not set")


class CDSOptionResults(InstrumentResults):
    """Engine-results carrier for CDSOption.

    # C++ parity: ``CdsOption::results`` (extends Option::results, which
    # itself ultimately extends Instrument::results). The Python port
    # inherits ``InstrumentResults`` directly — ``Option::results`` is a
    # pure mixin in C++ contributing only ``Greeks``/``MoreGreeks`` fields,
    # which the Black CDS engine does not populate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.risky_annuity: float | None = None

    def reset(self) -> None:
        super().reset()
        self.risky_annuity = None


class CDSOption(Option):
    """Option on a CreditDefaultSwap.

    # C++ parity: ``CdsOption`` class.

    Construction binds the option to an underlying ``CreditDefaultSwap``
    and an ``Exercise``. The ``knocks_out`` flag controls whether the
    option terminates on a credit event before expiry; receiver options
    (Seller-side underlying) must knock-out.
    """

    def __init__(
        self,
        underlying: CreditDefaultSwap,
        exercise: Exercise,
        knocks_out: bool = True,
    ) -> None:
        """Build the CDS option.

        # C++ parity: cdsoption.cpp:69-78.
        """
        super().__init__(_NullPayoff(), exercise)
        qassert.require(
            underlying.side() == ProtectionSide.Buyer or knocks_out,
            "receiver CDS options must knock out",
        )
        qassert.require(
            underlying.upfront() is None,
            "underlying must be running-spread only",
        )
        self._swap: CreditDefaultSwap = underlying
        self._knocks_out: bool = knocks_out
        self._risky_annuity: float | None = None
        underlying.register_with(self)

    # ---- Instrument interface --------------------------------------

    def is_expired(self) -> bool:
        """The exercise date has passed.

        # C++ parity: cdsoption.cpp:80-82 uses
        # ``detail::simple_event(exercise->dates().back()).hasOccurred()``.
        """
        today = ObservableSettings().evaluation_date_or_today()
        last = self._exercise.dates()[-1]
        return last <= today

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        qassert.require(
            isinstance(args, CDSOptionArguments),
            "CDSOption.setup_arguments: wrong argument type",
        )
        assert isinstance(args, CDSOptionArguments)
        # Defer to swap to fill the CDS-arg fields, then layer
        # option-specific data on top.
        self._swap.setup_arguments(args)
        super().setup_arguments(args)
        args.swap = self._swap
        args.knocks_out = self._knocks_out

    def fetch_results(self, results: PricingEngineResults) -> None:
        super().fetch_results(results)
        qassert.require(
            isinstance(results, CDSOptionResults),
            "CDSOption.fetch_results: wrong result type",
        )
        assert isinstance(results, CDSOptionResults)
        self._risky_annuity = results.risky_annuity

    def setup_expired(self) -> None:
        super().setup_expired()
        self._risky_annuity = 0.0

    # ---- inspectors -----------------------------------------------

    def underlying_swap(self) -> CreditDefaultSwap:
        return self._swap

    def knocks_out(self) -> bool:
        return self._knocks_out

    # ---- calculations ---------------------------------------------

    def atm_rate(self) -> float:
        """At-the-money rate = fair spread of the underlying.

        # C++ parity: cdsoption.cpp:110-112.
        """
        return self._swap.fair_spread()

    def risky_annuity(self) -> float:
        """Risky-annuity result from the engine.

        # C++ parity: cdsoption.cpp:114-118.
        """
        self.calculate()
        qassert.require(
            self._risky_annuity is not None, "risky annuity not provided"
        )
        return cast("float", self._risky_annuity)


__all__ = [
    "CDSOption",
    "CDSOptionArguments",
    "CDSOptionResults",
]
