"""CdsOption — option on a CreditDefaultSwap.

# C++ parity: ql/experimental/credit/cdsoption.{hpp,cpp} (v1.43).

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
  paid up-front (see ``BlackCdsOptionEngine``).

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
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.option import Option, OptionArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import NullPayoff
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.default_probability_term_structure import (
    DefaultProbabilityTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class CdsOptionArguments(CreditDefaultSwapArguments, OptionArguments):
    """Engine-arguments carrier for CdsOption.

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


class CdsOptionResults(InstrumentResults):
    """Engine-results carrier for CdsOption.

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


class CdsOption(Option):
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
        super().__init__(NullPayoff(), exercise)
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
            isinstance(args, CdsOptionArguments),
            "CdsOption.setup_arguments: wrong argument type",
        )
        assert isinstance(args, CdsOptionArguments)
        # Defer to swap to fill the CDS-arg fields, then layer
        # option-specific data on top.
        self._swap.setup_arguments(args)
        super().setup_arguments(args)
        args.swap = self._swap
        args.knocks_out = self._knocks_out

    def fetch_results(self, results: PricingEngineResults) -> None:
        super().fetch_results(results)
        qassert.require(
            isinstance(results, CdsOptionResults),
            "CdsOption.fetch_results: wrong result type",
        )
        assert isinstance(results, CdsOptionResults)
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

    def implied_volatility(
        self,
        target_value: float,
        term_structure: YieldTermStructure,
        probability: DefaultProbabilityTermStructure,
        recovery_rate: float,
        accuracy: float = 1.0e-4,
        max_evaluations: int = 100,
        min_vol: float = 1.0e-7,
        max_vol: float = 4.0,
    ) -> float:
        """Lognormal vol that reprices this option at ``target_value``.

        # C++ parity: cdsoption.cpp:120-139, including the anonymous-namespace
        # ``ImpliedVolHelper`` (cdsoption.cpp:34-64): a private
        # :class:`BlackCdsOptionEngine` is driven off a mutable
        # :class:`SimpleQuote`, this option's arguments are copied into it once,
        # and Brent brackets the vol in ``[min_vol, max_vol]`` from a hardcoded
        # guess of 0.10.

        Note the argument order: C++ takes ``(termStructure, probability)``
        while :class:`BlackCdsOptionEngine` is constructed
        ``(probability, ..., termStructure, ...)``. The order here follows
        the C++ method signature.
        """
        # C++ calls calculate() first: implied vol is only defined for a live
        # option, and calculate() is what raises if no engine is attached.
        self.calculate()
        qassert.require(not self.is_expired(), "instrument expired")

        # Deferred import: blackcdsoptionengine.hpp includes cdsoption.hpp,
        # so C++ resolves the cycle at the .cpp level (cdsoption.cpp:22).
        # Python resolves it here, at call time, for the same reason.
        from pquantlib.pricingengines.credit.black_cds_option_engine import (  # noqa: PLC0415
            BlackCdsOptionEngine,
        )

        vol = SimpleQuote(0.0)
        engine = BlackCdsOptionEngine(
            probability, recovery_rate, term_structure, vol
        )
        # C++ ImpliedVolHelper copies the arguments ONCE in its constructor and
        # then only pokes the quote, so the CDS's own engine is never re-run.
        self.setup_arguments(engine.get_arguments())
        results = engine.get_results()

        def f(x: float) -> float:
            vol.set_value(x)
            engine.calculate()
            value = results.value
            assert value is not None
            return value - target_value

        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        return solver.solve(f, accuracy, 0.10, min_vol, max_vol)


__all__ = [
    "CdsOption",
    "CdsOptionArguments",
    "CdsOptionResults",
]
