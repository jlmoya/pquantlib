"""BootstrapHelper — abstract base for instruments used to bootstrap a curve.

# C++ parity: ql/termstructures/bootstraphelper.hpp (v1.42.1)

C++ uses ``template <class TS> class BootstrapHelper`` to parameterize on
the term-structure type. PQuantLib uses PEP 695 generics
(``class BootstrapHelper[TS]``) for the same type-level discipline.

Concrete rate helpers (DepositRateHelper, FraRateHelper, SwapRateHelper,
etc.) land in L2-C.

``RelativeDateBootstrapHelper`` (C++ subclass that registers with
Settings.evaluationDate to re-initialize dates) is now supported via
the ``ObservableSettings`` observer plumbing landed in L3-A.

C++ also defines a ``BootstrapError`` template; that class is
**deprecated in v1.40** with a recommendation to use a lambda
instead. PQuantLib skips it — bootstrap solvers in L2-B will accept
a ``Callable[[float], float]`` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

from pquantlib import qassert
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.time.date import Date


class PillarChoice(IntEnum):
    """How a bootstrap helper picks its pillar (anchor) date.

    # C++ parity: ql/termstructures/bootstraphelper.hpp ``Pillar::Choice``
    """

    MaturityDate = 0
    LastRelevantDate = 1
    CustomDate = 2

    def __str__(self) -> str:
        return {
            PillarChoice.MaturityDate: "MaturityPillarDate",
            PillarChoice.LastRelevantDate: "LastRelevantPillarDate",
            PillarChoice.CustomDate: "CustomPillarDate",
        }[self]


class BootstrapHelper[TS](Observable, ABC):
    """Abstract base for bootstrap helpers.

    A bootstrap helper wraps a market quote and exposes the *implied*
    quote that the term structure would yield for the same instrument.
    The bootstrap solver drives the helper's discount-factor pillar
    until ``quote_error()`` reaches zero.
    """

    def __init__(self, quote: Quote | float) -> None:
        super().__init__()
        self._quote: Quote = (
            quote if isinstance(quote, Quote) else SimpleQuote(float(quote))
        )
        self._term_structure: TS | None = None
        self._earliest_date: Date | None = None
        self._latest_date: Date | None = None
        self._maturity_date: Date | None = None
        self._latest_relevant_date: Date | None = None
        self._pillar_date: Date | None = None
        self._quote.register_with(self)

    def quote(self) -> Quote:
        return self._quote

    @abstractmethod
    def implied_quote(self) -> float:
        """Implied quote as computed by the term structure being bootstrapped."""

    def quote_error(self) -> float:
        return self._quote.value() - self.implied_quote()

    def set_term_structure(self, ts: TS) -> None:
        qassert.require(ts is not None, "null term structure given")
        self._term_structure = ts

    def earliest_date(self) -> Date:
        qassert.require(
            self._earliest_date is not None,
            "earliest_date not initialized — subclass must set self._earliest_date",
        )
        assert self._earliest_date is not None
        return self._earliest_date

    def maturity_date(self) -> Date:
        if self._maturity_date is None:
            return self.latest_relevant_date()
        return self._maturity_date

    def latest_relevant_date(self) -> Date:
        if self._latest_relevant_date is None:
            return self.latest_date()
        return self._latest_relevant_date

    def pillar_date(self) -> Date:
        if self._pillar_date is None:
            return self.latest_date()
        return self._pillar_date

    def latest_date(self) -> Date:
        if self._latest_date is None:
            qassert.require(
                self._pillar_date is not None,
                "latest_date / pillar_date not initialized — "
                "subclass must set self._latest_date or self._pillar_date",
            )
            assert self._pillar_date is not None
            return self._pillar_date
        return self._latest_date

    def update(self) -> None:
        self.notify_observers()


class RelativeDateBootstrapHelper[TS](BootstrapHelper[TS], ABC):
    """Bootstrap helper with dates derived from the global evaluation date.

    # C++ parity: ql/termstructures/bootstraphelper.hpp ``class
    # RelativeDateBootstrapHelper`` (v1.42.1) — a thin subclass that
    # registers with ``Settings::instance().evaluationDate()`` so that
    # the helper's earliest/latest/pillar dates re-initialize whenever
    # the global evaluation date moves.

    Subclasses override :meth:`_initialize_dates` to (re-)compute the
    helper's date triple. The ABC calls it once at construction and
    again on every ``update()`` from ``ObservableSettings``.

    Note: ``update()`` here both invalidates the helper's date cache
    (by re-running ``_initialize_dates``) AND forwards the notification
    to the helper's own observers — same dual responsibility as C++.
    """

    def __init__(self, quote: Quote | float) -> None:
        super().__init__(quote)
        # Register with global Settings so we get notified on eval-date moves.
        ObservableSettings().register_with(self)
        # Track the eval date we last initialized against; lazy re-init.
        self._eval_date_at_last_init: Date | None = None

    @abstractmethod
    def _initialize_dates(self) -> None:
        """Subclass: (re-)compute ``self._earliest_date`` / ``_latest_date`` /
        ``_pillar_date`` / ``_maturity_date`` from the current global
        evaluation date.
        """

    def update(self) -> None:
        """Observer.update — re-initialize dates if eval date moved."""
        today = ObservableSettings().evaluation_date_or_today()
        if today != self._eval_date_at_last_init:
            self._initialize_dates()
            self._eval_date_at_last_init = today
        self.notify_observers()
