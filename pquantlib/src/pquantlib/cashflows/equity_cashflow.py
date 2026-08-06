"""Equity cash flow, its pricer interface, and the quanto pricer.

# C++ parity: ql/cashflows/equitycashflow.hpp + ql/cashflows/equitycashflow.cpp
# (v1.43).

Three classes plus one free function:

- :class:`EquityCashFlow` — an :class:`~pquantlib.cashflows.indexed_cashflow.IndexedCashFlow`
  over an :class:`~pquantlib.indexes.equity_index.EquityIndex`, defaulting to
  ``growth_only=True`` (the swap-type payoff ``I1/I0 - 1``) rather than the
  ``IndexedCashFlow`` default of ``False``. When a pricer is attached the
  amount becomes ``notional * pricer.price()`` instead.
- :class:`EquityCashFlowPricer` — the abstract Observer+Observable pricer base.
- :class:`EquityQuantoCashFlowPricer` — replaces the index's dividend curve
  with a :class:`~pquantlib.termstructures.yield_.quanto_term_structure.QuantoTermStructure`
  and its interest curve with the quanto-currency curve, so that the forward
  it reads back is the quanto forward
  ``spot * exp((r_local - q - rho * sigma_eq * sigma_fx) * t)``.
- :func:`set_coupon_pricer` — attaches one pricer to every ``EquityCashFlow``
  in a leg, leaving other flows untouched.

Python divergences from C++:

- ``Handle<...>`` is replaced by ``... | None``, an empty handle being
  ``None``. The quanto pricer's four inputs are therefore typed optional even
  though all four are required by ``initialize`` — C++ accepts empty handles
  at construction too and defers the check to ``initialize``
  (equitycashflow.cpp:100-112), and that deferral is observable.
- ``EquityCashFlow`` stores the index as an ``EquityIndex`` (C++ slices it to
  ``Index`` in the ``IndexedCashFlow`` base and recovers it with a
  ``dynamic_pointer_cast``). The C++ ``QL_FAIL("Equity index required.")``
  branch (equitycashflow.cpp:93-96) is therefore statically unreachable here,
  exactly as it is unreachable in C++ given the constructor signature.
- ``accept(AcyclicVisitor&)`` is omitted, as everywhere else in this port.
- C++ ``CashFlow`` is an Observer (through ``LazyObject``) and so can observe
  its pricer out of the box. PQuantLib's ``CashFlow`` is only an
  ``Observable``, so :meth:`EquityCashFlow.update` is declared here rather
  than inherited; being eager, it notifies without a cache to invalidate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.indexed_cashflow import IndexedCashFlow
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.indexes.equity_index import EquityIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.quanto_term_structure import QuantoTermStructure
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.date import Date

_NULL_DATE: Date = Date()


def _configure_dividend_curve(dividend: YieldTermStructure | None) -> YieldTermStructure:
    """The dividend curve, or a flat 0% stand-in when none was supplied.

    # C++ parity: anonymous ``configureDividendHandle`` (equitycashflow.cpp:32-42),
    # which builds ``FlatForward(0, NullCalendar(), SimpleQuote(0.0),
    # Actual365Fixed())`` — settlement days zero, hence reference date equal to
    # the evaluation date. PQuantLib's ``FlatForward`` has no settlement-days
    # constructor, so the equivalent fixed reference date is taken explicitly.
    # The curve is rebuilt on every ``price()`` call, so it re-snaps whenever
    # the evaluation date moves, as the C++ moving-mode curve would.
    """
    if dividend is not None:
        return dividend
    return FlatForward.from_rate(ObservableSettings().evaluation_date_or_today(), 0.0, Actual365Fixed())


class EquityCashFlowPricer(Observable, ABC):
    """Abstract pricer for an :class:`EquityCashFlow`.

    # C++ parity: ``EquityCashFlowPricer`` (equitycashflow.hpp:71-86) —
    # ``public virtual Observer, public virtual Observable``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._index: EquityIndex | None = None
        self._base_date: Date = _NULL_DATE
        self._fixing_date: Date = _NULL_DATE
        self._growth_only_payoff: bool = False

    @abstractmethod
    def price(self) -> float:
        """Payoff per unit of notional. Call :meth:`initialize` first."""

    @abstractmethod
    def initialize(self, cash_flow: EquityCashFlow) -> None:
        """Bind the pricer to ``cash_flow`` and validate its inputs."""

    def update(self) -> None:
        """Observer interface — re-broadcast upstream market-data changes.

        # C++ parity: inline ``EquityCashFlowPricer::update`` (hpp:81).
        """
        self.notify_observers()


class EquityCashFlow(IndexedCashFlow):
    """Cash flow paying an equity index return, optionally quanto-adjusted.

    # C++ parity: ``EquityCashFlow`` (equitycashflow.hpp:36-57).
    """

    def __init__(
        self,
        notional: float,
        index: EquityIndex,
        base_date: Date,
        fixing_date: Date,
        payment_date: Date,
        growth_only: bool = True,
    ) -> None:
        super().__init__(notional, index, base_date, fixing_date, payment_date, growth_only)
        self._equity_index: EquityIndex = index
        self._pricer: EquityCashFlowPricer | None = None

    # ---- inspectors ----------------------------------------------------------

    def index(self) -> EquityIndex:
        """The equity index, narrowed from the ``IndexedCashFlow`` base type."""
        return self._equity_index

    def pricer(self) -> EquityCashFlowPricer | None:
        """# C++ parity: inline ``EquityCashFlow::pricer`` (hpp:53)."""
        return self._pricer

    # ---- pricer wiring -------------------------------------------------------

    def set_pricer(self, pricer: EquityCashFlowPricer | None) -> None:
        """Attach (or, with ``None``, detach) the pricer.

        # C++ parity: ``EquityCashFlow::setPricer`` (equitycashflow.cpp:63-70).
        """
        if self._pricer is not None:
            self._pricer.unregister_with(self)
        self._pricer = pricer
        if pricer is not None:
            pricer.register_with(self)
        self.update()

    def update(self) -> None:
        """Observer interface — re-broadcast a pricer/market-data change.

        # C++ parity: ``EquityCashFlow`` observes its pricer
        # (equitycashflow.cpp:63-70) and inherits ``update`` from
        # ``LazyObject``, which invalidates the cached amount and notifies.
        # PQuantLib's ``CashFlow`` is an ``Observable`` but *not* an
        # ``Observer``, and computes eagerly, so only the notification
        # remains — and it has to be declared here rather than inherited.
        """
        self.notify_observers()

    # ---- CashFlow interface --------------------------------------------------

    def amount(self) -> float:
        """``notional * pricer.price()``, or the plain index ratio when unpriced.

        # C++ parity: ``EquityCashFlow::amount`` (equitycashflow.cpp:72-77).
        """
        if self._pricer is None:
            return super().amount()
        self._pricer.initialize(self)
        return self.notional() * self._pricer.price()


class EquityQuantoCashFlowPricer(EquityCashFlowPricer):
    """Quanto-adjusted pricer for an :class:`EquityCashFlow`.

    # C++ parity: ``EquityQuantoCashFlowPricer`` (equitycashflow.hpp:90-104,
    # equitycashflow.cpp:79-137).
    """

    def __init__(
        self,
        quanto_currency_term_structure: YieldTermStructure | None,
        equity_volatility: BlackVolTermStructure | None,
        fx_volatility: BlackVolTermStructure | None,
        correlation: Quote | None,
    ) -> None:
        super().__init__()
        self._quanto_currency_term_structure: YieldTermStructure | None = quanto_currency_term_structure
        self._equity_volatility: BlackVolTermStructure | None = equity_volatility
        self._fx_volatility: BlackVolTermStructure | None = fx_volatility
        self._correlation: Quote | None = correlation

        # C++ parity: equitycashflow.cpp:85-88.
        if quanto_currency_term_structure is not None:
            quanto_currency_term_structure.register_with(self)
        if equity_volatility is not None:
            equity_volatility.register_with(self)
        if fx_volatility is not None:
            fx_volatility.register_with(self)
        if correlation is not None:
            correlation.register_with(self)

    # ---- interface -----------------------------------------------------------

    def initialize(self, cash_flow: EquityCashFlow) -> None:
        """# C++ parity: ``EquityQuantoCashFlowPricer::initialize`` (cpp:91-118)."""
        self._index = cash_flow.index()
        self._base_date = cash_flow.base_date()
        self._fixing_date = cash_flow.fixing_date()
        qassert.require(
            self._fixing_date >= self._base_date,
            "Fixing date cannot fall before base date.",
        )
        self._growth_only_payoff = cash_flow.growth_only()

        qassert.require(
            self._quanto_currency_term_structure is not None,
            "Quanto currency term structure handle cannot be empty.",
        )
        qassert.require(
            self._equity_volatility is not None,
            "Equity volatility term structure handle cannot be empty.",
        )
        qassert.require(
            self._fx_volatility is not None,
            "FX volatility term structure handle cannot be empty.",
        )
        qassert.require(self._correlation is not None, "Correlation handle cannot be empty.")
        assert self._quanto_currency_term_structure is not None
        assert self._equity_volatility is not None
        assert self._fx_volatility is not None

        quanto_reference = self._quanto_currency_term_structure.reference_date()
        equity_reference = self._equity_volatility.reference_date()
        fx_reference = self._fx_volatility.reference_date()
        qassert.require(
            quanto_reference == equity_reference and equity_reference == fx_reference,
            "Quanto currency term structure, equity and FX volatility need to have the same reference date.",
        )

    def price(self) -> float:
        """Quanto payoff per unit of notional.

        # C++ parity: ``EquityQuantoCashFlowPricer::price`` (cpp:120-136).
        """
        index = self._index
        qassert.require(index is not None, "pricer has not been initialized")
        assert index is not None
        assert self._quanto_currency_term_structure is not None
        assert self._equity_volatility is not None
        assert self._fx_volatility is not None
        assert self._correlation is not None

        strike = index.fixing(self._fixing_date)
        dividend_curve = _configure_dividend_curve(index.equity_dividend_curve())

        interest_curve = index.equity_interest_rate_curve()
        # C++ dereferences the (possibly empty) handle inside
        # QuantoTermStructure::zeroYieldImpl and throws there; PQuantLib has
        # no handle indirection, so the same failure is raised up front.
        qassert.require(
            interest_curve is not None,
            f"null interest rate term structure set to this instance of {index.name()}",
        )
        assert interest_curve is not None

        quanto_term_structure = QuantoTermStructure(
            dividend_curve,
            self._quanto_currency_term_structure,
            interest_curve,
            self._equity_volatility,
            strike,
            self._fx_volatility,
            1.0,
            self._correlation.value(),
        )
        quanto_index = index.clone(self._quanto_currency_term_structure, quanto_term_structure, index.spot())

        i0 = quanto_index.fixing(self._base_date)
        i1 = quanto_index.fixing(self._fixing_date)

        if self._growth_only_payoff:
            return i1 / i0 - 1.0
        return i1 / i0


def set_coupon_pricer(leg: Iterable[CashFlow], pricer: EquityCashFlowPricer) -> None:
    """Attach ``pricer`` to every :class:`EquityCashFlow` in ``leg``.

    # C++ parity: ``setCouponPricer(const Leg&, ...)`` (equitycashflow.cpp:44-51).
    # Non-``EquityCashFlow`` entries are skipped silently.
    """
    for cf in leg:
        if isinstance(cf, EquityCashFlow):
            cf.set_pricer(pricer)
