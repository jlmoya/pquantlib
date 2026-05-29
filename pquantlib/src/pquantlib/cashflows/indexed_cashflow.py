"""IndexedCashFlow — cash flow whose amount is the ratio of two index fixings.

# C++ parity: ql/cashflows/indexedcashflow.{hpp,cpp} (v1.42.1).

The C++ class is the abstract base for any cash flow whose amount is a
function of two index fixings (one at a base date, one at a fixing
date). The amount is::

    amount = notional * (I1 / I0)             if growth_only == False
           = notional * (I1 / I0 - 1.0)       if growth_only == True

This is used directly by ``ZeroInflationCashFlow`` and ``CPICashFlow``
(both of which override ``base_fixing()`` / ``index_fixing()`` to apply
the CPI lagged-fixing math instead of a naive ``index.fixing(d)``).

Python divergences from C++:

- The index slot is typed as a runtime ``Index`` (so we can call
  ``add_fixing`` / inspect ``time_series``); we don't require
  ``InflationIndexProtocol`` here because ``IndexedCashFlow`` is the
  *generic* cash flow before the CPI/YoY specialisation.
- The C++ ``LazyObject::performCalculations`` deferred-execution cache
  is replaced by eager recomputation (consistent with our
  ``InflationCoupon`` choice).
- ``Visitor`` accept() is omitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.indexes.index import Index


class IndexedCashFlow(CashFlow):
    """Cash flow whose amount depends on an index ratio (no accruals).

    # C++ parity: ``IndexedCashFlow`` in indexedcashflow.hpp.
    """

    def __init__(
        self,
        notional: float,
        index: Index,
        base_date: Date,
        fixing_date: Date,
        payment_date: Date,
        growth_only: bool = False,
    ) -> None:
        # C++ parity: indexedcashflow.cpp:32-36 includes ``QL_REQUIRE(index_,
        # "no index provided")``. Python's typed parameter forbids None,
        # so the runtime check is redundant — we drop it.
        super().__init__()
        self._notional: float = notional
        self._index: Index = index
        self._base_date: Date = base_date
        self._fixing_date: Date = fixing_date
        self._payment_date: Date = payment_date
        self._growth_only: bool = growth_only

    # ---- Event/CashFlow interface ------------------------------------

    def date(self) -> Date:
        """C++ parity: ``IndexedCashFlow::date`` (inline in hpp:55)."""
        return self._payment_date

    # ---- inspectors --------------------------------------------------

    def notional(self) -> float:
        return self._notional

    def base_date(self) -> Date:
        return self._base_date

    def fixing_date(self) -> Date:
        return self._fixing_date

    def index(self) -> Index:
        return self._index

    def growth_only(self) -> bool:
        return self._growth_only

    def base_fixing(self) -> float:
        """Base-date index fixing.

        # C++ parity: ql/cashflows/indexedcashflow.hpp:62 (inline) —
        # default is ``index.fixing(baseDate())``. Subclasses
        # (ZeroInflationCashFlow, CPICashFlow) override.
        """
        return self._index.fixing(self._base_date)

    def index_fixing(self) -> float:
        """Fixing-date index fixing.

        # C++ parity: ql/cashflows/indexedcashflow.hpp:63 (inline).
        """
        return self._index.fixing(self._fixing_date)

    # ---- CashFlow interface ------------------------------------------

    def amount(self) -> float:
        """``notional * (I1/I0)`` or ``notional * (I1/I0 - 1)``.

        # C++ parity: ql/cashflows/indexedcashflow.cpp:43-51 — wrapped in
        # ``performCalculations`` then exposed via ``amount() ->
        # calculate(); return amount_;``. We compute eagerly.
        """
        i0 = self.base_fixing()
        i1 = self.index_fixing()
        if self._growth_only:
            return self._notional * (i1 / i0 - 1.0)
        return self._notional * (i1 / i0)
