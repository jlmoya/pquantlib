"""CashFlow — abstract base for cash flows.

# C++ parity: ql/cashflow.hpp + ql/cashflow.cpp (v1.42.1).

The C++ ``CashFlow`` derives from ``Event`` (date + Observable mixin) and
``LazyObject`` (deferred performCalculations). Python port:

- Inherit directly from ``pquantlib.patterns.observer.Observable`` (no
  separate Event class — the date()/hasOccurred() API is folded into
  CashFlow). This mirrors the jquantlib approach.
- ``LazyObject``'s deferred-calculation machinery is replaced by eager
  Python evaluation (every property recomputed on access). The C++
  ``performCalculations()`` default body in ``CashFlow`` is empty anyway,
  so this is a pure simplification.
- ``Settings::instance().evaluationDate()`` global mutable state is NOT
  ported. Python callers must pass an explicit ``ref_date`` to
  ``has_occurred``. ``ref_date=None`` is documented to mean "no reference
  — has_occurred returns False unless explicitly checking against a
  specific date".
- ``Settings::instance().includeReferenceDateEvents()`` flag is NOT
  ported. ``include_ref_date`` defaults to ``False`` (equivalent to the
  C++ default behaviour when neither override is set). Documented
  divergence.
- ``Visitor`` accept() machinery is omitted at the CashFlow level (the
  Visitor protocol exists in pquantlib.patterns.visitor but no cashflow
  classes implement accept() — visitor support is deferred per L2-D
  carve-out).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date

# Module-level null Date used as default for optional Date arguments
# (avoids ruff B008 by not constructing per call). Date is frozen+slots
# so this is safe to share.
_NULL_DATE: Date = Date()


class CashFlow(Observable, ABC):
    """Abstract base for a cash-flow event.

    Subclasses must implement ``date()`` and ``amount()``.
    """

    @abstractmethod
    def date(self) -> Date:
        """Returns the date at which the cash flow occurs."""

    @abstractmethod
    def amount(self) -> float:
        """Returns the (undiscounted) amount paid at ``date()``."""

    def ex_coupon_date(self) -> Date:
        """Date on which this cash flow trades ex-coupon. Default: null Date.

        C++ parity: ql/cashflow.hpp:66 — virtual returns ``Date()`` (null).
        """
        return _NULL_DATE

    def has_occurred(
        self,
        ref_date: Date | None = None,
        include_ref_date: bool | None = None,
    ) -> bool:
        """Whether this cash-flow event has already occurred at ``ref_date``.

        Logic mirrors C++ ``Event::hasOccurred`` (ql/event.cpp:28-39) plus
        the CashFlow override (ql/cashflow.cpp:27-49):

        - If ``ref_date`` is None or null Date, returns ``False``
          (no Settings.evaluationDate fallback in this port).
        - Otherwise: if ``date()`` < ref_date returns ``True``; if
          ``date()`` > ref_date returns ``False``; on equality, returns
          ``not include_ref_date`` (default ``not False`` == ``True``,
          matching C++ default of including the ref-date event).

        # C++ parity divergence: the C++ override calls
        # ``Settings::instance().includeTodaysCashFlows()`` to override the
        # bool when ref_date == today. This port omits Settings — pass
        # ``include_ref_date`` explicitly if your callsite needs the
        # equivalent of QL's "include today's cashflows" toggle.
        """
        if ref_date is None or ref_date == _NULL_DATE:
            return False
        cf_date = self.date()
        if ref_date < cf_date:
            return False
        if cf_date < ref_date:
            return True
        # cf_date == ref_date: include if include_ref_date is True.
        # C++ default when both are None and there's no Settings override:
        # includeRefDateEvent is the Settings flag. We default to False
        # (event has occurred on its own date) matching the most common
        # cashflow-analysis intent — see CashFlows.npv tests.
        include = include_ref_date if include_ref_date is not None else False
        return not include

    def trading_ex_coupon(self, ref_date: Date | None = None) -> bool:
        """Whether the cash flow is trading ex-coupon on ``ref_date``.

        C++ parity: ql/cashflow.cpp:51-61.
        """
        ecd = self.ex_coupon_date()
        if ecd == _NULL_DATE:
            return False
        if ref_date is None or ref_date == _NULL_DATE:
            # C++ falls back to Settings::evaluationDate(); we cannot —
            # return False matching the most conservative interpretation.
            return False
        return ecd <= ref_date
