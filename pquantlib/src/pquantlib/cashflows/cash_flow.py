"""CashFlow — abstract base for cash flows.

# C++ parity: ql/cashflow.hpp + ql/cashflow.cpp @ v1.43.

- ``CashFlow`` derives from :class:`pquantlib.event.Event` exactly as C++
  does (``class CashFlow : public Event, public LazyObject``), so the
  ``date()`` / ``hasOccurred()`` contract lives in one place.
- ``LazyObject``'s deferred-calculation machinery is replaced by eager
  Python evaluation (every property recomputed on access). The C++
  ``performCalculations()`` default body in ``CashFlow`` is empty anyway,
  so this is a pure simplification.
- ``Settings`` is consulted exactly where C++ consults it — see
  :meth:`has_occurred` and :meth:`trading_ex_coupon`. (An earlier revision of
  this module claimed "Settings global mutable state is NOT ported"; that
  premise was stale — :class:`pquantlib.patterns.observable_settings.ObservableSettings`
  has carried ``evaluation_date`` and ``include_reference_date_events`` for a
  long time. Consequence of the stale divergence: ``has_occurred()`` with no
  arguments returned ``False`` unconditionally, so ``Swap.is_expired()``,
  ``CapFloor.is_expired()``, ``YoYInflationCapFloor.is_expired()`` and
  ``SyntheticCDO.is_expired()`` could never report expiry.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.event import Event
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.time.date import Date

# Module-level null Date used as default for optional Date arguments
# (avoids ruff B008 by not constructing per call). Date is frozen+slots
# so this is safe to share.
_NULL_DATE: Date = Date()


class CashFlow(Event, ABC):
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

        # C++ parity: ``CashFlow::hasOccurred`` (ql/cashflow.cpp:26-49),
        # reproduced branch for branch: the quick strictly-before /
        # strictly-after answers first, then the "today" override from
        # ``Settings::instance().includeTodaysCashFlows()``, then delegation
        # to ``Event::hasOccurred``.
        #
        # ``ref_date=None`` and the null ``Date()`` are both the C++ "not
        # supplied" sentinel.
        """
        settings = ObservableSettings()

        # easy and quick handling of most cases
        if ref_date is not None and ref_date != _NULL_DATE:
            cf_date = self.date()
            if ref_date < cf_date:
                return False
            if cf_date < ref_date:
                return True
            supplied = True
        else:
            supplied = False

        if not supplied or ref_date == settings.evaluation_date_or_today():
            # today's date; we override the bool with the one
            # specified in the settings (if any)
            include_today = settings.include_todays_cash_flows
            if include_today is not None:
                include_ref_date = include_today

        return super().has_occurred(ref_date, include_ref_date)

    def trading_ex_coupon(self, ref_date: Date | None = None) -> bool:
        """Whether the cash flow is trading ex-coupon on ``ref_date``.

        # C++ parity: ``CashFlow::tradingExCoupon`` (ql/cashflow.cpp:51-61) —
        # an unsupplied ``ref_date`` falls back to the global evaluation date.
        """
        ecd = self.ex_coupon_date()
        if ecd == _NULL_DATE:
            return False
        if ref_date is None or ref_date == _NULL_DATE:
            ref = ObservableSettings().evaluation_date_or_today()
        else:
            ref = ref_date
        return ecd <= ref
