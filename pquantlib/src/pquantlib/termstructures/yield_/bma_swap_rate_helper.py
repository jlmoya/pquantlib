"""BMASwapRateHelper — bootstrap from a quoted BMA/Libor fraction.

# C++ parity: ql/termstructures/yield/ratehelpers.{hpp,cpp} class
# BMASwapRateHelper (v1.43).

The quote is the fraction of Libor that a BMA swap of the given tenor trades
at.  The helper builds a :class:`~pquantlib.instruments.bma_swap.BMASwap` with
an *arbitrary* Libor fraction of 0.75 (C++'s own comment) and reports
``swap.fair_libor_fraction()`` as the implied quote.

Two things in here are easy to get wrong and are worth stating plainly.

**The dates are not the swap's dates.**  ``earliest_date`` rolls the evaluation
date on the JOINT calendar of the helper calendar and the ibor index's fixing
calendar, then advances ``settlement_days`` on the helper calendar *alone*
(ratehelpers.cpp:678-682); the two steps use different calendars deliberately,
so any evaluation date that is a holiday on exactly one of them separates them.
``latest_date`` is not the swap maturity either — it is the value date of the
Wednesday *after* the adjusted swap maturity (ratehelpers.cpp:716-722), i.e.
between 2 and 8 days later, and the ``>= 4`` weekday branch means a maturity
that already falls on a Wednesday jumps a full further week.

**Three different curves are in play.**  The BMA leg forecasts off the curve
being bootstrapped; the Libor leg forecasts off, and the swap discounts on,
the *ibor index's own* forwarding curve (ratehelpers.cpp:713-714).  C++ gets
this by handing the BMA leg a clone of the index linked to
``termStructureHandle_`` while the engine holds ``iborIndex_->
forwardingTermStructure()``.  This port has no ``Handle`` (see the ``Handle``
entry in ``migration-harness/check_coverage.py``'s allowlist), so instead the
swap is rebuilt inside :meth:`implied_quote` with a fresh ``BMAIndex`` bound to
the term structure just set — same semantics, no pointer plumbing.  The dates
computed at construction are reused verbatim, so the rebuilt swap is the same
instrument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.indexes.bma_index import BMAIndex
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.termstructures.bootstrap_helper import RelativeDateBootstrapHelper
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.joint_calendar import JointCalendar
from pquantlib.time.date import Date
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.instruments.bma_swap import BMASwap
    from pquantlib.quotes.quote import Quote
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule

#: C++'s own "arbitrary" gearing on the Libor leg (ratehelpers.cpp:706).
#: ``fairLiborFraction`` divides it back out, so the value is immaterial —
#: but it has to be non-zero, and it has to match C++ bit for bit because the
#: division is done in floating point.
_ARBITRARY_LIBOR_FRACTION = 0.75
_ARBITRARY_LIBOR_SPREAD = 0.0
_NOMINAL = 100.0


class BMASwapRateHelper(RelativeDateBootstrapHelper[YieldTermStructureProtocol]):
    """Rate helper bootstrapping over BMA swap rates.

    # C++ parity: ``class BMASwapRateHelper : public RelativeDateRateHelper``
    # (ratehelpers.hpp:294-321).
    """

    def __init__(
        self,
        libor_fraction: Quote | float,
        tenor: Period,
        settlement_days: int,
        calendar: Calendar,
        # BMA leg
        bma_period: Period,
        bma_convention: BusinessDayConvention,
        bma_day_count: DayCounter,
        bma_index: BMAIndex,
        # Libor leg
        ibor_index: IborIndex,
    ) -> None:
        super().__init__(libor_fraction)
        self._tenor: Period = tenor
        self._settlement_days: int = settlement_days
        self._calendar: Calendar = calendar
        self._bma_period: Period = bma_period
        self._bma_convention: BusinessDayConvention = bma_convention
        self._bma_day_count: DayCounter = bma_day_count
        self._bma_index: BMAIndex = bma_index
        self._ibor_index: IborIndex = ibor_index
        # Cached so that ``implied_quote`` rebuilds the SAME swap the dates
        # were read off, rather than re-deriving them from a global that may
        # have moved since.
        self._swap_maturity: Date = Date()
        self._swap: BMASwap | None = None

        # C++ registers with both indexes (ratehelpers.cpp:670-671); this port
        # threads the objects directly, so there is nothing to relink.
        self._initialize_dates()
        self._eval_date_at_last_init = ObservableSettings().evaluation_date_or_today()

    # --- swap construction -----------------------------------------------

    def _make_swap(self, term_structure: YieldTermStructureProtocol | None) -> BMASwap:
        """Build the underlying BMA swap.

        # C++ parity: ratehelpers.cpp:686-714 — the block shared by
        # ``initializeDates`` and (through ``swap_``) ``impliedQuote``.

        ``term_structure`` is the curve being bootstrapped: the BMA leg, and
        only the BMA leg, forecasts off it.
        """
        # Local import: termstructures/ should not depend on instruments/ or
        # pricingengines/ at module-load time.
        from pquantlib.instruments.bma_swap import BMASwap  # noqa: PLC0415
        from pquantlib.instruments.swap import SwapType  # noqa: PLC0415
        from pquantlib.pricingengines.swap.discounting_swap_engine import (  # noqa: PLC0415
            DiscountingSwapEngine,
        )

        # C++ parity: ratehelpers.cpp:686-687 — a dummy BMA index bound to the
        # curve under construction. The clone shares the fixing history with
        # ``bma_index`` because IndexManager keys it by index name.
        cloned_index = BMAIndex(term_structure)  # type: ignore[arg-type]

        swap = BMASwap(
            SwapType.Payer,
            _NOMINAL,
            self._libor_schedule(),
            _ARBITRARY_LIBOR_FRACTION,
            _ARBITRARY_LIBOR_SPREAD,
            self._ibor_index,
            self._ibor_index.day_counter(),
            self._bma_schedule(),
            cloned_index,
            self._bma_day_count,
        )
        # C++ parity: ratehelpers.cpp:713-714 — the engine discounts on the
        # IBOR index's own curve, not on the curve being bootstrapped.
        forwarding = self._ibor_index.forecast_term_structure()
        if forwarding is not None:
            swap.set_pricing_engine(DiscountingSwapEngine(forwarding))
        return swap

    def _bma_schedule(self) -> Schedule:
        """# C++ parity: ratehelpers.cpp:689-694."""
        return (
            MakeSchedule()
            .from_date(self.earliest_date())
            .to(self._swap_maturity)
            .with_tenor(self._bma_period)
            .with_calendar(self._bma_index.fixing_calendar())
            .with_convention(self._bma_convention)
            .backwards()
            .build()
        )

    def _libor_schedule(self) -> Schedule:
        """# C++ parity: ratehelpers.cpp:696-702."""
        return (
            MakeSchedule()
            .from_date(self.earliest_date())
            .to(self._swap_maturity)
            .with_tenor(self._ibor_index.tenor())
            .with_calendar(self._ibor_index.fixing_calendar())
            .with_convention(self._ibor_index.business_day_convention())
            .with_end_of_month(self._ibor_index.end_of_month())
            .backwards()
            .build()
        )

    # --- dates -------------------------------------------------------------

    def _initialize_dates(self) -> None:
        """# C++ parity: ``BMASwapRateHelper::initializeDates`` (ratehelpers.cpp:675-723)."""
        today = ObservableSettings().evaluation_date_or_today()

        # C++ parity: ratehelpers.cpp:676-680 — if the evaluation date is not a
        # business day on BOTH the helper calendar and the index's fixing
        # calendar, move to the next day that is.
        joint = JointCalendar([self._calendar, self._ibor_index.fixing_calendar()])
        reference_date = joint.adjust(today)

        # C++ parity: ratehelpers.cpp:681-682 — the settlement advance is on the
        # helper calendar ALONE, not the joint one.
        self._earliest_date = self._calendar.advance(
            reference_date,
            self._settlement_days,
            TimeUnit.Days,
            BusinessDayConvention.Following,
        )
        # C++ parity: ratehelpers.cpp:684 — unadjusted date arithmetic.
        self._swap_maturity = self._earliest_date + self._tenor

        self._swap = self._make_swap(None)

        # C++ parity: ratehelpers.cpp:716-722.
        #   d             = calendar_.adjust(swap_->maturityDate(), Following)
        #   nextWednesday = (w >= 4) ? d + (11 - w) : d + (4 - w)
        #   latestDate_   = clonedIndex->valueDate(
        #                       clonedIndex->fixingCalendar().adjust(nextWednesday))
        # Weekday integers match C++ (Sunday=1 .. Saturday=7, Wednesday=4), so
        # a maturity that is ALREADY a Wednesday takes the ``>= 4`` branch and
        # rolls a full week rather than staying put.
        d = self._calendar.adjust(
            self._swap.maturity_date(), BusinessDayConvention.Following
        )
        w = int(d.weekday())
        next_wednesday = d + (11 - w) if w >= 4 else d + (4 - w)
        cloned_index = BMAIndex()
        self._latest_date = cloned_index.value_date(
            cloned_index.fixing_calendar().adjust(next_wednesday)
        )
        # C++ leaves maturityDate_ / latestRelevantDate_ / pillarDate_ unset, so
        # all three fall through to latestDate() (bootstraphelper.hpp:179-203).
        # Leaving them None here reproduces exactly that.

    # --- BootstrapHelper interface ----------------------------------------

    def implied_quote(self) -> float:
        """Fair Libor fraction of the underlying BMA swap.

        # C++ parity: ``BMASwapRateHelper::impliedQuote`` (ratehelpers.cpp:736-741)
        # — ``swap_->deepUpdate(); return swap_->fairLiborFraction();``.
        """
        qassert.require(
            self._term_structure is not None,
            "BMASwapRateHelper: term structure not set",
        )
        ts = self._term_structure
        assert ts is not None
        qassert.require(
            self._ibor_index.forecast_term_structure() is not None,
            f"null term structure set to this instance of {self._ibor_index.name()}",
        )
        # C++ relinks termStructureHandle_ under the already-built swap
        # (setTermStructure, ratehelpers.cpp:725-734); with no Handle the swap
        # is rebuilt on the new curve instead. Same instrument: every date it
        # depends on was cached at construction.
        self._swap = self._make_swap(ts)
        return self._swap.fair_libor_fraction()

    # --- inspectors --------------------------------------------------------

    def swap(self) -> BMASwap:
        """The underlying BMA swap.

        Port addition: C++ keeps ``swap_`` protected with no inspector, but
        without a ``Handle`` the swap is a real observable object here and the
        cross-validation tests need to read its schedules.
        """
        assert self._swap is not None
        return self._swap

    def bma_index(self) -> BMAIndex:
        return self._bma_index

    def ibor_index(self) -> IborIndex:
        return self._ibor_index


__all__ = ["BMASwapRateHelper"]
