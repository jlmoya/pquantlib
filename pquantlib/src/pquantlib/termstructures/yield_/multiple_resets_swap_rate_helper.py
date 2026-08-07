"""MultipleResetsSwapRateHelper — bootstrap from multiple-resets swap quotes.

# C++ parity: ql/termstructures/yield/multipleresetsswaphelper.{hpp,cpp} (v1.43).

The quote is the fixed rate of a swap whose floating leg compounds or averages
``resets_per_coupon`` consecutive ibor fixings inside each accrual period
(:class:`~pquantlib.instruments.multiple_resets_swap.MultipleResetsSwap`).
``implied_quote`` is ``swap.fair_rate()``.

The date triple is short but not obvious (multipleresetsswaphelper.cpp:69-71)::

    earliest_date = swap.start_date()
    latest_relevant_date = latest_date
        = max(fixed_leg[-1].date(), floating_leg[-1].date())

Those are PAYMENT dates, not accrual ends, so a fixed leg rolled on a different
business-day convention from the floating one can win the max on its own.
There is no ``Pillar::Choice`` in this helper at all — ``maturity_date`` falls
through to ``latest_relevant_date`` and ``pillar_date`` to ``latest_date``
(bootstraphelper.hpp:179-203), which is reproduced here by leaving
``_maturity_date`` and ``_pillar_date`` unset.

C++ clones the ibor index onto ``termStructureHandle_`` and then explicitly
``unregisterWith``es it so the clone does not perturb the bootstrap
(multipleresetsswaphelper.cpp:45-49); the discount handle is relinked to
either the exogenous curve or the curve being bootstrapped
(multipleresetsswaphelper.cpp:74-83).  This port has no ``Handle`` (see the
``Handle`` entry in ``migration-harness/check_coverage.py``'s allowlist), so
the swap is rebuilt inside :meth:`implied_quote` against the curve just set —
the same semantics without the observer plumbing.  C++'s
``simplifyNotificationGraph`` call is likewise pure observer bookkeeping and
has no effect on any number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.termstructures.bootstrap_helper import RelativeDateBootstrapHelper
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.instruments.multiple_resets_swap import MultipleResetsSwap
    from pquantlib.quotes.quote import Quote
    from pquantlib.time.period import Period


class MultipleResetsSwapRateHelper(
    RelativeDateBootstrapHelper[YieldTermStructureProtocol]
):
    """Rate helper bootstrapping from multiple-resets swap quotes.

    # C++ parity: ``class MultipleResetsSwapRateHelper : public
    # RelativeDateRateHelper`` (multipleresetsswaphelper.hpp:36-71).
    """

    def __init__(
        self,
        settlement_days: int,
        tenor: Period,
        fixed_rate: Quote | float,
        ibor_index: IborIndex,
        resets_per_coupon: int,
        discounting_curve: YieldTermStructureProtocol | None = None,
        averaging_method: RateAveraging = RateAveraging.Compound,
        spread: float = 0.0,
        fixed_frequency: Frequency = Frequency.NoFrequency,
        fixed_day_count: DayCounter | None = None,
        fixed_convention: BusinessDayConvention = BusinessDayConvention.ModifiedFollowing,
    ) -> None:
        super().__init__(fixed_rate)
        self._settlement_days: int = settlement_days
        self._tenor: Period = tenor
        self._ibor_index: IborIndex = ibor_index
        self._resets_per_coupon: int = resets_per_coupon
        self._averaging_method: RateAveraging = averaging_method
        self._spread: float = spread
        self._fixed_frequency: Frequency = fixed_frequency
        # C++ parity: multipleresetsswaphelper.cpp:42 — an empty DayCounter
        # means "use the index's". ``None`` is this port's spelling of empty.
        self._fixed_day_count: DayCounter = (
            fixed_day_count if fixed_day_count is not None else ibor_index.day_counter()
        )
        self._fixed_convention: BusinessDayConvention = fixed_convention
        self._discount_curve: YieldTermStructureProtocol | None = discounting_curve
        self._swap: MultipleResetsSwap | None = None

        self._initialize_dates()
        from pquantlib.patterns.observable_settings import (  # noqa: PLC0415
            ObservableSettings,
        )

        self._eval_date_at_last_init = ObservableSettings().evaluation_date_or_today()

    # --- swap construction -----------------------------------------------

    def _make_swap(
        self, term_structure: YieldTermStructureProtocol | None
    ) -> MultipleResetsSwap:
        """Build the underlying swap.

        # C++ parity: multipleresetsswaphelper.cpp:57-65 — the single
        # MakeMultipleResetsSwap chain that both ``initializeDates`` and
        # (through ``swap_``) ``impliedQuote`` observe.
        """
        # Local import: termstructures/ should not depend on instruments/ at
        # module-load time.
        from pquantlib.instruments.make_multiple_resets_swap import (  # noqa: PLC0415
            MakeMultipleResetsSwap,
        )

        # C++ parity: multipleresetsswaphelper.cpp:48 — the index is cloned
        # onto the curve being bootstrapped (an empty handle while the dates
        # are being computed, hence ``None`` here).
        index = self._ibor_index.clone(term_structure)

        builder = (
            MakeMultipleResetsSwap(self._tenor, index, self._resets_per_coupon)
            # A fixed rate of 0.0 (C++'s choice too) keeps the build curve-free:
            # MakeMultipleResetsSwap only needs an engine when it has to solve
            # for a par rate, and none of the dates depend on one.
            .with_fixed_rate(0.0)
            .with_settlement_days(self._settlement_days)
            .with_fixed_leg_frequency(self._fixed_frequency)
            .with_fixed_leg_day_count(self._fixed_day_count)
            .with_fixed_leg_convention(self._fixed_convention)
            .with_floating_leg_spread(self._spread)
            .with_averaging_method(self._averaging_method)
        )
        # C++ parity: multipleresetsswaphelper.cpp:65 + 74-83 — the engine
        # discounts on the exogenous curve when one was supplied, otherwise on
        # the curve being bootstrapped. While the dates are being computed both
        # are absent, which is C++'s empty ``discountRelinkableHandle_``.
        discount = (
            self._discount_curve if self._discount_curve is not None else term_structure
        )
        if discount is not None:
            builder = builder.with_discounting_term_structure(discount)
        return builder.build()

    # --- dates -------------------------------------------------------------

    def _initialize_dates(self) -> None:
        """# C++ parity: ``initializeDates`` (multipleresetsswaphelper.cpp:56-72)."""
        swap = self._make_swap(None)
        self._swap = swap
        self._earliest_date = swap.start_date()
        # C++ parity: multipleresetsswaphelper.cpp:70-71 — the max of the two
        # legs' final PAYMENT dates, not of their accrual ends.
        self._latest_relevant_date = max(
            swap.fixed_leg()[-1].date(), swap.floating_leg()[-1].date()
        )
        self._latest_date = self._latest_relevant_date
        # maturityDate_ and pillarDate_ are deliberately left unset: C++ never
        # assigns them, so both accessors fall through (bootstraphelper.hpp).

    # --- BootstrapHelper interface ----------------------------------------

    def implied_quote(self) -> float:
        """Fair fixed rate of the underlying swap.

        # C++ parity: ``impliedQuote`` (multipleresetsswaphelper.cpp:85-89) —
        # ``swap_->deepUpdate(); return swap_->fairRate();``.
        """
        qassert.require(
            self._term_structure is not None,
            "MultipleResetsSwapRateHelper: term structure not set",
        )
        ts = self._term_structure
        assert ts is not None
        self._swap = self._make_swap(ts)
        return self._swap.fair_rate()

    # --- inspectors --------------------------------------------------------

    def swap(self) -> MultipleResetsSwap:
        """The underlying multiple-resets swap.

        Port addition: C++ keeps ``swap_`` protected with no inspector, but
        without a ``Handle`` the swap is a real object here and the
        cross-validation tests need to read its legs.
        """
        assert self._swap is not None
        return self._swap

    def resets_per_coupon(self) -> int:
        return self._resets_per_coupon

    def averaging_method(self) -> RateAveraging:
        return self._averaging_method

    def spread(self) -> float:
        return self._spread


__all__ = ["MultipleResetsSwapRateHelper"]
