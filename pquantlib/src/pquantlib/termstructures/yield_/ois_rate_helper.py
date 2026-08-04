"""OISRateHelper — bootstrap from OIS rate quote.

# C++ parity: ql/termstructures/yield/oisratehelper.{hpp,cpp} class OISRateHelper.

C++ ``OISRateHelper`` builds an ``OvernightIndexedSwap`` via ``MakeOIS`` and
calls ``swap_->fairRate()`` for ``impliedQuote``. L3-C closes the carry-over:
``implied_quote`` now delegates to ``make_ois`` + ``swap.fair_rate()``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper, PillarChoice
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_ZERO_PERIOD = Period(0, TimeUnit.Days)


class OISRateHelper(BootstrapHelper[YieldTermStructureProtocol]):
    """OIS rate helper. Full ``implied_quote`` deferred to L3 (OvernightIndexedSwap)."""

    def __init__(
        self,
        settlement_days: int,
        tenor: Period,
        fixed_rate: Quote | float,
        overnight_index: OvernightIndex,
        discount_curve: YieldTermStructureProtocol | None = None,
        telescopic_value_dates: bool = False,
        payment_lag: int = 0,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        forward_start: Period = _ZERO_PERIOD,
        pillar: PillarChoice = PillarChoice.LastRelevantDate,
        custom_pillar_date: Date | None = None,
        end_of_month: bool | None = None,
        evaluation_date: Date | None = None,
    ) -> None:
        super().__init__(fixed_rate)
        self._settlement_days: int = settlement_days
        self._tenor: Period = tenor
        self._overnight_index: OvernightIndex = overnight_index
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        self._telescopic_value_dates: bool = telescopic_value_dates
        self._payment_lag: int = payment_lag
        self._payment_convention: BusinessDayConvention = payment_convention
        self._fwd_start: Period = forward_start
        self._end_of_month: bool | None = end_of_month
        self._pillar_choice: PillarChoice = pillar
        if custom_pillar_date is not None:
            self._pillar_date = custom_pillar_date
        if evaluation_date is not None:
            self.initialize_dates(evaluation_date)

    # --- BootstrapHelper interface --------------------------------------------

    def implied_quote(self) -> float:
        """Implied OIS rate from the underlying OvernightIndexedSwap.

        # C++ parity: ``OISRateHelper::impliedQuote`` (oisratehelper.cpp) —
        # ``swap_->fairRate()``.
        """
        # Local import: termstructures/ should not depend on instruments/.
        from pquantlib.instruments.make_ois import make_ois  # noqa: PLC0415

        qassert.require(
            self._term_structure is not None,
            "OISRateHelper: term structure not set yet",
        )
        ts = self._term_structure
        assert ts is not None
        idx = (
            self._overnight_index.clone(ts)
            if hasattr(self._overnight_index, "clone")
            else self._overnight_index
        )
        swap = make_ois(
            swap_tenor=self._tenor,
            overnight_index=idx,
            fixed_rate=None,
            forward_start=self._fwd_start,
            settlement_days=self._settlement_days,
            payment_lag=self._payment_lag,
            payment_adjustment=self._payment_convention,
            telescopic_value_dates=self._telescopic_value_dates,
            end_of_month=self._end_of_month,
            discount_curve=self._discount_curve if self._discount_curve is not None else ts,
            evaluation_date=ts.reference_date(),
        )
        return swap.fair_rate()

    # --- dates ---------------------------------------------------------------

    def initialize_dates(self, evaluation_date: Date) -> None:
        cal = self._overnight_index.fixing_calendar()
        ref = cal.adjust(evaluation_date, BusinessDayConvention.Following)
        spot = cal.advance(ref, self._settlement_days, TimeUnit.Days)
        if self._fwd_start.length != 0:
            earliest = cal.advance(
                spot, self._fwd_start.length, self._fwd_start.units,
                self._payment_convention, self._end_of_month or False,
            )
        else:
            earliest = spot
        maturity = cal.advance(
            earliest, self._tenor.length, self._tenor.units,
            self._payment_convention, self._end_of_month or False,
        )
        self._earliest_date = earliest
        self._maturity_date = maturity

        # C++ parity: oisratehelper.cpp:171-176 —
        #   latestRelevantDate_ = latestDate_
        #     = max(maturityDate_, lastPaymentDate, fixingEndDate)
        #
        # ``lastPaymentDate`` is the later of the two legs' final payment
        # dates. Both legs are paid ``payment_lag`` business days after the
        # accrual end, on the payment calendar — which for an OIS built by
        # ``make_ois`` is the overnight index's fixing calendar. A lag of 0
        # collapses to ``adjust(maturity)``, i.e. maturity itself.
        #
        # ``fixingEndDate`` is
        # ``index.maturity_date(index.value_date(last_fixing_date))``. For an
        # overnight index — zero fixing days, one-business-day tenor — the
        # last fixing's value date is the coupon's penultimate value date and
        # its maturity is the final one, i.e. the adjusted accrual end. So
        # that term is already covered by ``maturity`` and adds nothing here.
        # (It would not be, for the lookback / observation-shift variants C++
        # supports; this port's OvernightIndexedCoupon implements none of
        # them, and the helper does not accept them either.)
        last_payment_date = cal.advance(
            maturity, self._payment_lag, TimeUnit.Days, self._payment_convention
        )
        self._latest_relevant_date = max(maturity, last_payment_date)
        self._latest_date = self._latest_relevant_date

        # C++ parity: oisratehelper.cpp:178-195.
        if self._pillar_choice == PillarChoice.MaturityDate:
            self._pillar_date = maturity
        elif self._pillar_choice == PillarChoice.LastRelevantDate:
            self._pillar_date = self._latest_relevant_date
        elif self._pillar_choice == PillarChoice.CustomDate:
            # pillar_date already assigned at construction time
            qassert.require(
                self._pillar_date is not None,
                "CustomDate pillar requires custom_pillar_date argument",
            )
            assert self._pillar_date is not None
            qassert.require(
                self._pillar_date >= earliest,
                f"pillar date ({self._pillar_date}) must be later than or equal "
                f"to the instrument's earliest date ({earliest})",
            )
            qassert.require(
                self._pillar_date <= self._latest_relevant_date,
                f"pillar date ({self._pillar_date}) must be before or equal to "
                f"the instrument's latest relevant date "
                f"({self._latest_relevant_date})",
            )
        else:
            qassert.fail(f"unknown Pillar.Choice({int(self._pillar_choice)})")

    # --- inspectors ---------------------------------------------------------

    def telescopic_value_dates(self) -> bool:
        return self._telescopic_value_dates
