"""YoYInflationCoupon — coupon paying a YoY inflation rate.

# C++ parity: ql/cashflows/yoyinflationcoupon.{hpp,cpp} (v1.42.1).

The coupon's rate is set by a ``YoYInflationCouponPricer`` to
``gearing * adjusted_fixing + spread``, where ``adjusted_fixing`` is the
pricer-modified version of the YoY index fixing (no adjustment in the
default pricer). The index fixing itself is the lagged YoY rate
``CPI::laggedYoYRate(index, accrualEndDate, observationLag, interpolation)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.cashflows.inflation_coupon import InflationCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType, lagged_yoy_rate
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.time.date import Date
from pquantlib.time.period import Period

if TYPE_CHECKING:
    from pquantlib.cashflows.inflation_coupon_pricer import InflationCouponPricer


class YoYInflationCoupon(InflationCoupon):
    """Coupon paying ``nominal * (gearing * YoYrate + spread) * accrualPeriod``.

    # C++ parity: ``YoYInflationCoupon`` in yoyinflationcoupon.hpp.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        fixing_days: int,
        index: YoYInflationIndex,
        observation_lag: Period,
        interpolation: InterpolationType,
        day_counter: DayCounter,
        gearing: float = 1.0,
        spread: float = 0.0,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> None:
        super().__init__(
            payment_date=payment_date,
            nominal=nominal,
            accrual_start_date=accrual_start_date,
            accrual_end_date=accrual_end_date,
            fixing_days=fixing_days,
            index=index,
            observation_lag=observation_lag,
            day_counter=day_counter,
            ref_period_start=ref_period_start,
            ref_period_end=ref_period_end,
        )
        self._yoy_index: YoYInflationIndex = index
        self._interpolation: InterpolationType = interpolation
        self._gearing: float = gearing
        self._spread: float = spread

    # ---- inspectors --------------------------------------------------

    def gearing(self) -> float:
        """C++ parity: ql/cashflows/yoyinflationcoupon.hpp:54 (inline)."""
        return self._gearing

    def spread(self) -> float:
        """C++ parity: ql/cashflows/yoyinflationcoupon.hpp:56 (inline)."""
        return self._spread

    def yoy_index(self) -> YoYInflationIndex:
        """C++ parity: ql/cashflows/yoyinflationcoupon.hpp:80-83 (inline)."""
        return self._yoy_index

    def interpolation(self) -> InterpolationType:
        """C++ parity: ql/cashflows/yoyinflationcoupon.hpp:85-87 (inline)."""
        return self._interpolation

    # ---- InflationCoupon overrides -----------------------------------

    def index_fixing(self) -> float:
        """C++ parity: ql/cashflows/yoyinflationcoupon.cpp:62-64."""
        return lagged_yoy_rate(
            self._yoy_index,
            self._accrual_end_date,
            self._observation_lag,
            self._interpolation,
        )

    def check_pricer_impl(self, pricer: InflationCouponPricer) -> bool:
        """C++ parity: ql/cashflows/yoyinflationcoupon.cpp:56-60.

        Accepts only ``YoYInflationCouponPricer`` subtypes.
        """
        # Local import to avoid the coupon ↔ pricer cycle.
        from pquantlib.cashflows.yoy_inflation_coupon_pricer import (  # noqa: PLC0415
            YoYInflationCouponPricer,
        )

        return isinstance(pricer, YoYInflationCouponPricer)

    def adjusted_fixing(self) -> float:
        """``(rate - spread) / gearing``.

        # C++ parity: ql/cashflows/yoyinflationcoupon.hpp:89-91 (inline).
        """
        return (self.rate() - self._spread) / self._gearing
