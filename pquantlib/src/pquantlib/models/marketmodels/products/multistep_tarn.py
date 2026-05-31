"""MultiStepTarn — a target-redemption note as a multi-step product.

# C++ parity:
# ql/models/marketmodels/products/multistep/multisteptarn.{hpp,cpp}
# (v1.42.1).

A single product. At step ``i`` it pays a floating coupon
``(libor + spread) * floatingAccrual`` (time index ``lastIndex + i`` into the
doubled payment-time vector) and *receives* an inverse-floating "obvious"
coupon ``max(strike - mult * libor, 0) * accrual`` (paid as a negative
amount at time index ``i``). The note terminates early once the cumulative
coupon reaches ``totalCoupon``; on the terminating step the final coupon is
topped up so the total paid coupon equals ``totalCoupon`` exactly.

Divergence from C++: ``allPaymentTimes_`` is the floating payment times
concatenated *after* the fixed payment times (the C++ source re-pushes
``paymentTimes`` while iterating over ``paymentTimesFloating_``; this port
preserves that exact doubled-vector behaviour, including the floating-leg
``lastIndex + i`` time index).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.multiproduct_multistep import (
    MultiProductMultiStep,
)


class MultiStepTarn(MultiProductMultiStep):
    """A target-redemption note (TARN).

    # C++ parity: multisteptarn.hpp MultiStepTarn.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        accruals_floating: list[float],
        payment_times: list[float],
        payment_times_floating: list[float],
        total_coupon: float,
        strikes: list[float],
        multipliers: list[float],
        floating_spreads: list[float],
    ) -> None:
        super().__init__(rate_times)
        self._accruals = list(accruals)
        self._accruals_floating = list(accruals_floating)
        self._payment_times = list(payment_times)
        self._payment_times_floating = list(payment_times_floating)
        self._total_coupon = total_coupon
        self._strikes = list(strikes)
        self._multipliers = list(multipliers)
        self._floating_spreads = list(floating_spreads)
        n = len(rate_times)
        qassert.require(
            len(accruals) + 1 == n, "missized accruals in MultiStepTARN"
        )
        qassert.require(
            len(accruals_floating) + 1 == n,
            "missized accrualsFloating in MultiStepTARN",
        )
        qassert.require(
            len(payment_times) + 1 == n, "missized paymentTimes in MultiStepTARN"
        )
        qassert.require(
            len(payment_times_floating) + 1 == n,
            "missized paymentTimesFloating in MultiStepTARN",
        )
        qassert.require(
            len(strikes) + 1 == n, "missized strikes in MultiStepTARN"
        )
        qassert.require(
            len(floating_spreads) + 1 == n,
            "missized floatingSpreads in MultiStepTARN",
        )
        self._last_index = len(accruals)
        # C++ parity: allPaymentTimes_ = paymentTimes; then for each floating
        # payment time push back paymentTimes[i] (i.e. doubles the fixed times).
        self._all_payment_times = list(payment_times)
        for i in range(len(self._payment_times_floating)):
            self._all_payment_times.append(payment_times[i])
        self._coupon_paid = 0.0
        self._current_index = 0

    def possible_cash_flow_times(self) -> list[float]:
        return self._all_payment_times

    def number_of_products(self) -> int:
        return 1

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 2

    def reset(self) -> None:
        self._current_index = 0
        self._coupon_paid = 0.0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        i = self._current_index
        libor_rate = current_state.forward_rate(i)

        number_cash_flows_this_step[0] = 2

        floating = cash_flows_generated[0][0]
        floating.amount = (libor_rate + self._floating_spreads[i]) * self._accruals_floating[i]
        floating.time_index = self._last_index + i

        fixed = cash_flows_generated[0][1]
        fixed.time_index = i

        obvious_coupon = max(self._strikes[i] - self._multipliers[i] * libor_rate, 0.0) * self._accruals[i]

        self._coupon_paid += obvious_coupon

        self._current_index += 1

        if self._coupon_paid < self._total_coupon and self._current_index < self._last_index:
            fixed.amount = -obvious_coupon
            return False

        coupon = obvious_coupon + (self._total_coupon - self._coupon_paid)
        fixed.amount = -coupon
        return True

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepTarn(
            self._rate_times,
            self._accruals,
            self._accruals_floating,
            self._payment_times,
            self._payment_times_floating,
            self._total_coupon,
            self._strikes,
            self._multipliers,
            self._floating_spreads,
        )
