"""MultiStepPeriodCapletSwaptions — periodized caplets + swaptions product.

# C++ parity:
# ql/models/marketmodels/products/multistep/multistepperiodcapletswaptions.{hpp,cpp}
# (v1.42.1).

Bundles, for each "big" FRA (a period-spanning forward starting every
``period`` rates from ``offset``), both a forward-rate caplet and a coterminal
swaption on the remaining big FRAs. ``numberBigFRAs * 2`` products: the first
``numberBigFRAs`` are the caplets, the next ``numberBigFRAs`` the swaptions.
At each qualifying step both the caplet (struck via ``forwardPayOffs``) and the
swaption (struck via ``swapPayOffs``) are priced from the discount ratios of the
current state; only strictly-positive values generate a cash flow.
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
from pquantlib.models.marketmodels.utilities import check_increasing_times
from pquantlib.payoffs import StrikedTypePayoff


class MultiStepPeriodCapletSwaptions(MultiProductMultiStep):
    """Periodized caplets + coterminal swaptions (one pair per big FRA).

    # C++ parity:
    # multistepperiodcapletswaptions.hpp MultiStepPeriodCapletSwaptions.
    """

    def __init__(
        self,
        rate_times: list[float],
        forward_option_payment_times: list[float],
        swaption_payment_times: list[float],
        forward_payoffs: list[StrikedTypePayoff],
        swap_payoffs: list[StrikedTypePayoff],
        period: int,
        offset: int,
    ) -> None:
        super().__init__(rate_times)
        qassert.require(
            len(rate_times) >= 2,
            "we need at least two rate times in MultiStepPeriodCapletSwaptions ",
        )
        check_increasing_times(forward_option_payment_times)
        check_increasing_times(swaption_payment_times)
        self._forward_option_payment_times = list(forward_option_payment_times)
        self._swaption_payment_times = list(swaption_payment_times)
        self._forward_payoffs = list(forward_payoffs)
        self._swap_payoffs = list(swap_payoffs)
        self._period = period
        self._offset = offset
        # paymentTimes_ = forwardOptionPaymentTimes ++ swaptionPaymentTimes
        self._payment_times = list(forward_option_payment_times)
        self._payment_times.extend(self._swaption_payment_times)
        self._last_index = len(rate_times) - 1
        self._number_fras = len(rate_times) - 1
        self._number_big_fras = (self._number_fras - offset) // period
        self._current_index = 0
        self._product_index = 0

        qassert.require(
            offset < period,
            "the offset must be less then the period in MultiStepPeriodCapletSwaptions ",
        )
        qassert.require(
            self._number_big_fras > 0,
            "we must have at least one FRA after the periodizing in  "
            "MultiStepPeriodCapletSwaptions ",
        )
        qassert.require(
            len(self._forward_option_payment_times) == self._number_big_fras,
            "we must have precisely one payment time for each forward option  "
            "MultiStepPeriodCapletSwaptions ",
        )
        qassert.require(
            len(self._forward_payoffs) == self._number_big_fras,
            "we must have precisely one payoff  for each forward option  "
            "MultiStepPeriodCapletSwaptions ",
        )
        qassert.require(
            len(self._swaption_payment_times) == self._number_big_fras,
            "we must have precisely one payment time for each swaption in "
            "MultiStepPeriodCapletSwaptions ",
        )
        qassert.require(
            len(self._swap_payoffs) == self._number_big_fras,
            "we must have precisely one payoff  for each swaption in  "
            "MultiStepPeriodCapletSwaptions ",
        )

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return self._number_big_fras * 2

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._current_index = 0
        self._product_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0

        if (
            self._current_index >= self._offset
            and (self._current_index - self._offset) % self._period == 0
        ):
            ci = self._current_index
            period = self._period
            pidx = self._product_index

            # --- caplet first ---
            df = current_state.discount_ratio(ci + period, ci)
            tau = self._rate_times[ci + period] - self._rate_times[ci]
            forward = (1.0 / df - 1.0) / tau
            value = self._forward_payoffs[pidx](forward)
            value *= tau * current_state.discount_ratio(ci + period, ci)

            if value > 0:
                number_cash_flows_this_step[pidx] = 1
                cf = cash_flows_generated[pidx][0]
                cf.amount = value
                cf.time_index = pidx

            # --- now swaption ---
            number_periods = self._number_big_fras - pidx
            b = 0.0
            p0 = 1.0  # currentState.discountRatio(currentIndex_, currentIndex_)
            pn = current_state.discount_ratio(ci + number_periods * period, ci)
            for i in range(number_periods):
                tau_i = (
                    self._rate_times[ci + (i + 1) * period]
                    - self._rate_times[ci + i * period]
                )
                b += tau_i * current_state.discount_ratio(ci + (i + 1) * period, ci)

            swap_rate = (p0 - pn) / b
            swaption_value = self._swap_payoffs[pidx](swap_rate)
            swaption_value *= b

            if swaption_value > 0:
                number_cash_flows_this_step[pidx + self._number_big_fras] = 1
                cf = cash_flows_generated[pidx + self._number_big_fras][0]
                cf.amount = swaption_value
                cf.time_index = pidx + self._number_big_fras

            self._product_index += 1

        self._current_index += 1
        return self._product_index >= self._number_big_fras

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepPeriodCapletSwaptions(
            self._rate_times,
            self._forward_option_payment_times,
            self._swaption_payment_times,
            self._forward_payoffs,
            self._swap_payoffs,
            self._period,
            self._offset,
        )
