"""Pathwise (Greeks-aware) coterminal swaptions.

# C++ parity:
# ql/models/marketmodels/products/pathwise/pathwiseproductswaption.{hpp,cpp}
# (v1.42.1).

Payer, coterminal swaptions used mainly to test market pathwise vegas. Two
classes:

- ``MarketModelPathwiseCoterminalSwaptionsDeflated`` — the analytic pathwise
  version. At step ``i`` product ``i`` pays ``(swapRate_i - strike_i) *
  annuity_i`` (when positive), with the closed-form derivative w.r.t. each
  forward ``k >= i`` assembled from the discount-ratio geometry.
- ``MarketModelPathwiseCoterminalSwaptionsNumericalDeflated`` — the same value
  but with derivatives computed by central finite differences (bump each
  forward by ``bumpSize`` up/down and re-price). The easiest way to test the
  analytic version.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
    PathwiseCashFlow,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class MarketModelPathwiseCoterminalSwaptionsDeflated(MarketModelPathwiseMultiProduct):
    """Analytic pathwise coterminal payer swaptions.

    # C++ parity:
    # pathwiseproductswaption.hpp MarketModelPathwiseCoterminalSwaptionsDeflated.
    """

    def __init__(self, rate_times: list[float], strikes: list[float]) -> None:
        self._rate_times = list(rate_times)
        self._strikes = list(strikes)
        self._number_rates = len(rate_times) - 1
        self._current_index = 0
        check_increasing_times(rate_times)
        evol_times = list(self._rate_times[:-1])
        qassert.require(
            len(evol_times) == self._number_rates, "rateTimes.size()<> numberOfRates+1"
        )
        qassert.require(
            len(strikes) == self._number_rates, "strikes.size()<> numberOfRates"
        )
        self._evolution = EvolutionDescription(rate_times, evol_times)

    def already_deflated(self) -> bool:
        return False

    def suggested_numeraires(self) -> list[int]:
        return list(range(self._number_rates))

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return self._rate_times

    def number_of_products(self) -> int:
        return self._number_rates

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._current_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[PathwiseCashFlow]],
    ) -> bool:
        ci = self._current_index
        rt = self._rate_times
        swap_rate = current_state.coterminal_swap_rate(ci)
        cf = cash_flows_generated[ci][0]
        cf.time_index = ci

        annuity = current_state.coterminal_swap_annuity(ci, ci)
        cf.amount[0] = (swap_rate - self._strikes[ci]) * annuity

        for k in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[k] = 0

        if cf.amount[0] > 0:
            number_cash_flows_this_step[ci] = 1
            for i in range(1, self._number_rates + 1):
                cf.amount[i] = 0.0

            for k in range(ci, self._number_rates):
                cf.amount[k + 1] = (rt[k + 1] - rt[k]) * current_state.discount_ratio(
                    k + 1, ci
                )
                multiplier = -(rt[k + 1] - rt[k]) * current_state.discount_ratio(k + 1, k)
                for ell in range(k, self._number_rates):
                    cf.amount[k + 1] += (
                        (current_state.forward_rate(ell) - self._strikes[ci])
                        * (rt[ell + 1] - rt[ell])
                        * multiplier
                        * current_state.discount_ratio(ell + 1, ci)
                    )
        self._current_index += 1
        return self._current_index == len(self._strikes)

    def clone(self) -> MarketModelPathwiseMultiProduct:
        return MarketModelPathwiseCoterminalSwaptionsDeflated(
            self._rate_times, self._strikes
        )


class MarketModelPathwiseCoterminalSwaptionsNumericalDeflated(
    MarketModelPathwiseMultiProduct
):
    """Numerical-FD pathwise coterminal payer swaptions (analytic-value test).

    # C++ parity:
    # pathwiseproductswaption.hpp
    # MarketModelPathwiseCoterminalSwaptionsNumericalDeflated.
    """

    def __init__(
        self, rate_times: list[float], strikes: list[float], bump_size: float
    ) -> None:
        self._rate_times = list(rate_times)
        self._strikes = list(strikes)
        self._number_rates = len(rate_times) - 1
        self._bump_size = bump_size
        self._up = LMMCurveState(rate_times)
        self._down = LMMCurveState(rate_times)
        self._forwards = [0.0] * self._number_rates
        self._current_index = 0
        check_increasing_times(rate_times)
        evol_times = list(self._rate_times[:-1])
        qassert.require(
            len(evol_times) == self._number_rates, "rateTimes.size()<> numberOfRates+1"
        )
        qassert.require(
            len(strikes) == self._number_rates, "strikes.size()<> numberOfRates"
        )
        self._evolution = EvolutionDescription(rate_times, evol_times)

    def already_deflated(self) -> bool:
        return False

    def suggested_numeraires(self) -> list[int]:
        return list(range(self._number_rates))

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return self._rate_times

    def number_of_products(self) -> int:
        return self._number_rates

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._current_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[PathwiseCashFlow]],
    ) -> bool:
        ci = self._current_index
        swap_rate = current_state.coterminal_swap_rate(ci)
        cf = cash_flows_generated[ci][0]
        cf.time_index = ci

        annuity = current_state.coterminal_swap_annuity(ci, ci)
        cf.amount[0] = (swap_rate - self._strikes[ci]) * annuity

        for k in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[k] = 0

        if cf.amount[0] > 0:
            number_cash_flows_this_step[ci] = 1
            for i in range(1, self._number_rates + 1):
                cf.amount[i] = 0.0

            for k in range(ci, self._number_rates):
                self._forwards = current_state.forward_rates()
                self._forwards[k] += self._bump_size
                self._up.set_on_forward_rates(self._forwards)

                self._forwards[k] -= self._bump_size
                self._forwards[k] -= self._bump_size
                self._down.set_on_forward_rates(self._forwards)

                up_sr = self._up.coterminal_swap_rate(ci)
                up_annuity = self._up.coterminal_swap_annuity(ci, ci)
                up_value = (up_sr - self._strikes[ci]) * up_annuity

                down_sr = self._down.coterminal_swap_rate(ci)
                down_annuity = self._down.coterminal_swap_annuity(ci, ci)
                down_value = (down_sr - self._strikes[ci]) * down_annuity

                deriv = (up_value - down_value) / (2.0 * self._bump_size)
                cf.amount[k + 1] = deriv
        self._current_index += 1
        return self._current_index == len(self._strikes)

    def clone(self) -> MarketModelPathwiseMultiProduct:
        return MarketModelPathwiseCoterminalSwaptionsNumericalDeflated(
            self._rate_times, self._strikes, self._bump_size
        )
