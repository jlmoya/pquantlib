"""MarketModelPathwiseMultiCaplet — pathwise (Greeks-aware) caplets.

# C++ parity:
# ql/models/marketmodels/products/pathwise/pathwiseproductcaplet.{hpp,cpp}
# (v1.42.1).

Essentially a test class (we have better ways of computing caplet Greeks), but
the canonical demonstration of the pathwise methodology. Three classes:

- ``MarketModelPathwiseMultiCaplet`` — undeflated caplets. At step ``i`` product
  ``i`` pays ``(libor_i - strike_i) * accrual_i`` (when positive) with derivative
  ``accrual_i`` w.r.t. forward ``i`` (zero w.r.t. the others).
- ``MarketModelPathwiseMultiDeflatedCaplet`` — the same but pre-divided by the
  numeraire ``P(t_0)``, so the derivatives pick up the discount-ratio chain.
- ``MarketModelPathwiseMultiDeflatedCap`` — bundles deflated caplets into caps
  via ``(start, end)`` index ranges (mainly for testing pathwise market vegas).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
    PathwiseCashFlow,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class MarketModelPathwiseMultiCaplet(MarketModelPathwiseMultiProduct):
    """Undeflated pathwise caplets (one product per rate).

    # C++ parity: pathwiseproductcaplet.hpp MarketModelPathwiseMultiCaplet.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        payment_times: list[float],
        strikes: list[float],
    ) -> None:
        self._rate_times = list(rate_times)
        self._accruals = list(accruals)
        self._payment_times = list(payment_times)
        self._strikes = list(strikes)
        self._number_rates = len(self._accruals)
        self._current_index = 0
        check_increasing_times(rate_times)
        check_increasing_times(payment_times)
        evol_times = list(self._rate_times[:-1])
        qassert.require(
            len(evol_times) == self._number_rates, "rateTimes.size()<> numberOfRates+1"
        )
        qassert.require(
            len(payment_times) == self._number_rates,
            "paymentTimes.size()<> numberOfRates",
        )
        qassert.require(
            len(accruals) == self._number_rates, "accruals.size()<> numberOfRates"
        )
        qassert.require(
            len(strikes) == self._number_rates, "strikes.size()<> numberOfRates"
        )
        self._evolution = EvolutionDescription(rate_times, evol_times)

    def already_deflated(self) -> bool:
        return False

    def suggested_numeraires(self) -> list[int]:
        return [i + 1 for i in range(self._number_rates)]

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

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
        i = self._current_index
        libor_rate = current_state.forward_rate(i)
        cf = cash_flows_generated[i][0]
        cf.time_index = i
        cf.amount[0] = (libor_rate - self._strikes[i]) * self._accruals[i]

        for k in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[k] = 0

        if cf.amount[0] > 0:
            number_cash_flows_this_step[i] = 1
            for k in range(1, self._number_rates + 1):
                cf.amount[k] = 0.0
            cf.amount[i + 1] = self._accruals[i]
        self._current_index += 1
        return self._current_index == len(self._strikes)

    def clone(self) -> MarketModelPathwiseMultiProduct:
        return MarketModelPathwiseMultiCaplet(
            self._rate_times, self._accruals, self._payment_times, self._strikes
        )


class MarketModelPathwiseMultiDeflatedCaplet(MarketModelPathwiseMultiProduct):
    """Deflated pathwise caplets (pre-divided by the time-0 numeraire).

    # C++ parity:
    # pathwiseproductcaplet.hpp MarketModelPathwiseMultiDeflatedCaplet.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        payment_times: list[float],
        strikes: list[float] | float,
    ) -> None:
        self._rate_times = list(rate_times)
        self._accruals = list(accruals)
        self._payment_times = list(payment_times)
        self._number_rates = len(self._accruals)
        if isinstance(strikes, (int, float)):
            self._strikes = [float(strikes)] * self._number_rates
        else:
            self._strikes = list(strikes)
        self._current_index = 0
        check_increasing_times(rate_times)
        check_increasing_times(payment_times)
        evol_times = list(self._rate_times[:-1])
        qassert.require(
            len(evol_times) == self._number_rates, "rateTimes.size()<> numberOfRates+1"
        )
        qassert.require(
            len(payment_times) == self._number_rates,
            "paymentTimes.size()<> numberOfRates",
        )
        qassert.require(
            len(accruals) == self._number_rates, "accruals.size()<> numberOfRates"
        )
        qassert.require(
            len(self._strikes) == self._number_rates, "strikes.size()<> numberOfRates"
        )
        self._evolution = EvolutionDescription(rate_times, evol_times)

    def already_deflated(self) -> bool:
        return True

    def suggested_numeraires(self) -> list[int]:
        return list(range(self._number_rates))

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

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
        i = self._current_index
        libor_rate = current_state.forward_rate(i)
        cf = cash_flows_generated[i][0]
        cf.time_index = i
        cf.amount[0] = (
            (libor_rate - self._strikes[i])
            * self._accruals[i]
            * current_state.discount_ratio(i + 1, 0)
        )

        for k in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[k] = 0

        if cf.amount[0] > 0:
            number_cash_flows_this_step[i] = 1
            for k in range(1, self._number_rates + 1):
                cf.amount[k] = 0.0
            cf.amount[i + 1] = self._accruals[i] * current_state.discount_ratio(i + 1, 0)
            for k in range(i + 1):
                step_df = current_state.discount_ratio(k + 1, k)
                cf.amount[k + 1] -= self._accruals[k] * step_df * cf.amount[0]
        self._current_index += 1
        return self._current_index == len(self._strikes)

    def clone(self) -> MarketModelPathwiseMultiProduct:
        return MarketModelPathwiseMultiDeflatedCaplet(
            self._rate_times, self._accruals, self._payment_times, self._strikes
        )


class MarketModelPathwiseMultiDeflatedCap(MarketModelPathwiseMultiProduct):
    """Caps built from deflated caplets via ``(start, end)`` index ranges.

    # C++ parity: pathwiseproductcaplet.hpp MarketModelPathwiseMultiDeflatedCap.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        payment_times: list[float],
        strike: float,
        starts_and_ends: list[tuple[int, int]],
    ) -> None:
        self._underlying_caplets = MarketModelPathwiseMultiDeflatedCaplet(
            rate_times, accruals, payment_times, strike
        )
        self._number_rates = len(accruals)
        self._starts_and_ends = list(starts_and_ends)
        self._current_index = 0
        for j, (start, end) in enumerate(self._starts_and_ends):
            qassert.require(
                start < end, f"a cap must start before it ends: {j}{start}{end}"
            )
            qassert.require(
                end <= len(accruals),
                f"a cap must end when the underlying caplets: {j}{start}{end}",
            )
        n = self._underlying_caplets.max_number_of_cash_flows_per_product_per_step()
        self._inner_sizes = [0] * len(accruals)
        self._inner_generated: list[list[PathwiseCashFlow]] = [
            [PathwiseCashFlow(amount=[0.0] * (len(accruals) + 1)) for _ in range(n)]
            for _ in range(len(accruals))
        ]
        # retain constructor args for clone()
        self._ctor_args = (list(rate_times), list(accruals), list(payment_times), strike)

    def already_deflated(self) -> bool:
        return self._underlying_caplets.already_deflated()

    def suggested_numeraires(self) -> list[int]:
        return self._underlying_caplets.suggested_numeraires()

    def evolution(self) -> EvolutionDescription:
        return self._underlying_caplets.evolution()

    def possible_cash_flow_times(self) -> list[float]:
        return self._underlying_caplets.possible_cash_flow_times()

    def number_of_products(self) -> int:
        return len(self._starts_and_ends)

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return self._underlying_caplets.max_number_of_cash_flows_per_product_per_step()

    def reset(self) -> None:
        self._underlying_caplets.reset()
        self._current_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[PathwiseCashFlow]],
    ) -> bool:
        done = self._underlying_caplets.next_time_step(
            current_state, self._inner_sizes, self._inner_generated
        )

        for k in range(len(self._starts_and_ends)):
            number_cash_flows_this_step[k] = 0

        for j in range(self._number_rates):
            if self._inner_sizes[j] > 0:
                for k, (start, end) in enumerate(self._starts_and_ends):
                    if start <= j < end:
                        for ell in range(self._inner_sizes[j]):
                            dest = cash_flows_generated[k][number_cash_flows_this_step[k]]
                            src = self._inner_generated[j][ell]
                            dest.time_index = src.time_index
                            dest.amount = list(src.amount)
                            number_cash_flows_this_step[k] += 1
        return done

    def clone(self) -> MarketModelPathwiseMultiProduct:
        rate_times, accruals, payment_times, strike = self._ctor_args
        return MarketModelPathwiseMultiDeflatedCap(
            rate_times, accruals, payment_times, strike, self._starts_and_ends
        )
