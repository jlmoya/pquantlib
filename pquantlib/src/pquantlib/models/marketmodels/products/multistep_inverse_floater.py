"""MultiStepInverseFloater — an inverse floater as a multi-step product.

# C++ parity:
# ql/models/marketmodels/products/multistep/multistepinversefloater.{hpp,cpp}
# (v1.42.1).

A single product. At step ``i`` it pays the net of an inverse-floating coupon
``max(strike - mult * libor, 0) * fixedAccrual`` minus a floating coupon
``(libor + spread) * floatingAccrual``, scaled by ``multiplier`` (``-1`` payer,
``+1`` receiver), at payment-time index ``i``.
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


class MultiStepInverseFloater(MultiProductMultiStep):
    """An inverse floater.

    # C++ parity: multistepinversefloater.hpp MultiStepInverseFloater.
    """

    def __init__(
        self,
        rate_times: list[float],
        fixed_accruals: list[float],
        floating_accruals: list[float],
        fixed_strikes: list[float],
        fixed_multipliers: list[float],
        floating_spreads: list[float],
        payment_times: list[float],
        payer: bool = True,
    ) -> None:
        super().__init__(rate_times)
        self._fixed_accruals = list(fixed_accruals)
        self._floating_accruals = list(floating_accruals)
        self._fixed_strikes = list(fixed_strikes)
        self._fixed_multipliers = list(fixed_multipliers)
        self._floating_spreads = list(floating_spreads)
        self._payment_times = list(payment_times)
        self._multiplier = -1.0 if payer else 1.0
        self._last_index = len(rate_times) - 1
        self._current_index = 0
        check_increasing_times(payment_times)
        n = self._last_index
        qassert.require(
            len(self._fixed_accruals) == n,
            f" Incorrect number of fixedAccruals given, should be {n} "
            f"not {len(self._fixed_accruals)}",
        )
        qassert.require(
            len(self._floating_accruals) == n,
            f" Incorrect number of floatingAccruals given, should be {n} "
            f"not {len(self._floating_accruals)}",
        )
        qassert.require(
            len(self._fixed_strikes) == n,
            f" Incorrect number of fixedStrikes given, should be {n} "
            f"not {len(self._fixed_strikes)}",
        )
        qassert.require(
            len(self._fixed_multipliers) == n,
            f" Incorrect number of fixedMultipliers given, should be {n} "
            f"not {len(self._fixed_multipliers)}",
        )
        qassert.require(
            len(self._floating_spreads) == n,
            f" Incorrect number of floatingSpreads given, should be {n} "
            f"not {len(self._floating_spreads)}",
        )
        qassert.require(
            len(self._payment_times) == n,
            f" Incorrect number of paymentTimes given, should be {n} "
            f"not {len(self._payment_times)}",
        )

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return 1

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._current_index = 0

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        i = self._current_index
        libor_rate = current_state.forward_rate(i)
        inverse_floating_coupon = (
            max(self._fixed_strikes[i] - self._fixed_multipliers[i] * libor_rate, 0.0)
            * self._fixed_accruals[i]
        )
        floating_coupon = (libor_rate + self._floating_spreads[i]) * self._floating_accruals[i]

        cf = cash_flows_generated[0][0]
        cf.time_index = i
        cf.amount = self._multiplier * (inverse_floating_coupon - floating_coupon)

        number_cash_flows_this_step[0] = 1
        self._current_index += 1
        return self._current_index == self._last_index

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepInverseFloater(
            self._rate_times,
            self._fixed_accruals,
            self._floating_accruals,
            self._fixed_strikes,
            self._fixed_multipliers,
            self._floating_spreads,
            self._payment_times,
            self._multiplier < 0.0,
        )
