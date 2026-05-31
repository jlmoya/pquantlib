"""MultiStepRatchet — a ratchet floater as a multi-step product.

# C++ parity:
# ql/models/marketmodels/products/multistep/multistepratchet.{hpp,cpp}
# (v1.42.1).

A single product. At step ``i`` the coupon is the greater of a geared floor
(``gearingOfFloor * floor + spreadOfFloor``) and a geared fixing
(``gearingOfFixing * libor + spreadOfFixing``). The coupon ``* accrual *
multiplier`` is paid at payment-time index ``i``, and (full-ratchet) the floor
is reset to the just-paid coupon for the next step.
"""

from __future__ import annotations

from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.multiproduct_multistep import (
    MultiProductMultiStep,
)
from pquantlib.models.marketmodels.utilities import check_increasing_times


class MultiStepRatchet(MultiProductMultiStep):
    """A ratchet floater (full-ratchet coupon accumulation).

    # C++ parity: multistepratchet.hpp MultiStepRatchet.
    """

    def __init__(
        self,
        rate_times: list[float],
        accruals: list[float],
        payment_times: list[float],
        gearing_of_floor: float,
        gearing_of_fixing: float,
        spread_of_floor: float,
        spread_of_fixing: float,
        initial_floor: float,
        payer: bool = True,
    ) -> None:
        super().__init__(rate_times)
        self._accruals = list(accruals)
        self._payment_times = list(payment_times)
        self._gearing_of_floor = gearing_of_floor
        self._gearing_of_fixing = gearing_of_fixing
        self._spread_of_floor = spread_of_floor
        self._spread_of_fixing = spread_of_fixing
        self._multiplier = 1.0 if payer else -1.0
        self._last_index = len(rate_times) - 1
        self._initial_floor = initial_floor
        self._floor = initial_floor
        self._current_index = 0
        check_increasing_times(payment_times)

    def possible_cash_flow_times(self) -> list[float]:
        return self._payment_times

    def number_of_products(self) -> int:
        return 1

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._current_index = 0
        self._floor = self._initial_floor

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        libor_rate = current_state.forward_rate(self._current_index)
        current_coupon = max(
            self._gearing_of_floor * self._floor + self._spread_of_floor,
            self._gearing_of_fixing * libor_rate + self._spread_of_fixing,
        )

        cf = cash_flows_generated[0][0]
        cf.time_index = self._current_index
        cf.amount = (
            self._multiplier * self._accruals[self._current_index] * current_coupon
        )

        # full-ratchet: the floor for the next step is the coupon just paid.
        self._floor = current_coupon
        number_cash_flows_this_step[0] = 1

        self._current_index += 1
        return self._current_index == self._last_index

    def clone(self) -> MarketModelMultiProduct:
        return MultiStepRatchet(
            self._rate_times,
            self._accruals,
            self._payment_times,
            self._gearing_of_floor,
            self._gearing_of_fixing,
            self._spread_of_floor,
            self._spread_of_fixing,
            self._initial_floor,
            self._multiplier > 0.0,
        )
