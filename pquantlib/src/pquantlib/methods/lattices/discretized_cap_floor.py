"""DiscretizedCapFloor — interest-rate cap/floor on a lattice.

# C++ parity: ql/pricingengines/capfloor/discretizedcapfloor.{hpp,cpp} (v1.42.1).

Discretizes a CapFloor instrument on a tree/FD lattice. Each optionlet
fires at its ``start_time``: the value at that node is augmented by
the cap/floor payout discounted via a ``DiscretizedDiscountBond``
rolled back from the optionlet's pay (= end) time.

Past fixings (``start_time < 0``) are handled in the
``postAdjustValuesImpl`` at the pay time using the cached
``forwards[i]`` rate from the argument carrier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib.instruments.cap_floor import CapFloorType
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.methods.lattices.discretized_discount_bond import (
    DiscretizedDiscountBond,
)

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.instruments.cap_floor import CapFloorArguments
    from pquantlib.time.date import Date


class DiscretizedCapFloor(DiscretizedAsset):
    """Discretized cap / floor / collar on the lattice.

    # C++ parity: ``class DiscretizedCapFloor``
    # (discretizedcapfloor.{hpp,cpp}, v1.42.1).
    """

    def __init__(
        self,
        args: CapFloorArguments,
        reference_date: Date,
        day_counter: DayCounter,
    ) -> None:
        # # C++ parity: discretizedcapfloor.cpp:25-39.
        super().__init__()
        self._arguments: CapFloorArguments = args
        self._start_times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in args.start_dates
        ]
        self._end_times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in args.end_dates
        ]

    # --- DiscretizedAsset interface --------------------------------------

    def reset(self, size: int) -> None:
        """Zero-init the values array.

        # C++ parity: ``DiscretizedCapFloor::reset`` (discretizedcapfloor.cpp:41-44).
        """
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """Union of start and end times.

        # C++ parity: ``DiscretizedCapFloor::mandatoryTimes`` (discretizedcapfloor.cpp:46-51).
        """
        return [*self._start_times, *self._end_times]

    # --- adjustments ------------------------------------------------------

    def _pre_adjust_values_impl(self) -> None:
        """For each optionlet whose start is on this time, add its value.

        # C++ parity: ``preAdjustValuesImpl`` (discretizedcapfloor.cpp:53-86).
        """
        method = self._require_method()
        for i, start_t in enumerate(self._start_times):
            if not self.is_on_time(start_t):
                continue
            end_t = self._end_times[i]
            tenor = self._arguments.accrual_times[i]

            # # C++ parity: discretizedcapfloor.cpp:58-65 — build the
            # # zero-coupon bond maturing at the optionlet end and
            # # roll it back to ``self.time``.
            bond = DiscretizedDiscountBond()
            bond.initialize(method, end_t)
            bond.rollback(self._time)

            cap_type = self._arguments.type
            gearing = self._arguments.gearings[i]
            nominal = self._arguments.nominals[i]

            if cap_type in (CapFloorType.Cap, CapFloorType.Collar):
                accrual = 1.0 + self._arguments.cap_rates[i] * tenor
                strike = 1.0 / accrual
                # Cap payoff per optionlet: nominal * accrual * gearing *
                # max(strike - bond, 0).
                self._values = (
                    self._values
                    + nominal
                    * accrual
                    * gearing
                    * np.maximum(strike - bond.values, 0.0)
                )

            if cap_type in (CapFloorType.Floor, CapFloorType.Collar):
                accrual = 1.0 + self._arguments.floor_rates[i] * tenor
                strike = 1.0 / accrual
                mult = 1.0 if cap_type == CapFloorType.Floor else -1.0
                self._values = (
                    self._values
                    + nominal
                    * accrual
                    * mult
                    * gearing
                    * np.maximum(bond.values - strike, 0.0)
                )

    def _post_adjust_values_impl(self) -> None:
        """Past-fixing settlement: add intrinsic payout at end time.

        # C++ parity: ``postAdjustValuesImpl`` (discretizedcapfloor.cpp:88-115).
        """
        for i, end_t in enumerate(self._end_times):
            if not self.is_on_time(end_t):
                continue
            if self._start_times[i] >= 0.0:
                continue  # not a past fixing
            nominal = self._arguments.nominals[i]
            accrual_time = self._arguments.accrual_times[i]
            fixing = self._arguments.forwards[i]
            gearing = self._arguments.gearings[i]
            cap_type = self._arguments.type

            if cap_type in (CapFloorType.Cap, CapFloorType.Collar):
                cap_rate = self._arguments.cap_rates[i]
                caplet = max(fixing - cap_rate, 0.0)
                self._values = self._values + caplet * accrual_time * nominal * gearing

            if cap_type in (CapFloorType.Floor, CapFloorType.Collar):
                floor_rate = self._arguments.floor_rates[i]
                floorlet = max(floor_rate - fixing, 0.0)
                if cap_type == CapFloorType.Floor:
                    self._values = (
                        self._values + floorlet * accrual_time * nominal * gearing
                    )
                else:
                    self._values = (
                        self._values - floorlet * accrual_time * nominal * gearing
                    )


__all__ = ["DiscretizedCapFloor"]
