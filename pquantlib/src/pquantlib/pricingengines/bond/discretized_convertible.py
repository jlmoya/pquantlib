"""DiscretizedConvertible — lattice-discretized convertible bond.

# C++ parity: ql/pricingengines/bond/discretizedconvertible.{hpp,cpp}
#             (v1.42.1).

Reflects a :class:`ConvertibleBondArguments` onto a binomial lattice so the
:class:`~pquantlib.methods.lattices.tsiveriotis_fernandes_lattice.TsiveriotisFernandesLattice`
can roll back the embedded conversion + call/put optionality with the
credit-adjusted (Tsiveriotis-Fernandes) discounting.

Algorithm (matches the C++ source step-for-step):

* ``reset`` seeds every node with the redemption, runs ``adjust_values``
  (which applies the terminal convertibility + callability), then sets the
  per-node *blended* (spread-adjusted) discount rate from the conversion
  probability.
* ``post_adjust_values`` applies, in order: callability (caps for calls,
  floors for puts, conditioned by triggers), coupons, then convertibility.
* ``adjusted_grid`` adds back the present value of all future dividends to
  the lattice's underlying grid (so the conversion payoff sees the
  dividend-inclusive stock price).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.instruments.callability import CallabilityType
from pquantlib.math.array import Array
from pquantlib.math.closeness import close
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.cashflows.dividend import Dividend
    from pquantlib.instruments.bonds.convertible_bonds import (
        ConvertibleBondArguments,
    )
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib.quotes.quote import Quote
    from pquantlib.time.date import Date
    from pquantlib.time.time_grid import TimeGrid


class DiscretizedConvertible(DiscretizedAsset):
    """Lattice-discretized Tsiveriotis-Fernandes convertible bond.

    # C++ parity: ``class DiscretizedConvertible`` (discretizedconvertible.hpp).
    """

    def __init__(
        self,
        args: ConvertibleBondArguments,
        process: GeneralizedBlackScholesProcess,
        dividends: list[Dividend],
        credit_spread: Quote,
        grid: TimeGrid | None = None,
    ) -> None:
        super().__init__()
        self._arguments: ConvertibleBondArguments = args
        self._process: GeneralizedBlackScholesProcess = process
        self._credit_spread: Quote = credit_spread

        # C++ parity: discretizedconvertible.cpp:37-54.
        # Filter dividends not yet occurred at settlement; build dividend
        # value array (discounted dividend amounts) — note the C++ uses the
        # *unfiltered* ``dividends`` for dividendValues_, sized to the
        # *filtered* list. We replicate that exactly.
        self._dividends: list[Dividend] = []
        self._dividend_dates: list[Date] = []
        for d in dividends:
            if not d.has_occurred(args.settlement_date, False):
                self._dividends.append(d)
                self._dividend_dates.append(d.date())

        self._dividend_values: Array = np.zeros(len(self._dividends), dtype=np.float64)

        rf = process.risk_free_rate()
        ref_settlement = rf.reference_date()
        for i, d in enumerate(dividends):
            if i >= len(self._dividend_values):
                break
            if d.date() >= ref_settlement:
                self._dividend_values[i] = d.amount() * rf.discount(d.date())

        day_counter = rf.day_counter()
        bond_settlement = args.settlement_date

        exercise = args.exercise
        assert exercise is not None
        self._stopping_times: list[float] = [
            day_counter.year_fraction(bond_settlement, exercise.date(i))
            for i in range(len(exercise.dates()))
        ]

        self._callability_times: list[float] = [
            day_counter.year_fraction(bond_settlement, d)
            for d in args.callability_dates
        ]

        # Coupons: every cashflow except the last (the redemption).
        self._coupon_times: list[float] = []
        self._coupon_amounts: list[float] = []
        for cf in args.cashflows[:-1]:
            if not cf.has_occurred(bond_settlement, False):
                self._coupon_times.append(
                    day_counter.year_fraction(bond_settlement, cf.date())
                )
                self._coupon_amounts.append(cf.amount())

        self._dividend_times: list[float] = [
            day_counter.year_fraction(bond_settlement, d)
            for d in self._dividend_dates
        ]

        if grid is not None and not grid.empty():
            self._stopping_times = [grid.closest_time(t) for t in self._stopping_times]
            self._coupon_times = [grid.closest_time(t) for t in self._coupon_times]
            self._callability_times = [
                grid.closest_time(t) for t in self._callability_times
            ]
            self._dividend_times = [grid.closest_time(t) for t in self._dividend_times]

        self._conversion_probability: Array = np.empty(0, dtype=np.float64)
        self._spread_adjusted_rate: Array = np.empty(0, dtype=np.float64)

    # -- TF-lattice accessors ---------------------------------------------

    @property
    def conversion_probability(self) -> Array:
        return self._conversion_probability

    @conversion_probability.setter
    def conversion_probability(self, value: Array) -> None:
        self._conversion_probability = value

    @property
    def spread_adjusted_rate(self) -> Array:
        return self._spread_adjusted_rate

    @spread_adjusted_rate.setter
    def spread_adjusted_rate(self, value: Array) -> None:
        self._spread_adjusted_rate = value

    @property
    def dividend_values(self) -> Array:
        return self._dividend_values

    # -- DiscretizedAsset interface ---------------------------------------

    def mandatory_times(self) -> list[float]:
        # C++ parity: discretizedconvertible.hpp:55-64.
        return [
            *self._stopping_times,
            *self._callability_times,
            *self._coupon_times,
        ]

    def reset(self, size: int) -> None:
        # C++ parity: discretizedconvertible.cpp:100-130.
        assert self._arguments.redemption is not None
        self._values = np.full(size, self._arguments.redemption, dtype=np.float64)
        self._conversion_probability = np.zeros(size, dtype=np.float64)
        self._spread_adjusted_rate = np.zeros(size, dtype=np.float64)

        rf = self._process.risk_free_rate()
        rfdc = rf.day_counter()

        # this takes care of convertibility and conversion probabilities
        self.adjust_values()

        credit_spread = self._credit_spread.value()

        exercise = self._arguments.exercise
        assert exercise is not None
        risk_free_rate = rf.zero_rate(
            exercise.last_date(),
            Compounding.Continuous,
            Frequency.NoFrequency,
            result_day_counter=rfdc,
        ).rate()

        # Blended discount rate used on rollback.
        self._spread_adjusted_rate = (
            self._conversion_probability * risk_free_rate
            + (1.0 - self._conversion_probability) * (risk_free_rate + credit_spread)
        )

    def _post_adjust_values_impl(self) -> None:
        # C++ parity: discretizedconvertible.cpp:132-166.
        convertible = False
        exercise = self._arguments.exercise
        assert exercise is not None
        ex_type = exercise.type()
        if ex_type == exercise.Type.American:
            if self._stopping_times[0] <= self.time <= self._stopping_times[1]:
                convertible = True
        elif ex_type == exercise.Type.European:
            if self.is_on_time(self._stopping_times[0]):
                convertible = True
        elif ex_type == exercise.Type.Bermudan:
            for st in self._stopping_times:
                if self.is_on_time(st):
                    convertible = True
        else:  # pragma: no cover - exhaustive
            qassert.fail("invalid option type")

        for i, t in enumerate(self._callability_times):
            if self.is_on_time(t):
                self._apply_callability(i, convertible)

        for i, t in enumerate(self._coupon_times):
            if self.is_on_time(t):
                self._add_coupon(i)

        if convertible:
            self._apply_convertibility()

    # -- helpers ----------------------------------------------------------

    def _apply_convertibility(self) -> None:
        # C++ parity: discretizedconvertible.cpp:168-177.
        grid = self._adjusted_grid()
        assert self._arguments.conversion_ratio is not None
        ratio = self._arguments.conversion_ratio
        payoff = ratio * grid
        converted = self._values <= payoff
        self._values = np.where(converted, payoff, self._values)
        self._conversion_probability = np.where(
            converted, 1.0, self._conversion_probability
        )

    def _apply_callability(self, i: int, convertible: bool) -> None:
        # C++ parity: discretizedconvertible.cpp:179-224.
        grid = self._adjusted_grid()
        assert self._arguments.conversion_ratio is not None
        assert self._arguments.redemption is not None
        ratio = self._arguments.conversion_ratio
        c_type = self._arguments.callability_types[i]
        price = self._arguments.callability_prices[i]

        if c_type == CallabilityType.Call:
            trigger = self._arguments.callability_triggers[i]
            if trigger is not None:
                conversion_value = self._arguments.redemption / ratio
                trigger_level = conversion_value * trigger
                # conditioned by the trigger; ...and might trigger conversion
                callable_value = np.minimum(
                    np.maximum(price, ratio * grid), self._values
                )
                self._values = np.where(
                    grid >= trigger_level, callable_value, self._values
                )
            elif convertible:
                # exercising the callability might trigger conversion
                self._values = np.minimum(
                    np.maximum(price, ratio * grid), self._values
                )
            else:
                self._values = np.minimum(price, self._values)
        elif c_type == CallabilityType.Put:
            self._values = np.maximum(self._values, price)
        else:  # pragma: no cover - exhaustive
            qassert.fail("unknown callability type")

    def _add_coupon(self, i: int) -> None:
        # C++ parity: discretizedconvertible.cpp:226-228.
        self._values = self._values + self._coupon_amounts[i]

    def _adjusted_grid(self) -> Array:
        # C++ parity: discretizedconvertible.cpp:230-246.
        t = self.time
        method = self._require_method()
        grid = method.grid(t)
        rf = self._process.risk_free_rate()
        for i, dividend in enumerate(self._dividends):
            dividend_time = self._dividend_times[i]
            if dividend_time >= t or close(dividend_time, t):
                dividend_discount = rf.discount(dividend_time) / rf.discount(t)
                grid = grid + np.array(
                    [
                        dividend.amount_with_underlying(float(g)) * dividend_discount
                        for g in grid
                    ],
                    dtype=np.float64,
                )
        return grid


__all__ = ["DiscretizedConvertible"]
