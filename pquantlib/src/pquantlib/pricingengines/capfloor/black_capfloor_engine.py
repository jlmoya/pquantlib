"""Black-formula cap/floor engine.

# C++ parity: ql/pricingengines/capfloor/blackcapfloorengine.{hpp,cpp}
# (v1.42.1).

For each optionlet (forward-rate fixing) in the cap/floor leg, applies
the Black-76 formula (or Bachelier for the Normal counterpart) with
``forward = adjusted_fixing`` and ``stdDev = vol * sqrt(fixing_time)``.
Per-optionlet contributions are weighted by ``nominal * gearing *
accrual_time * discount(end_date)``.

Past fixings (fixing_date <= today) contribute their intrinsic value
``max(forward - strike, 0)`` * discount * accrual_factor (matches C++).

The Bachelier counterpart lives in ``bachelier_capfloor_engine.py``.
Both classes share the same loop structure; the Spec (lognormal vs
normal) is selected via a ``VolatilityType`` argument on
``BlackStyleCapFloorEngine``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.cap_floor import (
    CapFloorArguments,
    CapFloorResults,
    CapFloorType,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    bachelier_black_formula_std_dev_derivative,
    black_formula,
    black_formula_std_dev_derivative,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class BlackStyleCapFloorEngine(GenericEngine[CapFloorArguments, CapFloorResults]):
    """Per-optionlet Black-or-Bachelier engine for cap/floor.

    # C++ parity: the C++ codebase keeps two parallel classes
    # ``BlackCapFloorEngine`` + ``BachelierCapFloorEngine`` with
    # near-identical ``calculate`` loops. PQuantLib factors the loop
    # into a single class parameterised by ``VolatilityType``.
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        day_counter: DayCounter | None = None,
        displacement: float = 0.0,
    ) -> None:
        super().__init__(CapFloorArguments(), CapFloorResults())
        self._discount_curve: YieldTermStructureProtocol = discount_curve
        self._vol: float | Quote = vol
        self._volatility_type: VolatilityType = volatility_type
        self._day_counter: DayCounter = (
            day_counter if day_counter is not None else Actual365Fixed()
        )
        self._displacement: float = displacement

    # --- helpers ------------------------------------------------------

    def _vol_value(self) -> float:
        if isinstance(self._vol, Quote):
            return self._vol.value()
        return float(self._vol)

    @property
    def displacement(self) -> float:
        return self._displacement

    @property
    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    # --- engine -------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 (faithful port of C++ per-optionlet loop)
        # # C++ parity: blackcapfloorengine.cpp:77-166 (v1.42.1).
        args = self._arguments
        results = self._results
        results.reset()

        sigma = self._vol_value()
        # Effective displacement: Normal forces 0; ShiftedLognormal uses
        # the configured shift.
        eff_disp = (
            self._displacement
            if self._volatility_type == VolatilityType.ShiftedLognormal
            else 0.0
        )

        cap_type = args.type
        n = len(args.start_dates)
        ref_date = self._discount_curve.reference_date()

        total_value = 0.0
        total_vega = 0.0
        values: list[float] = [0.0] * n
        vegas: list[float] = [0.0] * n
        deltas: list[float] = [0.0] * n
        std_devs: list[float] = [0.0] * n
        discount_factors: list[float] = [0.0] * n

        for i in range(n):
            payment_date = args.end_dates[i]
            if payment_date <= ref_date:
                # Expired — skip (matches C++ comment "discard expired
                # caplets" — they contribute zero on a fully past leg).
                continue

            df = self._discount_curve.discount(payment_date)
            discount_factors[i] = df
            accrual_factor = args.nominals[i] * args.gearings[i] * args.accrual_times[i]
            discounted_accrual = df * accrual_factor
            forward = args.forwards[i]

            fixing_date = args.fixing_dates[i]
            sqrt_time = 0.0
            t_fixing = self._day_counter.year_fraction(ref_date, fixing_date)
            if fixing_date > ref_date and t_fixing > 0.0:
                sqrt_time = math.sqrt(t_fixing)

            # Cap leg.
            if cap_type in (CapFloorType.Cap, CapFloorType.Collar):
                strike = args.cap_rates[i]
                if sqrt_time > 0.0:
                    std_devs[i] = sigma * sqrt_time
                    if self._volatility_type == VolatilityType.ShiftedLognormal:
                        vegas[i] = (
                            black_formula_std_dev_derivative(
                                strike, forward, std_devs[i], discounted_accrual, eff_disp
                            )
                            * sqrt_time
                        )
                    else:
                        vegas[i] = (
                            bachelier_black_formula_std_dev_derivative(
                                strike, forward, std_devs[i], discounted_accrual
                            )
                            * sqrt_time
                        )
                # Past + live fixings both via the Black formula (std_dev=0
                # → intrinsic).
                if self._volatility_type == VolatilityType.ShiftedLognormal:
                    values[i] = black_formula(
                        OptionType.Call,
                        strike,
                        forward,
                        std_devs[i],
                        discounted_accrual,
                        eff_disp,
                    )
                else:
                    values[i] = bachelier_black_formula(
                        OptionType.Call,
                        strike,
                        forward,
                        std_devs[i],
                        discounted_accrual,
                    )

            # Floor leg.
            if cap_type in (CapFloorType.Floor, CapFloorType.Collar):
                strike = args.floor_rates[i]
                floorlet_vega = 0.0
                if sqrt_time > 0.0:
                    std_devs[i] = sigma * sqrt_time
                    if self._volatility_type == VolatilityType.ShiftedLognormal:
                        floorlet_vega = (
                            black_formula_std_dev_derivative(
                                strike, forward, std_devs[i], discounted_accrual, eff_disp
                            )
                            * sqrt_time
                        )
                    else:
                        floorlet_vega = (
                            bachelier_black_formula_std_dev_derivative(
                                strike, forward, std_devs[i], discounted_accrual
                            )
                            * sqrt_time
                        )
                if self._volatility_type == VolatilityType.ShiftedLognormal:
                    floorlet = black_formula(
                        OptionType.Put,
                        strike,
                        forward,
                        std_devs[i],
                        discounted_accrual,
                        eff_disp,
                    )
                else:
                    floorlet = bachelier_black_formula(
                        OptionType.Put,
                        strike,
                        forward,
                        std_devs[i],
                        discounted_accrual,
                    )
                if cap_type == CapFloorType.Floor:
                    values[i] = floorlet
                    vegas[i] = floorlet_vega
                else:
                    # Collar: long cap, short floor → subtract floorlet.
                    values[i] -= floorlet
                    vegas[i] -= floorlet_vega

            total_value += values[i]
            total_vega += vegas[i]

        results.value = total_value
        results.additional_results["vega"] = total_vega
        results.additional_results["optionletsPrice"] = values
        results.additional_results["optionletsVega"] = vegas
        results.additional_results["optionletsDelta"] = deltas
        results.additional_results["optionletsDiscountFactor"] = discount_factors
        results.additional_results["optionletsAtmForward"] = list(args.forwards)
        if cap_type != CapFloorType.Collar:
            results.additional_results["optionletsStdDev"] = std_devs


class BlackCapFloorEngine(BlackStyleCapFloorEngine):
    """Shifted-lognormal Black-76 cap/floor engine.

    # C++ parity: ``class BlackCapFloorEngine`` in
    # blackcapfloorengine.hpp:38-60 (v1.42.1).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote,
        day_counter: DayCounter | None = None,
        displacement: float = 0.0,
    ) -> None:
        super().__init__(
            discount_curve,
            vol,
            volatility_type=VolatilityType.ShiftedLognormal,
            day_counter=day_counter,
            displacement=displacement,
        )


class BachelierCapFloorEngine(BlackStyleCapFloorEngine):
    """Normal (Bachelier) cap/floor engine.

    # C++ parity: ``class BachelierCapFloorEngine`` in
    # bacheliercapfloorengine.hpp (v1.42.1).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(
            discount_curve,
            vol,
            volatility_type=VolatilityType.Normal,
            day_counter=day_counter,
            displacement=0.0,
        )


__all__ = [
    "BachelierCapFloorEngine",
    "BlackCapFloorEngine",
    "BlackStyleCapFloorEngine",
]
