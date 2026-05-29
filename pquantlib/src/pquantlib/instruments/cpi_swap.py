"""CPISwap — zero-inflation-indexed-ratio-with-base vs floating+spread swap.

# C++ parity: ql/instruments/cpiswap.{hpp,cpp} (v1.42.1).

A CPISwap has two legs:

* **CPI leg** — fixed-rate * CPI(i)/CPI(base) (paid in arrears via L7-C's
  CPILeg builder; L7-D ports the swap shell only, taking a pre-built leg).
* **Float leg** — floating + spread (an IborLeg).

Both legs may have notional exchanges at the end depending on
``subtract_inflation_nominal``. ``Type`` (Payer/Receiver) refers to the
*floating* leg.

L7-C will provide the CPILeg builder. Until then, callers pass in a
pre-built CPI leg of CashFlows that satisfy the leg-level NPV contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.instruments.swap import Swap, SwapType
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule


class CPISwap(Swap):
    """Zero-inflation-indexed swap (fixed*CPI/base) vs floating+spread.

    # C++ parity: ``CPISwap`` (cpiswap.{hpp,cpp}).

    L7-D port: the constructor takes pre-built CPI + float legs as
    Sequences of CashFlows. The C++ class builds them inline using
    CPILeg + IborLeg; L7-C will land that ergonomic API in a follow-up,
    matching the C++ signature.
    """

    def __init__(
        self,
        *,
        type_: SwapType,
        nominal: float,
        subtract_inflation_nominal: bool,
        # Float leg parameters (carried for inspectors)
        spread: float,
        float_day_count: DayCounter,
        float_schedule: Schedule,
        float_payment_roll: BusinessDayConvention,
        fixing_days: int,
        float_index: IborIndexProtocol,
        # Fixed leg parameters
        fixed_rate: float,
        base_cpi: float,
        fixed_day_count: DayCounter,
        fixed_schedule: Schedule,
        fixed_payment_roll: BusinessDayConvention,
        observation_lag: Period,
        fixed_index: ZeroInflationIndex,
        observation_interpolation: InterpolationType = InterpolationType.AsIndex,
        inflation_nominal: float | None = None,
        # Pre-built legs (until L7-C lands CPILeg / IborLeg builders for inflation)
        cpi_leg: Sequence[CashFlow] | None = None,
        float_leg: Sequence[CashFlow] | None = None,
    ) -> None:
        # # C++ parity: CPISwap::CPISwap (cpiswap.cpp:37-130).
        super().__init__(n_legs=2)
        self._type: SwapType = type_
        self._nominal: float = nominal
        self._subtract_inflation_nominal: bool = subtract_inflation_nominal
        self._spread: float = spread
        self._float_day_count: DayCounter = float_day_count
        self._float_schedule: Schedule = float_schedule
        self._float_payment_roll: BusinessDayConvention = float_payment_roll
        self._fixing_days: int = fixing_days
        self._float_index: IborIndexProtocol = float_index
        self._fixed_rate: float = fixed_rate
        self._base_cpi: float = base_cpi
        self._fixed_day_count: DayCounter = fixed_day_count
        self._fixed_schedule: Schedule = fixed_schedule
        self._fixed_payment_roll: BusinessDayConvention = fixed_payment_roll
        self._fixed_index: ZeroInflationIndex = fixed_index
        self._observation_lag: Period = observation_lag
        self._observation_interpolation: InterpolationType = observation_interpolation
        self._inflation_nominal: float = (
            inflation_nominal if inflation_nominal is not None else nominal
        )

        qassert.require(len(fixed_schedule) > 0, "empty fixed schedule")
        qassert.require(len(float_schedule) > 0, "empty float schedule")

        # L7-C will replace the leg-builder seam with CPILeg + IborLeg
        # constructions matching cpiswap.cpp:72-108. Until then we accept
        # pre-built legs.
        if cpi_leg is None or float_leg is None:
            qassert.fail(
                "L7-D CPISwap requires pre-built cpi_leg + float_leg; the L7-C "
                "CPILeg / IborLeg builders for inflation are not yet wired."
            )
        self._legs = [list(cpi_leg), list(float_leg)]
        # # C++ parity: payer multipliers (cpiswap.cpp:123-129). Note the
        # # inversion vs ZCIIS — here Payer means *floating leg* is paid.
        if type_ == SwapType.Payer:
            self._payer = [+1.0, -1.0]
        else:
            self._payer = [-1.0, +1.0]

        for cf in self._legs[0]:
            cf.register_with(self)
        for cf in self._legs[1]:
            cf.register_with(self)

    # ---- inspectors --------------------------------------------------

    def type(self) -> SwapType:
        return self._type

    def nominal(self) -> float:
        return self._nominal

    def subtract_inflation_nominal(self) -> bool:
        return self._subtract_inflation_nominal

    # Float-side
    def spread(self) -> float:
        return self._spread

    def float_day_count(self) -> DayCounter:
        return self._float_day_count

    def float_schedule(self) -> Schedule:
        return self._float_schedule

    def float_payment_roll(self) -> BusinessDayConvention:
        return self._float_payment_roll

    def fixing_days(self) -> int:
        return self._fixing_days

    def float_index(self) -> IborIndexProtocol:
        return self._float_index

    # Fixed-side
    def fixed_rate(self) -> float:
        return self._fixed_rate

    def base_cpi(self) -> float:
        return self._base_cpi

    def fixed_day_count(self) -> DayCounter:
        return self._fixed_day_count

    def fixed_schedule(self) -> Schedule:
        return self._fixed_schedule

    def fixed_payment_roll(self) -> BusinessDayConvention:
        return self._fixed_payment_roll

    def observation_lag(self) -> Period:
        return self._observation_lag

    def fixed_index(self) -> ZeroInflationIndex:
        return self._fixed_index

    def observation_interpolation(self) -> InterpolationType:
        return self._observation_interpolation

    def inflation_nominal(self) -> float:
        return self._inflation_nominal

    # legs
    def cpi_leg(self) -> list[CashFlow]:
        return self._legs[0]

    def float_leg(self) -> list[CashFlow]:
        return self._legs[1]

    # ---- results -----------------------------------------------------

    def fixed_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._leg_npv[0] is not None, "result not available")
        assert self._leg_npv[0] is not None
        return self._leg_npv[0]

    def float_leg_npv(self) -> float:
        self.calculate()
        qassert.require(self._leg_npv[1] is not None, "result not available")
        assert self._leg_npv[1] is not None
        return self._leg_npv[1]

    def fair_rate(self) -> float:
        """Fair fixed rate. # C++ parity: CPISwap::fairRate (cpiswap.cpp:145-149)."""
        self.calculate()
        leg_bps = self._leg_bps[0]
        qassert.require(
            leg_bps is not None and not math.isclose(leg_bps, 0.0, abs_tol=1e-30),
            "fair rate not available",
        )
        assert leg_bps is not None
        return self._fixed_rate - self.npv() / (leg_bps / 1e-4)

    def fair_spread(self) -> float:
        """Fair floating spread. # C++ parity: CPISwap::fairSpread (cpiswap.cpp:151-155)."""
        self.calculate()
        leg_bps = self._leg_bps[1]
        qassert.require(
            leg_bps is not None and not math.isclose(leg_bps, 0.0, abs_tol=1e-30),
            "fair spread not available",
        )
        assert leg_bps is not None
        return self._spread - self.npv() / (leg_bps / 1e-4)


__all__ = ["CPISwap"]
