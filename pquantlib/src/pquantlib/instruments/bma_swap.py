"""BMASwap — swap paying a fraction of Libor plus a spread against BMA coupons.

# C++ parity: ql/instruments/bmaswap.{hpp,cpp} (v1.43).

Leg 0 is the Libor leg (an ``IborLeg`` geared by ``libor_fraction`` and shifted
by ``libor_spread``); leg 1 is the BMA leg (an ``AverageBMALeg``).  ``Payer`` /
``Receiver`` refers to the **BMA** leg, so ``Payer`` pays BMA and receives
Libor.

Each leg picks up its own schedule's business-day convention as the payment
adjustment (``bmaswap.cpp:43-59``), which is why the two schedules can — and in
practice do — carry different conventions and calendars.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.average_bma_coupon import average_bma_leg
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.bma_index import BMAIndex
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.swap import Swap, SwapType
from pquantlib.time.schedule import Schedule

_BASIS_POINT = 1.0e-4


class BMASwap(Swap):
    """Swap paying Libor (geared + spread) against BMA coupons.

    # C++ parity: ``BMASwap`` (bmaswap.hpp:36-80, bmaswap.cpp:27-80).
    """

    def __init__(
        self,
        type_: SwapType,
        nominal: float,
        # Libor leg
        libor_schedule: Schedule,
        libor_fraction: float,
        libor_spread: float,
        libor_index: IborIndex,
        libor_day_count: DayCounter,
        # BMA leg
        bma_schedule: Schedule,
        bma_index: BMAIndex,
        bma_day_count: DayCounter,
    ) -> None:
        # # C++ parity: ``BMASwap::BMASwap`` (bmaswap.cpp:27-80).
        super().__init__(n_legs=2)
        self._type: SwapType = type_
        self._nominal: float = nominal
        self._libor_fraction: float = libor_fraction
        self._libor_spread: float = libor_spread

        convention = libor_schedule.business_day_convention
        self._legs[0] = list(
            ibor_leg(
                libor_schedule,
                libor_index,
                [nominal],
                payment_day_counter=libor_day_count,
                payment_adjustment=convention,
                fixing_days=libor_index.fixing_days(),
                gearings=libor_fraction,
                spreads=libor_spread,
            )
        )

        bma_convention = bma_schedule.business_day_convention
        self._legs[1] = list(
            average_bma_leg(
                bma_schedule,
                bma_index,
                [nominal],
                payment_day_counter=bma_day_count,
                payment_adjustment=bma_convention,
            )
        )

        for leg in self._legs:
            for cf in leg:
                cf.register_with(self)

        if type_ == SwapType.Payer:
            self._payer = [+1.0, -1.0]
        elif type_ == SwapType.Receiver:
            self._payer = [-1.0, +1.0]
        else:
            qassert.fail("Unknown BMA-swap type")

    # --- inspectors ----------------------------------------------------

    def libor_fraction(self) -> float:
        return self._libor_fraction

    def libor_spread(self) -> float:
        return self._libor_spread

    def nominal(self) -> float:
        return self._nominal

    def type(self) -> SwapType:
        """Payer / Receiver — the side refers to the BMA leg, not the Libor one."""
        return self._type

    def libor_leg(self) -> list[CashFlow]:
        return self._legs[0]

    def bma_leg(self) -> list[CashFlow]:
        return self._legs[1]

    # --- results -------------------------------------------------------

    def libor_leg_bps(self) -> float:
        """# C++ parity: ``BMASwap::liborLegBPS`` (bmaswap.cpp:107-111)."""
        self.calculate()
        bps = self._leg_bps[0]
        qassert.require(bps is not None, "result not available")
        assert bps is not None
        return bps

    def libor_leg_npv(self) -> float:
        """# C++ parity: ``BMASwap::liborLegNPV`` (bmaswap.cpp:113-117)."""
        self.calculate()
        npv = self._leg_npv[0]
        qassert.require(npv is not None, "result not available")
        assert npv is not None
        return npv

    def bma_leg_bps(self) -> float:
        """# C++ parity: ``BMASwap::bmaLegBPS`` (bmaswap.cpp:135-139)."""
        self.calculate()
        bps = self._leg_bps[1]
        qassert.require(bps is not None, "result not available")
        assert bps is not None
        return bps

    def bma_leg_npv(self) -> float:
        """# C++ parity: ``BMASwap::bmaLegNPV`` (bmaswap.cpp:141-145)."""
        self.calculate()
        npv = self._leg_npv[1]
        qassert.require(npv is not None, "result not available")
        assert npv is not None
        return npv

    def fair_libor_fraction(self) -> float:
        """# C++ parity: ``BMASwap::fairLiborFraction`` (bmaswap.cpp:119-127)."""
        spread_npv = (self._libor_spread / _BASIS_POINT) * self.libor_leg_bps()
        pure_libor_npv = self.libor_leg_npv() - spread_npv
        qassert.require(pure_libor_npv != 0.0, "result not available (null libor NPV)")
        return -self._libor_fraction * (self.bma_leg_npv() + spread_npv) / pure_libor_npv

    def fair_libor_spread(self) -> float:
        """# C++ parity: ``BMASwap::fairLiborSpread`` (bmaswap.cpp:129-133)."""
        return self._libor_spread - self.npv() / (self.libor_leg_bps() / _BASIS_POINT)


__all__ = ["BMASwap"]
