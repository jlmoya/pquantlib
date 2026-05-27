"""FixedVsFloatingSwap — abstract fixed-rate vs floating-rate swap.

# C++ parity: ql/instruments/fixedvsfloatingswap.{hpp,cpp} (v1.42.1).

Intermediate base between ``Swap`` and concrete fixed-vs-floating swaps
(``VanillaSwap`` for IBOR, ``OvernightIndexedSwap`` for overnight rates).
The base builds the fixed leg in its constructor and leaves the floating
leg for the subclass to populate. ``fair_rate`` / ``fair_spread`` are
derived from per-leg BPS / NPV.

C++ uses a Visitor-via-arguments dispatch (``setupFloatingArguments``)
that fills product-specific argument fields on the engine's argument
carrier. Python uses simple subclass overrides — the
``DiscountingSwapEngine`` doesn't actually need those per-product
arguments (it works off ``SwapArguments.legs`` only), so the carrier
extension is light.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.swap import Swap, SwapArguments, SwapResults, SwapType
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.schedule import Schedule


class FixedVsFloatingSwapArguments(SwapArguments):
    """Engine arguments carrier for FixedVsFloatingSwap.

    # C++ parity: ``FixedVsFloatingSwap::arguments`` (fixedvsfloatingswap.hpp:131-151).
    """

    def __init__(self) -> None:
        super().__init__()
        self.swap_type: SwapType = SwapType.Receiver
        self.nominal: float | None = None
        # Fixed leg fields (denormalised from the leg cashflows).
        self.fixed_nominals: list[float] = []
        self.fixed_reset_dates: list[object] = []
        self.fixed_pay_dates: list[object] = []
        self.fixed_coupons: list[float] = []
        # Floating leg fields (subclass-filled).
        self.floating_nominals: list[float] = []
        self.floating_accrual_times: list[float] = []
        self.floating_reset_dates: list[object] = []
        self.floating_fixing_dates: list[object] = []
        self.floating_pay_dates: list[object] = []
        self.floating_spreads: list[float] = []
        self.floating_coupons: list[float | None] = []


class FixedVsFloatingSwapResults(SwapResults):
    """Engine results carrier with fair_rate / fair_spread fields.

    # C++ parity: ``FixedVsFloatingSwap::results`` (fixedvsfloatingswap.hpp:154-159).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fair_rate: float | None = None
        self.fair_spread: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_rate = None
        self.fair_spread = None


class FixedVsFloatingSwap(Swap):
    """Abstract fixed-vs-floating-rate swap.

    Constructs the fixed leg in ``__init__`` from the fixed schedule +
    nominals + rate + day-counter. The floating leg is left for the
    subclass to build.
    """

    _BASIS_POINT: float = 1.0e-4

    def __init__(
        self,
        swap_type: SwapType,
        fixed_nominals: list[float],
        fixed_schedule: Schedule,
        fixed_rate: float,
        fixed_day_count: DayCounter,
        floating_nominals: list[float],
        floating_schedule: Schedule,
        ibor_index: IborIndexProtocol,
        spread: float,
        floating_day_count: DayCounter,
        payment_convention: BusinessDayConvention | None = None,
        payment_lag: int = 0,
        payment_calendar: Calendar | None = None,
    ) -> None:
        # C++ parity: FixedVsFloatingSwap::FixedVsFloatingSwap (fixedvsfloatingswap.cpp:33-104).
        # Python divergence: the C++ ctor accepts an empty ``DayCounter()``
        # and falls back to ``iborIndex_->dayCounter()``. Python has no
        # null-DayCounter — callers must pass an explicit DayCounter (the
        # parent ``VanillaSwap`` ctor + leg builders enforce this).
        super().__init__(n_legs=2)

        self._fixed_day_count: DayCounter = fixed_day_count
        # Resolve payment_convention: caller wins, else floating
        # schedule's business-day convention.
        self._payment_convention: BusinessDayConvention = (
            payment_convention
            if payment_convention is not None
            else floating_schedule.business_day_convention
        )

        self._swap_type: SwapType = swap_type
        self._fixed_nominals: list[float] = list(fixed_nominals)
        self._fixed_schedule: Schedule = fixed_schedule
        self._fixed_rate: float = fixed_rate
        self._floating_nominals: list[float] = list(floating_nominals)
        self._floating_schedule: Schedule = floating_schedule
        self._ibor_index: IborIndexProtocol = ibor_index
        self._spread: float = spread
        self._floating_day_count: DayCounter = floating_day_count
        self._payment_lag: int = payment_lag
        self._payment_calendar: Calendar | None = payment_calendar

        # Build the fixed leg now.
        pay_cal = (
            payment_calendar
            if payment_calendar is not None
            else fixed_schedule.calendar
        )
        self._legs[0] = fixed_rate_leg(
            fixed_schedule,
            self._fixed_nominals,
            [fixed_rate],
            day_counter=self._fixed_day_count,
            payment_adjustment=self._payment_convention,
            payment_calendar=pay_cal,
        )
        for cf in self._legs[0]:
            cf.register_with(self)

        # Subclass populates self._legs[1]; we wire payer signs.
        if swap_type == SwapType.Payer:
            self._payer[0] = -1.0
            self._payer[1] = +1.0
        elif swap_type == SwapType.Receiver:
            self._payer[0] = +1.0
            self._payer[1] = -1.0
        else:
            qassert.fail("Unknown vanilla-swap type")

        # Sample-nominals helpers — mirror C++ constantNominals_ / sameNominals_.
        self._same_nominals: bool = list(fixed_nominals) == list(floating_nominals)
        if not self._same_nominals:
            self._constant_nominals: bool = False
        else:
            front = fixed_nominals[0] if fixed_nominals else 0.0
            self._constant_nominals = all(x == front for x in fixed_nominals)

        # Cached fair_rate / fair_spread (filled by fetch_results).
        self._fair_rate: float | None = None
        self._fair_spread: float | None = None

    # --- subclass hook ------------------------------------------------------

    @abstractmethod
    def _setup_floating_arguments(self, args: FixedVsFloatingSwapArguments) -> None:
        """Subclass: fill the floating-leg fields in the argument carrier.

        # C++ parity: ``FixedVsFloatingSwap::setupFloatingArguments`` —
        # pure-virtual hook invoked from ``setupArguments``.
        """

    # --- Instrument / Swap interface ---------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Populate the engine arguments — legs + payer (from Swap) +
        product-specific fixed-leg fields + delegate floating-leg fields
        to the subclass.

        # C++ parity: ``FixedVsFloatingSwap::setupArguments`` (fixedvsfloatingswap.cpp:106-138).
        """
        # Allow the engine to be a plain SwapEngine without our extended
        # argument carrier — in that case skip the product-specific fields.
        super().setup_arguments(args)
        if not isinstance(args, FixedVsFloatingSwapArguments):
            return

        args.swap_type = self._swap_type
        args.nominal = (
            self._fixed_nominals[0] if self._constant_nominals else None
        )

        n_fixed = len(self._legs[0])
        args.fixed_pay_dates = [None] * n_fixed
        args.fixed_reset_dates = [None] * n_fixed
        args.fixed_nominals = [0.0] * n_fixed
        args.fixed_coupons = [0.0] * n_fixed

        for i, cf in enumerate(self._legs[0]):
            qassert.require(
                isinstance(cf, FixedRateCoupon),
                "fixed leg expected to contain FixedRateCoupons",
            )
            assert isinstance(cf, FixedRateCoupon)
            args.fixed_pay_dates[i] = cf.date()
            args.fixed_reset_dates[i] = cf.accrual_start_date()
            args.fixed_coupons[i] = cf.amount()
            args.fixed_nominals[i] = cf.nominal()
        self._setup_floating_arguments(args)

    def setup_expired(self) -> None:
        """Mirror C++ expired setup — zero BPS, null fair_rate / fair_spread.

        # C++ parity: ``FixedVsFloatingSwap::setupExpired`` (fixedvsfloatingswap.cpp:176-182).
        """
        super().setup_expired()
        self._leg_bps[0] = 0.0
        if len(self._leg_bps) > 1:
            self._leg_bps[1] = 0.0
        self._fair_rate = None
        self._fair_spread = None

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull leg NPVs + BPS (from Swap) + fair_rate / fair_spread.

        If the engine returns a generic ``SwapResults`` (not our
        extended carrier), fair_rate / fair_spread are derived from the
        per-leg BPS + NPV — matches the C++ fallback path.

        # C++ parity: ``FixedVsFloatingSwap::fetchResults`` (fixedvsfloatingswap.cpp:183-207).
        """
        super().fetch_results(results)
        if isinstance(results, FixedVsFloatingSwapResults):
            self._fair_rate = results.fair_rate
            self._fair_spread = results.fair_spread
        else:
            self._fair_rate = None
            self._fair_spread = None
        # Always recompute from NPV+BPS when the engine didn't supply
        # them directly (matches C++ fallback in fetchResults).
        npv = self._npv if self._npv is not None else 0.0
        if self._fair_rate is None and self._leg_bps[0] is not None:
            self._fair_rate = self._fixed_rate - npv / (
                self._leg_bps[0] / self._BASIS_POINT
            )
        if self._fair_spread is None and self._leg_bps[1] is not None:
            self._fair_spread = self._spread - npv / (
                self._leg_bps[1] / self._BASIS_POINT
            )

    # --- inspectors --------------------------------------------------------

    def swap_type(self) -> SwapType:
        return self._swap_type

    def nominal(self) -> float:
        """C++ parity: throws if nominal isn't constant."""
        qassert.require(self._constant_nominals, "nominal is not constant")
        return self._fixed_nominals[0]

    def nominals(self) -> list[float]:
        """C++ parity: throws if fixed and floating nominals differ."""
        qassert.require(
            self._same_nominals,
            "different nominals on fixed and floating leg",
        )
        return list(self._fixed_nominals)

    def fixed_nominals(self) -> list[float]:
        return list(self._fixed_nominals)

    def fixed_schedule(self) -> Schedule:
        return self._fixed_schedule

    def fixed_rate(self) -> float:
        return self._fixed_rate

    def fixed_day_count(self) -> DayCounter:
        return self._fixed_day_count

    def floating_nominals(self) -> list[float]:
        return list(self._floating_nominals)

    def floating_schedule(self) -> Schedule:
        return self._floating_schedule

    def ibor_index(self) -> IborIndexProtocol:
        return self._ibor_index

    def spread(self) -> float:
        return self._spread

    def floating_day_count(self) -> DayCounter:
        return self._floating_day_count

    def payment_convention(self) -> BusinessDayConvention:
        return self._payment_convention

    def fixed_leg(self):  # type: ignore[no-untyped-def]
        return self._legs[0]

    def floating_leg(self):  # type: ignore[no-untyped-def]
        return self._legs[1]

    # --- results ----------------------------------------------------------

    def fixed_leg_bps(self) -> float:
        return self.leg_bps(0)

    def fixed_leg_npv(self) -> float:
        return self.leg_npv(0)

    def floating_leg_bps(self) -> float:
        return self.leg_bps(1)

    def floating_leg_npv(self) -> float:
        return self.leg_npv(1)

    def fair_rate(self) -> float:
        self.calculate()
        qassert.require(self._fair_rate is not None, "result not available")
        assert self._fair_rate is not None
        return self._fair_rate

    def fair_spread(self) -> float:
        self.calculate()
        qassert.require(self._fair_spread is not None, "result not available")
        assert self._fair_spread is not None
        return self._fair_spread


__all__ = [
    "FixedVsFloatingSwap",
    "FixedVsFloatingSwapArguments",
    "FixedVsFloatingSwapResults",
]
