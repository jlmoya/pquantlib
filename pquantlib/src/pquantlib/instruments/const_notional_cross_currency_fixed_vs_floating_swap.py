"""ConstNotionalCrossCurrencyFixedVsFloatingSwap — fixed vs floating, two currencies.

# C++ parity: ql/instruments/constnotionalcrosscurrencyfixedvsfloatingswap.{hpp,cpp}
  — new in v1.43.

One leg pays a fixed rate in one currency, the other a floating rate in
another, and both exchange notional at each end. ``SwapType.Payer`` puts the
fixed leg first (i.e. the fixed leg is paid), ``SwapType.Receiver`` puts the
floating leg first.

The floating leg is built from an ``IborLeg`` or, when the index is an
:class:`~pquantlib.indexes.overnight_index.OvernightIndex`, from an
``OvernightLeg`` — the same ``dynamic_pointer_cast`` dispatch C++ performs.

Carve-out — overnight modifiers
-------------------------------
As in the basis swap, the ``float_compound_spread`` / ``float_lookback_days`` /
``float_observation_shift`` / ``float_lockout_days`` /
``float_averaging_method`` / ``telescopic_value_dates`` arguments exist so the
C++ signature is preserved, but PQuantLib's ``OvernightIndexedCoupon``
implements none of them; a non-default value raises ``NotImplementedError``
rather than being silently ignored.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.cashflows.overnight_leg import overnight_leg
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.const_notional_cross_currency_swap import (
    ConstNotionalCrossCurrencySwap,
    ConstNotionalCrossCurrencySwapArguments,
    ConstNotionalCrossCurrencySwapResults,
)
from pquantlib.instruments.swap import (
    Leg,
    SwapType,
    leg_maturity_date,
    leg_start_date,
)
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.schedule import Schedule


class ConstNotionalCrossCurrencyFixedVsFloatingSwapArguments(
    ConstNotionalCrossCurrencySwapArguments
):
    """Arguments carrier — adds the fixed rate and the floating spread.

    # C++ parity: ``ConstNotionalCrossCurrencyFixedVsFloatingSwap::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fixed_rate: float | None = None
        self.spread: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.fixed_rate is not None, "fixed rate cannot be null")
        qassert.require(self.spread is not None, "spread cannot be null")


class ConstNotionalCrossCurrencyFixedVsFloatingSwapResults(
    ConstNotionalCrossCurrencySwapResults
):
    """Results carrier — adds the fair fixed rate and the fair spread.

    # C++ parity: ``ConstNotionalCrossCurrencyFixedVsFloatingSwap::results``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fair_fixed_rate: float | None = None
        self.fair_spread: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_fixed_rate = None
        self.fair_spread = None


class ConstNotionalCrossCurrencyFixedVsFloatingSwap(ConstNotionalCrossCurrencySwap):
    """Cross-currency swap paying fixed in one currency against floating in another."""

    _BASIS_POINT: float = 1.0e-4

    def __init__(
        self,
        swap_type: SwapType,
        fixed_nominal: float,
        fixed_currency: Currency,
        fixed_schedule: Schedule,
        fixed_rate: float,
        fixed_day_count: DayCounter,
        fixed_payment_bdc: BusinessDayConvention,
        fixed_payment_lag: int,
        fixed_payment_calendar: Calendar,
        float_nominal: float,
        float_currency: Currency,
        float_schedule: Schedule,
        float_index: IborIndexProtocol,
        float_spread: float,
        float_payment_bdc: BusinessDayConvention,
        float_payment_lag: int,
        float_payment_calendar: Calendar,
        telescopic_value_dates: bool = False,
        float_compound_spread: bool = False,
        float_lookback_days: int | None = None,
        float_observation_shift: bool = False,
        float_lockout_days: int = 0,
        float_averaging_method: RateAveraging = RateAveraging.Compound,
    ) -> None:
        """Mirrors the C++ constructor argument-for-argument.

        ``float_lookback_days`` uses ``None`` where C++ uses ``Null<Natural>()``.
        """
        super().__init__(2)
        self._swap_type: SwapType = swap_type

        self._fixed_nominal: float = fixed_nominal
        self._fixed_currency: Currency = fixed_currency
        self._fixed_schedule: Schedule = fixed_schedule
        self._fixed_rate: float = fixed_rate
        self._fixed_day_count: DayCounter = fixed_day_count
        self._fixed_payment_bdc: BusinessDayConvention = fixed_payment_bdc
        self._fixed_payment_lag: int = fixed_payment_lag
        self._fixed_payment_calendar: Calendar = fixed_payment_calendar

        self._float_nominal: float = float_nominal
        self._float_currency: Currency = float_currency
        self._float_schedule: Schedule = float_schedule
        self._float_index: IborIndexProtocol = float_index
        self._float_spread: float = float_spread
        self._float_payment_bdc: BusinessDayConvention = float_payment_bdc
        self._float_payment_lag: int = float_payment_lag
        self._float_payment_calendar: Calendar = float_payment_calendar

        self._telescopic_value_dates: bool = telescopic_value_dates
        self._float_compound_spread: bool = float_compound_spread
        self._float_lookback_days: int | None = float_lookback_days
        self._float_observation_shift: bool = float_observation_shift
        self._float_lockout_days: int = float_lockout_days
        self._float_averaging_method: RateAveraging = float_averaging_method

        self._fair_fixed_rate: float | None = None
        self._fair_spread: float | None = None

        float_leg = self._build_float_leg()
        for cf in float_leg:
            cf.register_with(self)

        fixed_leg = fixed_rate_leg(
            fixed_schedule,
            nominals=[fixed_nominal],
            rates=[fixed_rate],
            day_counter=fixed_day_count,
            payment_adjustment=fixed_payment_bdc,
            payment_calendar=fixed_payment_calendar,
            payment_lag=fixed_payment_lag,
        )

        earliest_date = min(leg_start_date(float_leg), leg_start_date(fixed_leg))
        maturity_date = max(leg_maturity_date(float_leg), leg_maturity_date(fixed_leg))

        self.add_notional_exchanges_to_leg(
            float_leg,
            float_payment_calendar,
            earliest_date,
            maturity_date,
            float_payment_lag,
            float_payment_bdc,
            float_nominal,
        )
        self.add_notional_exchanges_to_leg(
            fixed_leg,
            fixed_payment_calendar,
            earliest_date,
            maturity_date,
            fixed_payment_lag,
            fixed_payment_bdc,
            fixed_nominal,
        )

        # Deriving from the cross-currency swap, where the first leg holds the
        # pay flows and the second the receive flows.
        self._payer[0] = -1.0
        self._payer[1] = +1.0
        if swap_type == SwapType.Payer:
            self._legs[0] = fixed_leg
            self._currencies[0] = fixed_currency
            self._legs[1] = float_leg
            self._currencies[1] = float_currency
        elif swap_type == SwapType.Receiver:
            self._legs[1] = fixed_leg
            self._currencies[1] = fixed_currency
            self._legs[0] = float_leg
            self._currencies[0] = float_currency
        else:
            qassert.fail("unknown cross currency fix float swap type")

    # --- construction ------------------------------------------------------

    def _build_float_leg(self) -> Leg:
        """Overnight leg when the index is an OvernightIndex, else an IBOR leg.

        # C++ parity: the ``dynamic_pointer_cast<OvernightIndex>`` branch of the
        # ``ConstNotionalCrossCurrencyFixedVsFloatingSwap`` constructor. Unlike
        # the basis swap, both branches here *do* set the payment adjustment and
        # calendar explicitly, from the ``floatPaymentBdc`` / ``floatPaymentCalendar``
        # arguments.
        """
        if isinstance(self._float_index, OvernightIndex):
            self.reject_unsupported_overnight_modifiers(
                "float",
                compound_spread=self._float_compound_spread,
                lookback_days=self._float_lookback_days,
                observation_shift=self._float_observation_shift,
                lockout_days=self._float_lockout_days,
                averaging_method=self._float_averaging_method,
                telescopic_value_dates=self._telescopic_value_dates,
            )
            return overnight_leg(
                self._float_schedule,
                self._float_index,
                nominals=[self._float_nominal],
                payment_adjustment=self._float_payment_bdc,
                payment_calendar=self._float_payment_calendar,
                payment_lag=self._float_payment_lag,
                spreads=self._float_spread,
            )
        leg = ibor_leg(
            self._float_schedule,
            self._float_index,
            nominals=[self._float_nominal],
            payment_adjustment=self._float_payment_bdc,
            payment_calendar=self._float_payment_calendar,
            payment_lag=self._float_payment_lag,
            spreads=self._float_spread,
        )
        # C++'s ``IborLeg::operator Leg()`` attaches a default
        # BlackIborCouponPricer when there are no caps/floors and the coupons
        # are not in arrears. PQuantLib's ``ibor_leg`` builder leaves the leg
        # pricerless and each instrument attaches the plain IborCouponPricer
        # instead — same swaplet rate with no vol surface, and the convention
        # VanillaSwap already follows.
        set_coupon_pricer(leg, IborCouponPricer())
        return leg

    # --- Instrument interface ----------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Add the fixed rate and the floating spread on top of the base carrier.

        # C++ parity: ``ConstNotionalCrossCurrencyFixedVsFloatingSwap::setupArguments``.
        """
        super().setup_arguments(args)
        if not isinstance(
            args, ConstNotionalCrossCurrencyFixedVsFloatingSwapArguments
        ):
            return
        args.fixed_rate = self._fixed_rate
        args.spread = self._float_spread

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull the fair rate / spread, deriving them from the leg BPS if absent.

        # C++ parity: ``ConstNotionalCrossCurrencyFixedVsFloatingSwap::fetchResults``.
        """
        super().fetch_results(results)

        if isinstance(results, ConstNotionalCrossCurrencyFixedVsFloatingSwapResults):
            self._fair_fixed_rate = results.fair_fixed_rate
            self._fair_spread = results.fair_spread
        else:
            self._fair_fixed_rate = None
            self._fair_spread = None

        npv = self._npv if self._npv is not None else 0.0
        idx_fixed = 0 if self._swap_type == SwapType.Payer else 1
        fixed_bps = self._leg_bps[idx_fixed]
        if self._fair_fixed_rate is None and fixed_bps is not None:
            self._fair_fixed_rate = self._fixed_rate - npv / (
                fixed_bps / self._BASIS_POINT
            )

        idx_float = 1 if self._swap_type == SwapType.Payer else 0
        float_bps = self._leg_bps[idx_float]
        if self._fair_spread is None and float_bps is not None:
            self._fair_spread = self._float_spread - npv / (
                float_bps / self._BASIS_POINT
            )

    def setup_expired(self) -> None:
        """# C++ parity: ``ConstNotionalCrossCurrencyFixedVsFloatingSwap::setupExpired``."""
        super().setup_expired()
        self._fair_fixed_rate = None
        self._fair_spread = None

    # --- inspectors --------------------------------------------------------

    def swap_type(self) -> SwapType:
        return self._swap_type

    def fixed_nominal(self) -> float:
        return self._fixed_nominal

    def fixed_currency(self) -> Currency:
        return self._fixed_currency

    def fixed_schedule(self) -> Schedule:
        return self._fixed_schedule

    def fixed_rate(self) -> float:
        return self._fixed_rate

    def fixed_day_count(self) -> DayCounter:
        return self._fixed_day_count

    def fixed_payment_bdc(self) -> BusinessDayConvention:
        return self._fixed_payment_bdc

    def fixed_payment_lag(self) -> int:
        return self._fixed_payment_lag

    def fixed_payment_calendar(self) -> Calendar:
        return self._fixed_payment_calendar

    def float_nominal(self) -> float:
        return self._float_nominal

    def float_currency(self) -> Currency:
        return self._float_currency

    def float_schedule(self) -> Schedule:
        return self._float_schedule

    def float_index(self) -> IborIndexProtocol:
        return self._float_index

    def float_spread(self) -> float:
        return self._float_spread

    def float_payment_bdc(self) -> BusinessDayConvention:
        return self._float_payment_bdc

    def float_payment_lag(self) -> int:
        return self._float_payment_lag

    def float_payment_calendar(self) -> Calendar:
        return self._float_payment_calendar

    def float_compound_spread(self) -> bool:
        return self._float_compound_spread

    def float_lookback_days(self) -> int | None:
        return self._float_lookback_days

    def float_lockout_days(self) -> int:
        return self._float_lockout_days

    def float_averaging_method(self) -> RateAveraging:
        return self._float_averaging_method

    # --- results -----------------------------------------------------------

    def fair_rate(self) -> float:
        """Fixed rate that would make the swap worth zero."""
        self.calculate()
        qassert.require(
            self._fair_fixed_rate is not None, "fair fixed rate is not available"
        )
        assert self._fair_fixed_rate is not None
        return self._fair_fixed_rate

    def fair_spread(self) -> float:
        """Floating-leg spread that would make the swap worth zero."""
        self.calculate()
        qassert.require(self._fair_spread is not None, "fair spread is not available")
        assert self._fair_spread is not None
        return self._fair_spread


__all__ = [
    "ConstNotionalCrossCurrencyFixedVsFloatingSwap",
    "ConstNotionalCrossCurrencyFixedVsFloatingSwapArguments",
    "ConstNotionalCrossCurrencyFixedVsFloatingSwapResults",
]
