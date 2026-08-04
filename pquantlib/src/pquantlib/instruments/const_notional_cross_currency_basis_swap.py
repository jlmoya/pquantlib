"""ConstNotionalCrossCurrencyBasisSwap — float/float cross-currency basis swap.

# C++ parity: ql/instruments/constnotionalcrosscurrencybasisswap.{hpp,cpp} — new in v1.43.

Both legs float, each in its own currency, and each exchanges notional at start
and at maturity. The first leg holds the pay-currency cashflows and the second
the receive-currency cashflows.

Each leg is built from an ``IborLeg`` or, when its index is an
:class:`~pquantlib.indexes.overnight_index.OvernightIndex`, from an
``OvernightLeg`` — the same ``dynamic_pointer_cast`` dispatch C++ performs.

Carve-out — overnight modifiers
-------------------------------
C++ forwards ``compoundSpreadDaily`` / ``lookbackDays`` / ``observationShift``
/ ``lockoutDays`` / ``averagingMethod`` / ``telescopicValueDates`` to
``OvernightLeg``. PQuantLib's ``OvernightIndexedCoupon`` implements none of
them: it compounds the daily fixings over the plain business-day series and
applies gearing/spread once, at the coupon level.

Rather than accept those arguments and quietly price as though they were off,
the constructor takes them (so the C++ signature is preserved) and raises
``NotImplementedError`` as soon as one is set to a non-default value. Anything
else would report a number that is not the number the caller asked for.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import IborCouponPricer, set_coupon_pricer
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.cashflows.overnight_leg import overnight_leg
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.currencies.currency import Currency
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.const_notional_cross_currency_swap import (
    ConstNotionalCrossCurrencySwap,
    ConstNotionalCrossCurrencySwapArguments,
    ConstNotionalCrossCurrencySwapResults,
)
from pquantlib.instruments.swap import Leg, leg_maturity_date, leg_start_date
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.termstructures.protocols import IborIndexProtocol
from pquantlib.time.schedule import Schedule


class ConstNotionalCrossCurrencyBasisSwapArguments(
    ConstNotionalCrossCurrencySwapArguments
):
    """Arguments carrier — adds the two leg spreads.

    # C++ parity: ``ConstNotionalCrossCurrencyBasisSwap::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pay_spread: float | None = None
        self.rec_spread: float | None = None

    def validate(self) -> None:
        super().validate()
        qassert.require(self.pay_spread is not None, "pay spread cannot be null")
        qassert.require(self.rec_spread is not None, "rec spread cannot be null")


class ConstNotionalCrossCurrencyBasisSwapResults(ConstNotionalCrossCurrencySwapResults):
    """Results carrier — adds the two fair spreads.

    # C++ parity: ``ConstNotionalCrossCurrencyBasisSwap::results``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fair_pay_spread: float | None = None
        self.fair_rec_spread: float | None = None

    def reset(self) -> None:
        super().reset()
        self.fair_pay_spread = None
        self.fair_rec_spread = None


class ConstNotionalCrossCurrencyBasisSwap(ConstNotionalCrossCurrencySwap):
    """Cross-currency basis swap — floating vs floating, notionals exchanged."""

    _BASIS_POINT: float = 1.0e-4

    def __init__(
        self,
        pay_nominal: float,
        pay_currency: Currency,
        pay_schedule: Schedule,
        pay_index: IborIndexProtocol,
        pay_spread: float,
        pay_gearing: float,
        rec_nominal: float,
        rec_currency: Currency,
        rec_schedule: Schedule,
        rec_index: IborIndexProtocol,
        rec_spread: float,
        rec_gearing: float,
        pay_payment_lag: int = 0,
        rec_payment_lag: int = 0,
        pay_compound_spread: bool = False,
        pay_lookback_days: int | None = None,
        pay_observation_shift: bool = False,
        pay_lockout_days: int = 0,
        pay_averaging_method: RateAveraging = RateAveraging.Compound,
        rec_compound_spread: bool = False,
        rec_lookback_days: int | None = None,
        rec_observation_shift: bool = False,
        rec_lockout_days: int = 0,
        rec_averaging_method: RateAveraging = RateAveraging.Compound,
        telescopic_value_dates: bool = False,
    ) -> None:
        """Mirrors the C++ constructor argument-for-argument.

        ``pay_lookback_days`` / ``rec_lookback_days`` use ``None`` where C++
        uses ``Null<Natural>()``.
        """
        super().__init__(2)
        self._pay_nominal: float = pay_nominal
        self._pay_currency: Currency = pay_currency
        self._pay_schedule: Schedule = pay_schedule
        self._pay_index: IborIndexProtocol = pay_index
        self._pay_spread: float = pay_spread
        self._pay_gearing: float = pay_gearing

        self._rec_nominal: float = rec_nominal
        self._rec_currency: Currency = rec_currency
        self._rec_schedule: Schedule = rec_schedule
        self._rec_index: IborIndexProtocol = rec_index
        self._rec_spread: float = rec_spread
        self._rec_gearing: float = rec_gearing

        self._pay_payment_lag: int = pay_payment_lag
        self._rec_payment_lag: int = rec_payment_lag

        # Overnight-only modifiers.
        self._pay_compound_spread: bool = pay_compound_spread
        self._pay_lookback_days: int | None = pay_lookback_days
        self._pay_observation_shift: bool = pay_observation_shift
        self._pay_lockout_days: int = pay_lockout_days
        self._pay_averaging_method: RateAveraging = pay_averaging_method
        self._rec_compound_spread: bool = rec_compound_spread
        self._rec_lookback_days: int | None = rec_lookback_days
        self._rec_observation_shift: bool = rec_observation_shift
        self._rec_lockout_days: int = rec_lockout_days
        self._rec_averaging_method: RateAveraging = rec_averaging_method
        self._telescopic_value_dates: bool = telescopic_value_dates

        self._fair_pay_spread: float | None = None
        self._fair_rec_spread: float | None = None

        register = getattr(pay_index, "register_with", None)
        if register is not None:
            register(self)
        register = getattr(rec_index, "register_with", None)
        if register is not None:
            register(self)

        self._initialize()

    # --- construction ------------------------------------------------------

    def _build_leg(
        self,
        schedule: Schedule,
        index: IborIndexProtocol,
        nominal: float,
        spread: float,
        gearing: float,
        payment_lag: int,
        compound_spread: bool,
        lookback_days: int | None,
        observation_shift: bool,
        lockout_days: int,
        averaging_method: RateAveraging,
        side: str,
    ) -> Leg:
        """One leg — overnight when the index is an OvernightIndex, else IBOR.

        Neither branch sets a payment adjustment or calendar: C++ leaves both
        ``IborLeg`` and ``OvernightLeg`` at their defaults here, i.e. ``Following``
        against the schedule's own calendar. That is deliberately *not* the
        schedule's business-day convention — the notional exchanges added later
        do use the schedule's convention, and this asymmetry is C++'s.

        # C++ parity: the two ``dynamic_pointer_cast<OvernightIndex>`` branches
        # of ``ConstNotionalCrossCurrencyBasisSwap::initialize``.
        """
        if isinstance(index, OvernightIndex):
            self.reject_unsupported_overnight_modifiers(
                side,
                compound_spread=compound_spread,
                lookback_days=lookback_days,
                observation_shift=observation_shift,
                lockout_days=lockout_days,
                averaging_method=averaging_method,
                telescopic_value_dates=self._telescopic_value_dates,
            )
            return overnight_leg(
                schedule,
                index,
                nominals=[nominal],
                payment_lag=payment_lag,
                gearings=gearing,
                spreads=spread,
            )
        leg = ibor_leg(
            schedule,
            index,
            nominals=[nominal],
            payment_lag=payment_lag,
            gearings=gearing,
            spreads=spread,
        )
        # C++'s ``IborLeg::operator Leg()`` attaches a default
        # BlackIborCouponPricer when there are no caps/floors and the coupons
        # are not in arrears. PQuantLib's ``ibor_leg`` builder leaves the leg
        # pricerless and each instrument attaches the plain IborCouponPricer
        # instead — same swaplet rate with no vol surface, and the convention
        # VanillaSwap already follows.
        set_coupon_pricer(leg, IborCouponPricer())
        return leg

    def _initialize(self) -> None:
        """Build both legs, attach the notional exchanges, wire the observers.

        # C++ parity: ``ConstNotionalCrossCurrencyBasisSwap::initialize``.
        """
        pay_leg = self._build_leg(
            self._pay_schedule,
            self._pay_index,
            self._pay_nominal,
            self._pay_spread,
            self._pay_gearing,
            self._pay_payment_lag,
            self._pay_compound_spread,
            self._pay_lookback_days,
            self._pay_observation_shift,
            self._pay_lockout_days,
            self._pay_averaging_method,
            "pay",
        )
        self._legs[0] = pay_leg
        self._payer[0] = -1.0
        self._currencies[0] = self._pay_currency

        rec_leg = self._build_leg(
            self._rec_schedule,
            self._rec_index,
            self._rec_nominal,
            self._rec_spread,
            self._rec_gearing,
            self._rec_payment_lag,
            self._rec_compound_spread,
            self._rec_lookback_days,
            self._rec_observation_shift,
            self._rec_lockout_days,
            self._rec_averaging_method,
            "rec",
        )
        self._legs[1] = rec_leg
        self._payer[1] = +1.0
        self._currencies[1] = self._rec_currency

        earliest_date = min(leg_start_date(pay_leg), leg_start_date(rec_leg))
        maturity_date = max(leg_maturity_date(pay_leg), leg_maturity_date(rec_leg))

        self.add_notional_exchanges_to_leg(
            pay_leg,
            self._pay_schedule.calendar,
            earliest_date,
            maturity_date,
            self._pay_payment_lag,
            self._pay_schedule.business_day_convention,
            self._pay_nominal,
        )
        self.add_notional_exchanges_to_leg(
            rec_leg,
            self._rec_schedule.calendar,
            earliest_date,
            maturity_date,
            self._rec_payment_lag,
            self._rec_schedule.business_day_convention,
            self._rec_nominal,
        )

        for leg in self._legs:
            for cf in leg:
                cf.register_with(self)

    # --- Instrument interface ----------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Add the two spreads on top of the base carrier.

        Returns early when the engine only understands the base
        cross-currency-swap arguments — mirrors the C++ null-``dynamic_cast``
        early return.

        # C++ parity: ``ConstNotionalCrossCurrencyBasisSwap::setupArguments``.
        """
        super().setup_arguments(args)
        if not isinstance(args, ConstNotionalCrossCurrencyBasisSwapArguments):
            return
        args.pay_spread = self._pay_spread
        args.rec_spread = self._rec_spread

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull the fair spreads, deriving them from the leg BPS when absent.

        # C++ parity: ``ConstNotionalCrossCurrencyBasisSwap::fetchResults``.
        """
        super().fetch_results(results)

        if isinstance(results, ConstNotionalCrossCurrencyBasisSwapResults):
            self._fair_pay_spread = results.fair_pay_spread
            self._fair_rec_spread = results.fair_rec_spread
        else:
            self._fair_pay_spread = None
            self._fair_rec_spread = None

        npv = self._npv if self._npv is not None else 0.0
        if self._fair_pay_spread is None and self._leg_bps[0] is not None:
            self._fair_pay_spread = self._pay_spread - npv / (
                self._leg_bps[0] / self._BASIS_POINT
            )
        if self._fair_rec_spread is None and self._leg_bps[1] is not None:
            self._fair_rec_spread = self._rec_spread - npv / (
                self._leg_bps[1] / self._BASIS_POINT
            )

    def setup_expired(self) -> None:
        """# C++ parity: ``ConstNotionalCrossCurrencyBasisSwap::setupExpired``."""
        super().setup_expired()
        self._fair_pay_spread = None
        self._fair_rec_spread = None

    # --- inspectors --------------------------------------------------------

    def pay_nominal(self) -> float:
        return self._pay_nominal

    def pay_currency(self) -> Currency:
        return self._pay_currency

    def pay_schedule(self) -> Schedule:
        return self._pay_schedule

    def pay_index(self) -> IborIndexProtocol:
        return self._pay_index

    def pay_spread(self) -> float:
        return self._pay_spread

    def pay_gearing(self) -> float:
        return self._pay_gearing

    def rec_nominal(self) -> float:
        return self._rec_nominal

    def rec_currency(self) -> Currency:
        return self._rec_currency

    def rec_schedule(self) -> Schedule:
        return self._rec_schedule

    def rec_index(self) -> IborIndexProtocol:
        return self._rec_index

    def rec_spread(self) -> float:
        return self._rec_spread

    def rec_gearing(self) -> float:
        return self._rec_gearing

    # --- results -----------------------------------------------------------

    def fair_pay_spread(self) -> float:
        """Pay-leg spread that would make the swap worth zero."""
        self.calculate()
        qassert.require(
            self._fair_pay_spread is not None, "fair pay spread is not available"
        )
        assert self._fair_pay_spread is not None
        return self._fair_pay_spread

    def fair_rec_spread(self) -> float:
        """Receive-leg spread that would make the swap worth zero."""
        self.calculate()
        qassert.require(
            self._fair_rec_spread is not None, "fair rec spread is not available"
        )
        assert self._fair_rec_spread is not None
        return self._fair_rec_spread


__all__ = [
    "ConstNotionalCrossCurrencyBasisSwap",
    "ConstNotionalCrossCurrencyBasisSwapArguments",
    "ConstNotionalCrossCurrencyBasisSwapResults",
]
