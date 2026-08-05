"""DigitalIborCoupon + digital_ibor_leg — digital option on an Ibor coupon.

# C++ parity: ql/cashflows/digitaliborcoupon.{hpp,cpp} (v1.42.1, 099987f0).

:class:`DigitalIborCoupon` is a :class:`~pquantlib.cashflows.digital_coupon.DigitalCoupon`
whose underlying is an :class:`~pquantlib.cashflows.ibor_coupon.IborCoupon`.
``digital_ibor_leg`` is the free-function port of the C++ ``DigitalIborLeg``
chained builder (the ``with*`` setters become keyword arguments — same idiom
as :func:`~pquantlib.cashflows.ibor_leg.ibor_leg`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.cashflows import cash_flow_vectors as cfv
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.digital_coupon import DigitalCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.position import PositionType
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.cashflows.replication import DigitalReplication
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.schedule import Schedule


class DigitalIborCoupon(DigitalCoupon):
    """Digital call/put option on an Ibor coupon.

    # C++ parity: ql/cashflows/digitaliborcoupon.hpp:36-55 + .cpp:28-42.
    """

    def __init__(
        self,
        underlying: IborCoupon,
        call_strike: float | None = None,
        call_position: PositionType = PositionType.Long,
        is_call_atm_included: bool = False,
        call_digital_payoff: float | None = None,
        put_strike: float | None = None,
        put_position: PositionType = PositionType.Long,
        is_put_atm_included: bool = False,
        put_digital_payoff: float | None = None,
        replication: DigitalReplication | None = None,
        naked_option: bool = False,
    ) -> None:
        super().__init__(
            underlying,
            call_strike,
            call_position,
            is_call_atm_included,
            call_digital_payoff,
            put_strike,
            put_position,
            is_put_atm_included,
            put_digital_payoff,
            replication,
            naked_option,
        )


class DigitalIborLeg:
    """Chained builder for a sequence of digital Ibor coupons.

    # C++ parity: ``DigitalIborLeg`` (digitaliborcoupon.hpp:58-110,
    # .cpp:54-200).

    Every C++ ``withXxx`` setter is present as ``with_xxx`` and returns
    ``self``; C++'s ``operator Leg()`` is :meth:`build`.

    # C++ parity divergence: C++'s ``nakedOption_`` member has no default
    # initialiser and the constructor does not set it (digitaliborcoupon.hpp:108,
    # .cpp:54-55), so a leg built without ``withNakedOption`` reads an
    # indeterminate value. Python defaults it to ``False``, which is
    # ``DigitalCoupon``'s own default and the only sane reading.
    """

    def __init__(self, schedule: Schedule, index: IborIndexProtocol) -> None:
        self._schedule: Schedule = schedule
        self._index: IborIndexProtocol = index
        self._notionals: list[float] = []
        self._payment_day_counter: DayCounter | None = None
        self._payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following
        self._fixing_days: list[int] = []
        self._gearings: list[float] = []
        self._spreads: list[float] = []
        self._in_arrears: bool = False
        self._call_strikes: list[float] = []
        self._call_payoffs: list[float] = []
        self._long_call_option: PositionType = PositionType.Long
        self._call_atm: bool = False
        self._put_strikes: list[float] = []
        self._put_payoffs: list[float] = []
        self._long_put_option: PositionType = PositionType.Long
        self._put_atm: bool = False
        self._replication: DigitalReplication | None = None
        self._naked_option: bool = False

    # --- chained setters -----------------------------------------------

    def with_notionals(self, notionals: float | Sequence[float]) -> DigitalIborLeg:
        """# C++ parity: ``withNotionals`` (.cpp:57-66)."""
        self._notionals = cfv.as_float_list(notionals)
        return self

    def with_payment_day_counter(self, day_counter: DayCounter) -> DigitalIborLeg:
        """# C++ parity: ``withPaymentDayCounter`` (.cpp:68-72)."""
        self._payment_day_counter = day_counter
        return self

    def with_payment_adjustment(
        self, convention: BusinessDayConvention
    ) -> DigitalIborLeg:
        """# C++ parity: ``withPaymentAdjustment`` (.cpp:74-78)."""
        self._payment_adjustment = convention
        return self

    def with_fixing_days(self, fixing_days: int | Sequence[int]) -> DigitalIborLeg:
        """# C++ parity: ``withFixingDays`` (.cpp:80-89)."""
        self._fixing_days = cfv.as_int_list(fixing_days)
        return self

    def with_gearings(self, gearings: float | Sequence[float]) -> DigitalIborLeg:
        """# C++ parity: ``withGearings`` (.cpp:91-100).

        A gearing of exactly ``0.0`` degenerates the period to a fixed
        coupon, as in ``FloatingDigitalLeg``.
        """
        self._gearings = cfv.as_float_list(gearings)
        return self

    def with_spreads(self, spreads: float | Sequence[float]) -> DigitalIborLeg:
        """# C++ parity: ``withSpreads`` (.cpp:102-111)."""
        self._spreads = cfv.as_float_list(spreads)
        return self

    def in_arrears(self, flag: bool = True) -> DigitalIborLeg:
        """# C++ parity: ``inArrears`` (.cpp:113-116)."""
        self._in_arrears = flag
        return self

    def with_call_strikes(self, strikes: float | Sequence[float]) -> DigitalIborLeg:
        """# C++ parity: ``withCallStrikes`` (.cpp:118-127)."""
        self._call_strikes = cfv.as_float_list(strikes)
        return self

    def with_long_call_option(self, position: PositionType) -> DigitalIborLeg:
        """# C++ parity: ``withLongCallOption`` (.cpp:129-132)."""
        self._long_call_option = position
        return self

    def with_call_atm(self, flag: bool = True) -> DigitalIborLeg:
        """# C++ parity: ``withCallATM`` (.cpp:134-137)."""
        self._call_atm = flag
        return self

    def with_call_payoffs(self, payoffs: float | Sequence[float]) -> DigitalIborLeg:
        """# C++ parity: ``withCallPayoffs`` (.cpp:139-148)."""
        self._call_payoffs = cfv.as_float_list(payoffs)
        return self

    def with_put_strikes(self, strikes: float | Sequence[float]) -> DigitalIborLeg:
        """# C++ parity: ``withPutStrikes`` (.cpp:150-159)."""
        self._put_strikes = cfv.as_float_list(strikes)
        return self

    def with_long_put_option(self, position: PositionType) -> DigitalIborLeg:
        """# C++ parity: ``withLongPutOption`` (.cpp:161-164)."""
        self._long_put_option = position
        return self

    def with_put_atm(self, flag: bool = True) -> DigitalIborLeg:
        """# C++ parity: ``withPutATM`` (.cpp:166-169)."""
        self._put_atm = flag
        return self

    def with_put_payoffs(self, payoffs: float | Sequence[float]) -> DigitalIborLeg:
        """# C++ parity: ``withPutPayoffs`` (.cpp:171-180)."""
        self._put_payoffs = cfv.as_float_list(payoffs)
        return self

    def with_replication(self, replication: DigitalReplication) -> DigitalIborLeg:
        """# C++ parity: ``withReplication`` (.cpp:182-186)."""
        self._replication = replication
        return self

    def with_naked_option(self, naked_option: bool = True) -> DigitalIborLeg:
        """# C++ parity: ``withNakedOption`` (.cpp:188-191)."""
        self._naked_option = naked_option
        return self

    # --- operator Leg() -------------------------------------------------

    def _make_underlying(self, spec: cfv.FloatingCouponSpec) -> IborCoupon:
        return IborCoupon(
            spec.payment_date,
            spec.nominal,
            spec.accrual_start_date,
            spec.accrual_end_date,
            spec.fixing_days,
            self._index,
            spec.gearing,
            spec.spread,
            spec.ref_period_start,
            spec.ref_period_end,
            self._payment_day_counter
            if self._payment_day_counter is not None
            else self._index.day_counter(),
            self._in_arrears,
        )

    def _make_digital(
        self,
        underlying: FloatingRateCoupon,
        call_strike: float | None,
        call_payoff: float | None,
        put_strike: float | None,
        put_payoff: float | None,
    ) -> CashFlow:
        assert isinstance(underlying, IborCoupon)
        return DigitalIborCoupon(
            underlying,
            call_strike,
            self._long_call_option,
            self._call_atm,
            call_payoff,
            put_strike,
            self._long_put_option,
            self._put_atm,
            put_payoff,
            self._replication,
            self._naked_option,
        )

    def build(self) -> list[CashFlow]:
        """Build the leg.

        # C++ parity: ``DigitalIborLeg::operator Leg()`` (.cpp:193-200).
        """
        return cfv.floating_digital_leg(
            self._schedule,
            self._notionals,
            self._index.fixing_days(),
            self._payment_day_counter,
            self._payment_adjustment,
            self._fixing_days,
            self._gearings,
            self._spreads,
            self._call_strikes,
            self._call_payoffs,
            self._put_strikes,
            self._put_payoffs,
            self._make_underlying,
            self._make_digital,
        )


def digital_ibor_leg(
    schedule: Schedule,
    index: IborIndexProtocol,
    nominals: float | Sequence[float],
    *,
    payment_day_counter: DayCounter | None = None,
    payment_adjustment: BusinessDayConvention = BusinessDayConvention.Following,
    fixing_days: int | None = None,
    gearings: float | Sequence[float] = 1.0,
    spreads: float | Sequence[float] = 0.0,
    in_arrears: bool = False,
    call_strikes: float | Sequence[float] | None = None,
    long_call_option: PositionType = PositionType.Long,
    call_atm: bool = False,
    call_payoffs: float | Sequence[float] | None = None,
    put_strikes: float | Sequence[float] | None = None,
    long_put_option: PositionType = PositionType.Long,
    put_atm: bool = False,
    put_payoffs: float | Sequence[float] | None = None,
    replication: DigitalReplication | None = None,
    naked_option: bool = False,
) -> list[CashFlow]:
    """Build a leg of :class:`DigitalIborCoupon` from a schedule + Ibor index.

    Keyword-argument façade over :class:`DigitalIborLeg`; the leg logic lives
    there, so there is exactly one implementation.

    ``fixing_days`` defaults to ``index.fixing_days()``; ``payment_day_counter``
    defaults to ``index.day_counter()``.
    """
    builder = (
        DigitalIborLeg(schedule, index)
        .with_notionals(nominals)
        .with_payment_adjustment(payment_adjustment)
        .with_gearings(gearings)
        .with_spreads(spreads)
        .in_arrears(in_arrears)
        .with_long_call_option(long_call_option)
        .with_call_atm(call_atm)
        .with_long_put_option(long_put_option)
        .with_put_atm(put_atm)
        .with_naked_option(naked_option)
    )
    if payment_day_counter is not None:
        builder = builder.with_payment_day_counter(payment_day_counter)
    if fixing_days is not None:
        builder = builder.with_fixing_days(fixing_days)
    if call_strikes is not None:
        builder = builder.with_call_strikes(call_strikes)
    if call_payoffs is not None:
        builder = builder.with_call_payoffs(call_payoffs)
    if put_strikes is not None:
        builder = builder.with_put_strikes(put_strikes)
    if put_payoffs is not None:
        builder = builder.with_put_payoffs(put_payoffs)
    if replication is not None:
        builder = builder.with_replication(replication)
    return builder.build()


__all__ = ["DigitalIborCoupon", "DigitalIborLeg", "digital_ibor_leg"]
