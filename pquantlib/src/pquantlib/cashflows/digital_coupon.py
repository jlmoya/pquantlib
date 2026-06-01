"""DigitalCoupon — floating coupon with an embedded digital call/put option.

# C++ parity: ql/cashflows/digitalcoupon.{hpp,cpp} (v1.42.1, 099987f0).

A :class:`~pquantlib.cashflows.floating_rate_coupon.FloatingRateCoupon` whose
rate carries a cash-or-nothing or asset-or-nothing digital call and/or put.
The digital option is valued by the **call/put-spread replication** method:
two :class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredCoupon`
instances at strikes ``strike ± gap`` are differenced to approximate the
step-function payoff.

Payoffs (``csi`` = +1 long / -1 short):

* cash-or-nothing call: ``rate + csi * payoffRate * H(rate - strike)``
* cash-or-nothing put:  ``rate + csi * payoffRate * H(strike - rate)``
* asset-or-nothing call: ``rate + csi * rate * H(rate - strike)``
* asset-or-nothing put:  ``rate + csi * rate * H(strike - rate)``

If ``naked_option`` is set, the underlying ``rate`` term is dropped.

# C++ parity divergences:
# - LazyObject deferred-calculation cache + Observer wiring not ported
#   (``rate()`` recomputes each call); consistent with the cashflows port.
# - ``accept`` / ``AcyclicVisitor`` Visitability not ported.
# - ``Position::Type`` → :class:`~pquantlib.position.PositionType`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.capped_floored_coupon import CappedFlooredCoupon
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.cashflows.replication import DigitalReplication, Replication
from pquantlib.position import PositionType

if TYPE_CHECKING:
    from pquantlib.cashflows.coupon_pricer import FloatingRateCouponPricer

# Heaviside threshold (C++ digitalcoupon.cpp uses 1e-16).
_EPS = 1.0e-16


class DigitalCoupon(FloatingRateCoupon):
    """Floating coupon with an embedded digital call/put option.

    # C++ parity: ql/cashflows/digitalcoupon.hpp:79-187 + .cpp:28-330.
    """

    def __init__(
        self,
        underlying: FloatingRateCoupon,
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
            underlying.date(),
            underlying.nominal(),
            underlying.accrual_start_date(),
            underlying.accrual_end_date(),
            underlying.fixing_days(),
            underlying.index(),
            underlying.gearing(),
            underlying.spread(),
            underlying.reference_period_start(),
            underlying.reference_period_end(),
            underlying.day_counter(),
            underlying.is_in_arrears(),
        )
        if replication is None:
            replication = DigitalReplication()

        self._underlying: FloatingRateCoupon = underlying
        self._is_call_atm_included: bool = is_call_atm_included
        self._is_put_atm_included: bool = is_put_atm_included
        self._naked_option: bool = naked_option

        gap = replication.gap()
        qassert.require(gap > 0.0, "Non positive epsilon not allowed")
        # Default Central-replication gaps.
        self._call_left_eps: float = gap / 2.0
        self._call_right_eps: float = gap / 2.0
        self._put_left_eps: float = gap / 2.0
        self._put_right_eps: float = gap / 2.0
        self._replication_type: Replication = replication.replication_type()

        self._has_call_strike: bool = False
        self._has_put_strike: bool = False
        self._call_strike: float = 0.0
        self._put_strike: float = 0.0
        self._call_csi: float = 0.0
        self._put_csi: float = 0.0
        self._is_call_cash_or_nothing: bool = False
        self._is_put_cash_or_nothing: bool = False
        self._call_digital_payoff: float = 0.0
        self._put_digital_payoff: float = 0.0

        # --- consistency checks (C++ digitalcoupon.cpp:62-69) -------------
        if put_strike is None:
            qassert.require(
                put_digital_payoff is None,
                "Put Cash rate non allowed if put strike is null",
            )
        if call_strike is None:
            qassert.require(
                call_digital_payoff is None,
                "Call Cash rate non allowed if call strike is null",
            )

        # --- call leg (C++ digitalcoupon.cpp:70-87) -----------------------
        if call_strike is not None:
            self._has_call_strike = True
            self._call_strike = call_strike
            self._call_csi = 1.0 if call_position == PositionType.Long else -1.0
            if call_digital_payoff is not None:
                self._call_digital_payoff = call_digital_payoff
                self._is_call_cash_or_nothing = True

        # --- put leg (C++ digitalcoupon.cpp:88-105) -----------------------
        if put_strike is not None:
            self._has_put_strike = True
            self._put_strike = put_strike
            self._put_csi = 1.0 if put_position == PositionType.Long else -1.0
            if put_digital_payoff is not None:
                self._put_digital_payoff = put_digital_payoff
                self._is_put_cash_or_nothing = True

        # --- replication-type gap placement (C++ digitalcoupon.cpp:107-173)
        self._apply_replication_gaps(gap, call_position, put_position)

    def _apply_replication_gaps(
        self, gap: float, call_position: PositionType, put_position: PositionType
    ) -> None:
        """Set left/right replication gaps per the replication type.

        # C++ parity: ql/cashflows/digitalcoupon.cpp:107-173.
        """
        if self._replication_type == Replication.Central:
            return  # default gap/2 split already set
        if self._replication_type == Replication.Sub:
            if self._has_call_strike:
                if call_position == PositionType.Long:
                    self._call_left_eps, self._call_right_eps = 0.0, gap
                else:
                    self._call_left_eps, self._call_right_eps = gap, 0.0
            if self._has_put_strike:
                if put_position == PositionType.Long:
                    self._put_left_eps, self._put_right_eps = gap, 0.0
                else:
                    self._put_left_eps, self._put_right_eps = 0.0, gap
        elif self._replication_type == Replication.Super:
            if self._has_call_strike:
                if call_position == PositionType.Long:
                    self._call_left_eps, self._call_right_eps = gap, 0.0
                else:
                    self._call_left_eps, self._call_right_eps = 0.0, gap
            if self._has_put_strike:
                if put_position == PositionType.Long:
                    self._put_left_eps, self._put_right_eps = 0.0, gap
                else:
                    self._put_left_eps, self._put_right_eps = gap, 0.0
        else:  # pragma: no cover - enum exhaustive
            qassert.fail("unsupported replication type")

    # --- inspectors ----------------------------------------------------

    def underlying(self) -> FloatingRateCoupon:
        """# C++ parity: ``DigitalCoupon::underlying``."""
        return self._underlying

    def has_call(self) -> bool:
        """# C++ parity: ``DigitalCoupon::hasCall`` (inline)."""
        return self._has_call_strike

    def has_put(self) -> bool:
        """# C++ parity: ``DigitalCoupon::hasPut`` (inline)."""
        return self._has_put_strike

    def has_collar(self) -> bool:
        """# C++ parity: ``DigitalCoupon::hasCollar`` (inline)."""
        return self._has_call_strike and self._has_put_strike

    def is_long_call(self) -> bool:
        """# C++ parity: ``DigitalCoupon::isLongCall`` (inline)."""
        return self._call_csi == 1.0

    def is_long_put(self) -> bool:
        """# C++ parity: ``DigitalCoupon::isLongPut`` (inline)."""
        return self._put_csi == 1.0

    def call_strike(self) -> float | None:
        """# C++ parity: ql/cashflows/digitalcoupon.cpp:260-265."""
        return self._call_strike if self._has_call_strike else None

    def put_strike(self) -> float | None:
        """# C++ parity: ql/cashflows/digitalcoupon.cpp:267-272."""
        return self._put_strike if self._has_put_strike else None

    def call_digital_payoff(self) -> float | None:
        """# C++ parity: ql/cashflows/digitalcoupon.cpp:274-279."""
        return self._call_digital_payoff if self._is_call_cash_or_nothing else None

    def put_digital_payoff(self) -> float | None:
        """# C++ parity: ql/cashflows/digitalcoupon.cpp:281-286."""
        return self._put_digital_payoff if self._is_put_cash_or_nothing else None

    # --- option-rate replication ---------------------------------------

    def call_option_rate(self) -> float:
        """Call digital option rate via call-spread replication.

        # C++ parity: ql/cashflows/digitalcoupon.cpp:179-198.
        """
        call_option_rate = 0.0
        if self._has_call_strike:
            # Step function value.
            call_option_rate = (
                self._call_digital_payoff
                if self._is_call_cash_or_nothing
                else self._call_strike
            )
            nxt = self._capped(self._call_strike + self._call_right_eps)
            prev = self._capped(self._call_strike - self._call_left_eps)
            call_option_rate *= (nxt.rate() - prev.rate()) / (
                self._call_left_eps + self._call_right_eps
            )
            if not self._is_call_cash_or_nothing:
                at_strike = self._capped(self._call_strike)
                call = self._underlying.rate() - at_strike.rate()
                call_option_rate += call
        return call_option_rate

    def put_option_rate(self) -> float:
        """Put digital option rate via put-spread replication.

        # C++ parity: ql/cashflows/digitalcoupon.cpp:200-219.
        """
        put_option_rate = 0.0
        if self._has_put_strike:
            put_option_rate = (
                self._put_digital_payoff
                if self._is_put_cash_or_nothing
                else self._put_strike
            )
            nxt = self._floored(self._put_strike + self._put_right_eps)
            prev = self._floored(self._put_strike - self._put_left_eps)
            put_option_rate *= (nxt.rate() - prev.rate()) / (
                self._put_left_eps + self._put_right_eps
            )
            if not self._is_put_cash_or_nothing:
                at_strike = self._floored(self._put_strike)
                put = -self._underlying.rate() + at_strike.rate()
                put_option_rate -= put
        return put_option_rate

    def _capped(self, cap: float) -> CappedFlooredCoupon:
        """A CappedFlooredCoupon(underlying, cap) sharing the underlying's pricer."""
        c = CappedFlooredCoupon(self._underlying, cap, None)
        c.set_pricer(self._underlying.pricer())
        return c

    def _floored(self, floor: float) -> CappedFlooredCoupon:
        """A CappedFlooredCoupon(underlying, None, floor) sharing the pricer."""
        c = CappedFlooredCoupon(self._underlying, None, floor)
        c.set_pricer(self._underlying.pricer())
        return c

    # --- payoff (only when index already fixed) ------------------------

    def _call_payoff(self) -> float:
        """# C++ parity: ql/cashflows/digitalcoupon.cpp:297-312.

        Pays when in-the-money (``L - X > eps``) or, if ATM is included, when
        at-the-money (``|X - L| <= eps``). The payoff value is identical in
        both C++ branches, so we fold them into one condition.
        """
        payoff = 0.0
        if self._has_call_strike:
            underlying_rate = self._underlying.rate()
            itm = (underlying_rate - self._call_strike) > _EPS
            atm = self._is_call_atm_included and abs(self._call_strike - underlying_rate) <= _EPS
            if itm or atm:
                payoff = (
                    self._call_digital_payoff
                    if self._is_call_cash_or_nothing
                    else underlying_rate
                )
        return payoff

    def _put_payoff(self) -> float:
        """# C++ parity: ql/cashflows/digitalcoupon.cpp:314-330.

        See :meth:`_call_payoff` — folded ITM / ATM branches (identical payoff).
        """
        payoff = 0.0
        if self._has_put_strike:
            underlying_rate = self._underlying.rate()
            itm = (self._put_strike - underlying_rate) > _EPS
            atm = self._is_put_atm_included and abs(self._put_strike - underlying_rate) <= _EPS
            if itm or atm:
                payoff = (
                    self._put_digital_payoff
                    if self._is_put_cash_or_nothing
                    else underlying_rate
                )
        return payoff

    # --- pricer wiring -------------------------------------------------

    def set_pricer(self, pricer: FloatingRateCouponPricer | None) -> None:
        """Attach the pricer to both this coupon and the underlying.

        # C++ parity: ql/cashflows/digitalcoupon.hpp:136-144.
        """
        super().set_pricer(pricer)
        self._underlying.set_pricer(pricer)

    # --- Coupon interface ----------------------------------------------

    def rate(self) -> float:
        """Digital-coupon rate.

        # C++ parity: ql/cashflows/digitalcoupon.cpp:226-254.

        If the index has already fixed (``fixingDate < today``), the
        deterministic payoff is used; otherwise the replicated option rate.
        """
        from pquantlib.patterns.observable_settings import ObservableSettings  # noqa: PLC0415

        underlying_pricer = self._underlying.pricer()
        qassert.require(underlying_pricer is not None, "pricer not set")
        fixing_date = self._underlying.fixing_date()
        today = ObservableSettings().evaluation_date_or_today()
        underlying_rate = 0.0 if self._naked_option else self._underlying.rate()
        if fixing_date < today:
            return underlying_rate + self._call_csi * self._call_payoff() + self._put_csi * self._put_payoff()
        if fixing_date == today:
            # might have been fixed — check the index history
            idx = self._underlying.index()
            has_fixing_fn = getattr(idx, "has_historical_fixing", None)
            if has_fixing_fn is not None and has_fixing_fn(fixing_date):
                return (
                    underlying_rate
                    + self._call_csi * self._call_payoff()
                    + self._put_csi * self._put_payoff()
                )
        return (
            underlying_rate
            + self._call_csi * self.call_option_rate()
            + self._put_csi * self.put_option_rate()
        )

    def convexity_adjustment(self) -> float:
        """# C++ parity: ql/cashflows/digitalcoupon.cpp:256-258."""
        return self._underlying.convexity_adjustment()


__all__ = ["DigitalCoupon"]
