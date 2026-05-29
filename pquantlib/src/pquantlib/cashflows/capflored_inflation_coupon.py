"""CappedFlooredInflationCoupon family — capped/floored YoY inflation coupons.

# C++ parity: ql/cashflows/capflooredinflationcoupon.{hpp,cpp} (v1.42.1).

The C++ header ships exactly one class — ``CappedFlooredYoYInflationCoupon``
— which extends ``YoYInflationCoupon`` and overrides ``rate()`` to wrap
the swaplet rate with cap/floor adjustments computed by the pricer.

The L7-C spec asks for three names:

* ``CappedFlooredInflationCoupon`` — a *generic* abstract wrapper to
  match the spec's surface. We expose it as a Python alias of the YoY
  variant since C++ ships only YoY (Phase 7 carve-out: no CPI capfloor
  coupon ships in C++ v1.42.1 either — only ``CPICapFloor`` *instruments*
  at L7-D).
* ``CappedFlooredCPICoupon`` — same alias. Surfaced for downstream
  L7-D readability; the L7-D ``CPICapFloor`` instrument will pin down
  the actual CPI cap/floor semantics.
* ``CappedFlooredYoYInflationCoupon`` — the C++ class proper.

# C++ parity divergence: the ``CappedFlooredCPICoupon`` and
# ``CappedFlooredInflationCoupon`` aliases are PYTHON-ONLY. They reduce
# downstream import friction without changing semantics. Documented here
# so future L7-D + L7-E readers know to expect them.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class CappedFlooredYoYInflationCoupon(YoYInflationCoupon):
    """Capped/floored YoY inflation coupon.

    # C++ parity: ``CappedFlooredYoYInflationCoupon`` in
    # capflooredinflationcoupon.hpp:66-138.

    The C++ class supports two construction modes:

    * **Underlying-watching mode** (``CappedFlooredYoYInflationCoupon(underlying,
      cap, floor)``): the coupon wraps a constructed ``YoYInflationCoupon``
      and forwards pricer attachments to it. We expose this via the
      ``from_underlying`` classmethod.
    * **Standalone mode** (the full-fields constructor below): constructs
      the YoY coupon and applies cap/floor on top.

    The cap/floor application logic follows C++
    ``CappedFlooredYoYInflationCoupon::setCommon`` (capflooredinflationcoupon.cpp:25-54):

    * Positive gearing: ``cap`` caps, ``floor`` floors.
    * Negative gearing: roles swap — the C++ comment is "the negative
      gearing flips the inequality direction".
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        fixing_days: int,
        index: YoYInflationIndex,
        observation_lag: Period,
        interpolation: InterpolationType,
        day_counter: DayCounter,
        gearing: float = 1.0,
        spread: float = 0.0,
        cap: float | None = None,
        floor: float | None = None,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
    ) -> None:
        # C++ parity: ql/cashflows/capflooredinflationcoupon.hpp:75-96 —
        # standalone constructor (no underlying coupon).
        super().__init__(
            payment_date=payment_date,
            nominal=nominal,
            accrual_start_date=accrual_start_date,
            accrual_end_date=accrual_end_date,
            fixing_days=fixing_days,
            index=index,
            observation_lag=observation_lag,
            interpolation=interpolation,
            day_counter=day_counter,
            gearing=gearing,
            spread=spread,
            ref_period_start=ref_period_start,
            ref_period_end=ref_period_end,
        )
        self._underlying: YoYInflationCoupon | None = None
        self._is_floored: bool = False
        self._is_capped: bool = False
        # Storage slots match C++ — populated by ``_set_common``.
        self._cap: float = 0.0
        self._floor: float = 0.0
        self._set_common(cap, floor)

    # ---- alternate constructor ---------------------------------------

    @classmethod
    def from_underlying(
        cls,
        underlying: YoYInflationCoupon,
        cap: float | None = None,
        floor: float | None = None,
    ) -> CappedFlooredYoYInflationCoupon:
        """Wrap an existing ``YoYInflationCoupon`` with cap / floor.

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:57-76.

        The underlying's pricer (if any) is propagated through
        ``set_pricer``.
        """
        instance = cls(
            payment_date=underlying.date(),
            nominal=underlying.nominal(),
            accrual_start_date=underlying.accrual_start_date(),
            accrual_end_date=underlying.accrual_end_date(),
            fixing_days=underlying.fixing_days(),
            index=underlying.yoy_index(),
            observation_lag=underlying.observation_lag(),
            interpolation=underlying.interpolation(),
            day_counter=underlying.day_counter(),
            gearing=underlying.gearing(),
            spread=underlying.spread(),
            cap=cap,
            floor=floor,
            ref_period_start=underlying.reference_period_start(),
            ref_period_end=underlying.reference_period_end(),
        )
        instance._underlying = underlying
        return instance

    # ---- inspectors --------------------------------------------------

    def is_capped(self) -> bool:
        """C++ parity: ql/cashflows/capflooredinflationcoupon.hpp:125 (inline)."""
        return self._is_capped

    def is_floored(self) -> bool:
        """C++ parity: ql/cashflows/capflooredinflationcoupon.hpp:126 (inline)."""
        return self._is_floored

    def cap(self) -> float | None:
        """User-visible cap (sign-aware).

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:108-114.

        When gearing is negative, the cap is actually stored in the
        ``floor_`` slot (C++ ``setCommon`` swap). This accessor reverses
        the swap so callers see the original cap they passed at
        construction.
        """
        if self._gearing > 0 and self._is_capped:
            return self._cap
        if self._gearing < 0 and self._is_floored:
            return self._floor
        return None

    def floor(self) -> float | None:
        """User-visible floor (sign-aware). See ``cap`` docstring.

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:117-123.
        """
        if self._gearing > 0 and self._is_floored:
            return self._floor
        if self._gearing < 0 and self._is_capped:
            return self._cap
        return None

    def effective_cap(self) -> float:
        """``(cap - spread) / gearing``.

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:126-128.
        """
        return (self._cap - self._spread) / self._gearing

    def effective_floor(self) -> float:
        """``(floor - spread) / gearing``.

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:131-133.
        """
        return (self._floor - self._spread) / self._gearing

    def underlying_rate(self) -> float:
        """Pre-cap/floor expected rate.

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:88-90.
        """
        if self._underlying is not None:
            return self._underlying.rate()
        # Skip the cap/floor wrap on the YoY-base rate.
        return YoYInflationCoupon.rate(self)

    # ---- rate override -----------------------------------------------

    def rate(self) -> float:
        """``swapletRate + floorletRate - capletRate``.

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:92-105.
        """
        swaplet = self.underlying_rate()
        coupon_pricer = (
            self._underlying.pricer() if self._underlying is not None else self._pricer
        )
        if self._is_floored or self._is_capped:
            qassert.require(coupon_pricer is not None, "pricer not set")
        from pquantlib.cashflows.yoy_inflation_coupon_pricer import (  # noqa: PLC0415
            YoYInflationCouponPricer,
        )

        assert coupon_pricer is None or isinstance(coupon_pricer, YoYInflationCouponPricer)
        floor_rate = (
            coupon_pricer.floorlet_rate(self.effective_floor())
            if self._is_floored and coupon_pricer is not None
            else 0.0
        )
        cap_rate = (
            coupon_pricer.caplet_rate(self.effective_cap())
            if self._is_capped and coupon_pricer is not None
            else 0.0
        )
        return swaplet + floor_rate - cap_rate

    # ---- pricer wiring -----------------------------------------------

    def set_pricer(self, pricer: object) -> None:
        """Propagate pricer to underlying (if any).

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:79-85.

        Accepts ``object`` here (rather than the typed pricer) because
        the runtime ``check_pricer_impl`` predicate handles the type
        guard. This mirrors the C++ shape where ``setPricer`` takes a
        ``shared_ptr<YoYInflationCouponPricer>`` but dispatches through
        ``InflationCoupon::setPricer`` for the registration handshake.
        """
        # Local import for the coupon ↔ pricer cycle.
        from pquantlib.cashflows.inflation_coupon_pricer import (  # noqa: PLC0415
            InflationCouponPricer,
        )
        from pquantlib.cashflows.yoy_inflation_coupon_pricer import (  # noqa: PLC0415
            YoYInflationCouponPricer,
        )

        qassert.require(
            pricer is None or isinstance(pricer, YoYInflationCouponPricer),
            "CappedFlooredYoYInflationCoupon requires a YoYInflationCouponPricer",
        )
        assert pricer is None or isinstance(pricer, InflationCouponPricer)
        # Use base-class setter (validates via check_pricer_impl).
        YoYInflationCoupon.set_pricer(self, pricer)
        if self._underlying is not None:
            self._underlying.set_pricer(pricer)

    # ---- internal -----------------------------------------------------

    def _set_common(self, cap: float | None, floor: float | None) -> None:
        """Apply the C++ sign-aware cap/floor storage logic.

        # C++ parity: ql/cashflows/capflooredinflationcoupon.cpp:25-54.
        """
        self._is_floored = False
        self._is_capped = False
        if self._gearing > 0:
            if cap is not None:
                self._is_capped = True
                self._cap = cap
            if floor is not None:
                self._floor = floor
                self._is_floored = True
        else:
            # Negative gearing flips the inequality direction.
            if cap is not None:
                self._floor = cap
                self._is_floored = True
            if floor is not None:
                self._is_capped = True
                self._cap = floor
        if self._is_capped and self._is_floored:
            assert cap is not None
            assert floor is not None
            qassert.require(
                cap >= floor,
                f"cap level ({cap}) less than floor level ({floor})",
            )


# Python-side aliases. See module docstring for the rationale.
CappedFlooredInflationCoupon = CappedFlooredYoYInflationCoupon
CappedFlooredCPICoupon = CappedFlooredYoYInflationCoupon
