"""Overnight-indexed coupon pricers — compounding + arithmetic average.

# C++ parity: ql/cashflows/overnightindexedcouponpricer.{hpp,cpp}
# (v1.42.1, 099987f0).

This is the full-fidelity home of the overnight-coupon pricer family that
prices an :class:`~pquantlib.cashflows.overnight_indexed_coupon.OvernightIndexedCoupon`:

- :class:`OvernightIndexedCouponPricer` — abstract base (the common
  ``initialize`` + interface; C++ also carries an optionlet-vol handle for
  capped/floored variants, see *divergences* below).
- :class:`CompoundingOvernightIndexedCouponPricer` — daily-**compounded**
  overnight rate (C++ ``compute()``):

      compound = prod_i (1 + fixing_i * dt_i)
      rate     = (compound - 1) / accrued_period
      swapletRate = gearing * rate + spread

- :class:`ArithmeticAveragedOvernightIndexedCouponPricer` — arithmetically
  **averaged** overnight rate (Katsumi Takada 2011):

      accumulated = sum_i fixing_i * dt_i
      rate        = accumulated / accrued_period
      swapletRate = gearing * rate + spread

Both split each fixing into a *past* part (fixing date strictly before the
global evaluation date — read from the index's stored history) and a
*forecast* part (fixing date on/after the evaluation date — projected via the
index's ``fixing()``). When the index carries no stored history at all (e.g.
a flat-forecast mock with no ``has_historical_fixing``), every fixing flows
through the forecast branch, so the result reduces to the pure-forecast
compounding/averaging used by the L2-D coupon test surface.

Past/forecast split uses :class:`~pquantlib.Settings`'s evaluation date when
available; if ``Settings`` is not initialised, the split degenerates to
"forecast everything" (identical to the prior L2-D behaviour).

Python divergences from C++:

- **Optionlet volatility / capped-floored variants.** The C++ base pricer
  carries a ``Handle<OptionletVolatilityStructure>`` and ``capletRate(Rate,
  bool)`` / ``floorletRate(Rate, bool)`` / ``averageRate(Date)`` virtuals
  used by ``CappedFlooredOvernightIndexedCoupon`` and the Black overnight
  pricers. Those coupon/pricer variants are a deferred carve-out in this
  port (``CappedFlooredOvernightIndexedCoupon`` is not ported), so the vol
  handle and caplet/floorlet machinery are omitted. ``averageRate(date)``
  *is* ported (it drives the arithmetic pricer's ``swapletRate``).
- **Telescopic forward-discount optimisation + lockout (rate cutoff) +
  observation shift + compound-spread-daily.** The C++ ``compute()`` has a
  fast path using ``curve->discount(valueDates[...])`` ratios plus lockout
  and observation-shift handling. The ported
  :class:`~pquantlib.cashflows.overnight_indexed_coupon.OvernightIndexedCoupon`
  does not expose ``lockoutDays`` / ``applyObservationShift`` /
  ``compoundSpreadDaily`` / ``canApplyTelescopicFormula`` (all fixed at
  0 / False), so the forward part here is the straightforward per-fixing
  projection loop (correct for the no-lockout, no-observation-shift,
  spread-not-compounded-daily case, which is the standard OIS coupon).
  Telescopic speed-up + lockout + observation shift are deferred carve-outs.
- ``effectiveSpread`` / ``effectiveIndexFixing`` (only meaningful with
  compound-spread-daily) collapse to ``spread`` / ``rate`` respectively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import CouponPricer
from pquantlib.exceptions import LibraryException
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon


def _evaluation_date() -> Date | None:
    """Global evaluation date, or ``None`` when it has never been pinned.

    # C++ parity: ``Settings::instance().evaluationDate()`` in
    # overnightindexedcouponpricer.cpp:109 / :283. PQuantLib's
    # :class:`~pquantlib.patterns.observable_settings.ObservableSettings`
    # singleton returns ``None`` from ``evaluation_date`` until pinned; an
    # unpinned date means "forecast everything" here, which reproduces the
    # L2-D coupon's pure-forecast behaviour for the flat-forecast mock.
    """
    from pquantlib.patterns.observable_settings import (  # noqa: PLC0415 (lazy)
        ObservableSettings,
    )

    d = ObservableSettings().evaluation_date
    if d is None or d == Date():
        return None
    return d


class OvernightIndexedCouponPricer(CouponPricer):
    """Abstract base pricer for overnight-indexed floating coupons.

    # C++ parity: ``OvernightIndexedCouponPricer`` in
    # overnightindexedcouponpricer.hpp:51-101.
    """

    def __init__(self) -> None:
        super().__init__()
        self._coupon: OvernightIndexedCoupon | None = None

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        """Bind the (overnight) coupon being priced.

        # C++ parity: overnightindexedcouponpricer.cpp:54-64 — also unwraps a
        # ``CappedFlooredOvernightIndexedCoupon``; that variant is not ported,
        # so we only accept a plain ``OvernightIndexedCoupon``.
        """
        from pquantlib.cashflows.overnight_indexed_coupon import (  # noqa: PLC0415
            OvernightIndexedCoupon,
        )

        qassert.require(
            isinstance(coupon, OvernightIndexedCoupon),
            "OvernightIndexedCouponPricer: unsupported coupon type",
        )
        assert isinstance(coupon, OvernightIndexedCoupon)
        self._coupon = coupon

    # --- shared helpers ------------------------------------------------

    def _coupon_required(self) -> OvernightIndexedCoupon:
        qassert.require(self._coupon is not None, "coupon not set")
        assert self._coupon is not None
        return self._coupon

    def _span(self, i: int, date: Date, full: bool) -> float:
        """Day-count fraction contributed by overnight interval ``i``.

        With ``full`` (the whole-coupon swaplet case) every interval
        contributes its full ``dt[i]``. Otherwise the interval *containing*
        ``date`` is truncated at ``date`` (C++ ``date >= interestDates[i+1] ?
        dt[i] : dc.yearFraction(interestDates[i], date)``).
        """
        coupon = self._coupon_required()
        dt = coupon.dt()
        if full:
            return dt[i]
        interest_dates = coupon.interest_dates()
        if date >= interest_dates[i + 1]:
            return dt[i]
        return coupon.overnight_index().day_counter().year_fraction(
            interest_dates[i], date
        )

    def _denominator(self, date: Date, full: bool) -> float:
        """Accrual denominator: full ``accrualPeriod`` or ``accruedPeriod(date)``.

        The ported coupon's terminal interest (value) date can fall after the
        unadjusted accrual end / payment date because value dates are snapped
        to business days; ``accruedPeriod`` past the payment date returns 0,
        so the whole-coupon case uses ``accrualPeriod()`` directly.
        """
        coupon = self._coupon_required()
        if full:
            return coupon.accrual_period()
        return coupon.accrued_period(date)

    @staticmethod
    def _fixing_value(index: object, fixing_date: Date, today: Date | None) -> float:
        """One overnight fixing, reading history before ``today`` else forecasting.

        Mirrors the per-fixing logic of C++ ``compute`` / ``averageRate``:
        a fixing strictly before ``today`` must come from stored history; a
        fixing on/after ``today`` (or when ``today`` is None) is forecast.
        ``today``-dated fixings prefer history when present, else forecast
        (the C++ "today is a border case" branch).
        """
        has_hist = getattr(index, "has_historical_fixing", None)
        if today is not None and has_hist is not None:
            if fixing_date < today:
                qassert.require(
                    has_hist(fixing_date),
                    f"Missing fixing for {fixing_date}",
                )
                return index.past_fixing(fixing_date)  # type: ignore[attr-defined]
            if fixing_date == today and has_hist(fixing_date):
                return index.past_fixing(fixing_date)  # type: ignore[attr-defined]
        # forecast
        return index.fixing(fixing_date, False)  # type: ignore[attr-defined]


class CompoundingOvernightIndexedCouponPricer(OvernightIndexedCouponPricer):
    """Daily-compounded overnight rate pricer.

    # C++ parity: ``CompoundingOvernightIndexedCouponPricer`` in
    # overnightindexedcouponpricer.{hpp,cpp}.
    """

    def __init__(self) -> None:
        super().__init__()
        self._swaplet_rate: float = 0.0
        self._effective_spread: float = 0.0
        self._effective_index_fixing: float = 0.0

    def _compute(self, date: Date, full: bool) -> tuple[float, float, float]:
        """Return ``(swapletRate, effectiveSpread, effectiveIndexFixing)`` at ``date``.

        # C++ parity: overnightindexedcouponpricer.cpp:108-259, restricted to
        # the no-lockout / no-observation-shift / spread-not-daily case
        # (the only configuration the ported coupon exposes). ``full`` selects
        # the whole-coupon swaplet (all spans = dt[i], denom = accrualPeriod).
        """
        coupon = self._coupon_required()
        index = coupon.overnight_index()
        today = _evaluation_date()

        fixing_dates = coupon.fixing_dates()
        n = coupon.n()

        compound = 1.0
        for i in range(n):
            span = self._span(i, date, full)
            fixing = self._fixing_value(index, fixing_dates[i], today)
            compound *= 1.0 + fixing * span

        rate = (compound - 1.0) / self._denominator(date, full)
        swaplet_rate = coupon.gearing() * rate + coupon.spread()
        # spread not compounded daily ⇒ effectiveSpread == spread,
        # effectiveIndexFixing == rate (C++ cpp:249-252).
        return swaplet_rate, coupon.spread(), rate

    def swaplet_rate(self) -> float:
        coupon = self._coupon_required()
        sr, es, ef = self._compute(coupon.accrual_end_date(), full=True)
        self._swaplet_rate = sr
        self._effective_spread = es
        self._effective_index_fixing = ef
        return sr

    def average_rate(self, date: Date) -> float:
        """Compounded rate up to ``date``. C++ parity: cpp:91-94."""
        sr, _es, _ef = self._compute(date, full=False)
        return sr

    def effective_spread(self) -> float:
        """C++ parity: cpp:96-100."""
        coupon = self._coupon_required()
        _sr, es, _ef = self._compute(coupon.accrual_end_date(), full=True)
        self._effective_spread = es
        return es

    def effective_index_fixing(self) -> float:
        """C++ parity: cpp:102-106."""
        coupon = self._coupon_required()
        _sr, _es, ef = self._compute(coupon.accrual_end_date(), full=True)
        self._effective_index_fixing = ef
        return ef

    # cap/floor not available on the compounding pricer (C++ QL_FAIL).
    def swaplet_price(self) -> float:
        msg = "swapletPrice not available"
        raise LibraryException(msg)

    def caplet_price(self, effective_cap: float) -> float:
        del effective_cap
        msg = "capletPrice not available"
        raise LibraryException(msg)

    def caplet_rate(self, effective_cap: float) -> float:
        del effective_cap
        msg = "capletRate not available"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorletPrice not available"
        raise LibraryException(msg)

    def floorlet_rate(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorletRate not available"
        raise LibraryException(msg)


class ArithmeticAveragedOvernightIndexedCouponPricer(OvernightIndexedCouponPricer):
    """Arithmetically-averaged overnight rate pricer.

    # C++ parity: ``ArithmeticAveragedOvernightIndexedCouponPricer`` in
    # overnightindexedcouponpricer.{hpp,cpp}. Reference: Katsumi Takada 2011,
    # "Valuation of Arithmetically Average of Fed Funds Rates and Construction
    # of the US Dollar Swap Yield Curve".

    The Hull-White convexity correction (``byApprox`` / ``convAdj1`` /
    ``convAdj2``) applies only to the *forward* part with a non-zero
    volatility. With the default ``volatility = 0`` there is no convexity
    adjustment (C++ default), so it is omitted from the forward loop; the
    convexity-adjusted forward path is a deferred carve-out (it requires a
    forwarding term structure not exercised by the deterministic test
    surface).
    """

    def __init__(
        self,
        mean_reversion: float = 0.03,
        volatility: float = 0.0,
        by_approx: bool = False,
    ) -> None:
        super().__init__()
        self._mrs: float = mean_reversion
        self._vol: float = volatility
        self._by_approx: bool = by_approx

    def _average_rate(self, date: Date, full: bool) -> float:
        coupon = self._coupon_required()
        index = coupon.overnight_index()
        today = _evaluation_date()

        fixing_dates = coupon.fixing_dates()
        n = coupon.n()

        accumulated = 0.0
        for i in range(n):
            span = self._span(i, date, full)
            fixing = self._fixing_value(index, fixing_dates[i], today)
            accumulated += fixing * span

        rate = accumulated / self._denominator(date, full)
        return coupon.gearing() * rate + coupon.spread()

    def average_rate(self, date: Date) -> float:
        """Arithmetic-average rate up to ``date``.

        # C++ parity: overnightindexedcouponpricer.cpp:265-357, restricted to
        # the no-convexity-adjustment forward path (vol == 0 default).
        """
        return self._average_rate(date, full=False)

    def swaplet_rate(self) -> float:
        coupon = self._coupon_required()
        return self._average_rate(coupon.accrual_end_date(), full=True)

    def swaplet_price(self) -> float:
        msg = "swapletPrice not available"
        raise LibraryException(msg)

    def caplet_price(self, effective_cap: float) -> float:
        del effective_cap
        msg = "capletPrice not available"
        raise LibraryException(msg)

    def caplet_rate(self, effective_cap: float) -> float:
        del effective_cap
        msg = "capletRate not available"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorletPrice not available"
        raise LibraryException(msg)

    def floorlet_rate(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorletRate not available"
        raise LibraryException(msg)
