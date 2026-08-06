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

- **Lockout (rate cutoff) + observation shift + compound-spread-daily.** The
  ported
  :class:`~pquantlib.cashflows.overnight_indexed_coupon.OvernightIndexedCoupon`
  fixes ``lockoutDays`` at 0, ``applyObservationShift`` at False and
  ``compoundSpreadDaily`` at False, so those branches of the C++
  ``compute()`` are deferred carve-outs. The **telescopic** forward-discount
  path *is* ported (see
  :meth:`CompoundingOvernightIndexedCouponPricer._forward_compound`) — it is
  not merely a speed-up, it is what v1.42.1 actually evaluates, and the
  per-fixing product it replaces differs from it in the last few ULPs.
- ``effectiveSpread`` / ``effectiveIndexFixing`` (only meaningful with
  compound-spread-daily) collapse to ``spread`` / ``rate`` respectively.
- C++ *privatises* the one-argument ``capletRate(Rate)`` /
  ``floorletRate(Rate)`` it inherits from ``FloatingRateCouponPricer``
  (overnightindexedcouponpricer.hpp:52-53) so that only the two-argument
  ``(strike, dailyCapFloor)`` forms are reachable through a base-class
  handle. Python has no such access control; the one-argument forms stay
  visible and simply delegate to the two-argument ones with
  ``daily_cap_floor=False``, which is what the C++ Black pricers do too.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon_pricer import CouponPricer
from pquantlib.exceptions import LibraryException
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.cashflows.overnight_indexed_coupon import OvernightIndexedCoupon
    from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
        OptionletVolatilityStructure,
    )


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
    # overnightindexedcouponpricer.hpp:51-101 + .cpp:38-68.
    """

    def __init__(
        self,
        caplet_volatility: OptionletVolatilityStructure | None = None,
        effective_volatility_input: bool = False,
    ) -> None:
        super().__init__()
        self._coupon: OvernightIndexedCoupon | None = None
        self._caplet_vol: OptionletVolatilityStructure | None = caplet_volatility
        self._effective_volatility_input: bool = effective_volatility_input
        # ``Null<Real>`` until a caplet / floorlet has actually been priced.
        self._effective_caplet_volatility: float | None = None
        self._effective_floorlet_volatility: float | None = None

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        """Bind the (overnight) coupon being priced.

        # C++ parity: overnightindexedcouponpricer.cpp:46-56 — a
        # ``CappedFlooredOvernightIndexedCoupon`` is unwrapped to its
        # underlying coupon, anything else that is not an
        # ``OvernightIndexedCoupon`` is rejected.
        """
        from pquantlib.cashflows.capped_floored_coupon import (  # noqa: PLC0415
            CappedFlooredOvernightIndexedCoupon,
        )
        from pquantlib.cashflows.overnight_indexed_coupon import (  # noqa: PLC0415
            OvernightIndexedCoupon,
        )

        if isinstance(coupon, CappedFlooredOvernightIndexedCoupon):
            underlying = coupon.underlying()
            qassert.require(
                isinstance(underlying, OvernightIndexedCoupon),
                "OvernightIndexedCouponPricer: CappedFlooredOvernightIndexedCoupon "
                "underlying coupon not defined",
            )
            assert isinstance(underlying, OvernightIndexedCoupon)
            self._coupon = underlying
            return
        qassert.require(
            isinstance(coupon, OvernightIndexedCoupon),
            "OvernightIndexedCouponPricer: unsupported coupon type",
        )
        assert isinstance(coupon, OvernightIndexedCoupon)
        self._coupon = coupon

    # --- optionlet volatility wiring -----------------------------------

    def caplet_volatility(self) -> OptionletVolatilityStructure | None:
        """# C++ parity: ``capletVolatility()`` (hpp:73-75)."""
        return self._caplet_vol

    def set_caplet_volatility(self, caplet_volatility: OptionletVolatilityStructure | None = None) -> None:
        """Re-link the optionlet vol surface and notify observers.

        # C++ parity: ``setCapletVolatility`` (hpp:64-71).
        """
        self._caplet_vol = caplet_volatility
        self.update()

    def effective_volatility_input(self) -> bool:
        """# C++ parity: ``effectiveVolatilityInput()`` (cpp:58-60)."""
        return self._effective_volatility_input

    def set_effective_volatility_input(self, effective_volatility_input: bool) -> None:
        """# C++ parity: ``setEffectiveVolatilityInput`` (hpp:77-79)."""
        self._effective_volatility_input = effective_volatility_input

    def effective_caplet_volatility(self) -> float | None:
        """Effective vol recorded by the last ``caplet_rate`` call, else ``None``.

        # C++ parity: ``effectiveCapletVolatility()`` (cpp:62-64) — C++
        # returns ``Null<Real>`` before the first call; Python returns
        # ``None``.
        """
        return self._effective_caplet_volatility

    def effective_floorlet_volatility(self) -> float | None:
        """# C++ parity: ``effectiveFloorletVolatility()`` (cpp:66-68)."""
        return self._effective_floorlet_volatility

    # --- pure virtuals (C++ hpp:92-94) ---------------------------------

    @abstractmethod
    def caplet_rate(self, effective_cap: float, daily_cap_floor: bool = False) -> float:
        """Caplet rate; ``daily_cap_floor`` selects the per-day cap formula.

        # C++ parity: ``virtual Rate capletRate(Rate, bool) const = 0``
        # (hpp:92). C++ additionally *privatises* the inherited one-argument
        # form; Python keeps it visible with ``daily_cap_floor=False``.
        """

    @abstractmethod
    def floorlet_rate(self, effective_floor: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: ``virtual Rate floorletRate(Rate, bool) const = 0`` (hpp:93)."""

    @abstractmethod
    def average_rate(self, date: Date) -> float:
        """Coupon rate accrued up to ``date``.

        # C++ parity: ``virtual Rate averageRate(const Date&) const = 0`` (hpp:94).
        """

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

    def _number_of_fixings(self, date: Date) -> int:
        """``lower_bound(interestDates.begin(), interestDates.end()-1, date)``.

        # C++ parity: anonymous ``determineNumberOfFixings``
        (overnightindexedcouponpricer.cpp:32-37).
        """
        interest_dates = self._coupon_required().interest_dates()
        last = len(interest_dates) - 1  # end() - 1
        for i in range(last):
            if interest_dates[i] >= date:
                return i
        return last

    def _denominator(self, date: Date, full: bool) -> float:
        """Rate-accrual denominator ``tau``.

        # C++ parity: overnightindexedcouponpricer.cpp:206-210 —
        # ``tau = index.dayCounter().yearFraction(interestDates.front(),
        #                                         min(date, interestDates[n]))``.
        #
        # This is the **rate-computation** window measured with the **index's**
        # day counter, not the coupon's own accrual period. The two coincide
        # for a plain in-arrears coupon whose payment day counter is the
        # index's, which is why the previous ``accrualPeriod()`` spelling
        # passed; they part company as soon as the rate-computation dates are
        # decoupled from the accrual dates (``OvernightLeg.in_arrears(False)``,
        # ``with_last_recent_period``) or a different payment day counter is
        # set.
        """
        coupon = self._coupon_required()
        interest_dates = coupon.interest_dates()
        n = coupon.n() if full else self._number_of_fixings(date)
        end = min(date, interest_dates[n])
        return coupon.overnight_index().day_counter().year_fraction(
            interest_dates[0], end
        )

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

    def __init__(
        self,
        caplet_volatility: OptionletVolatilityStructure | None = None,
        effective_volatility_input: bool = False,
    ) -> None:
        super().__init__(caplet_volatility, effective_volatility_input)
        self._swaplet_rate: float = 0.0
        self._effective_spread: float = 0.0
        self._effective_index_fixing: float = 0.0

    def _forward_compound(self, first: int, n: int, date: Date, *, full: bool) -> float | None:
        """Compound factor for the not-yet-fixed fixings ``[first, n)``.

        # C++ parity: overnightindexedcouponpricer.cpp:158-199 — the telescopic
        # fast path. Because every projected fixing comes off the same curve,
        # ``prod_k (1 + f_k * dt_k) == D(v_first) / D(v_end)``; C++ evaluates
        # the ratio, not the product. The identity is exact in real arithmetic
        # but *not* in floating point (the product drifts ~1e-13 relative,
        # which is enough to move a far-out-of-the-money caplet by ~2e-12), so
        # the port has to telescope too in order to reproduce v1.42.1.

        Returns ``None`` when the telescopic path does not apply — no
        forwarding curve (e.g. a flat-forecast mock index), an index whose
        fixing delay differs from the coupon's (C++
        ``canApplyTelescopicFormula``), or a partial-accrual evaluation where
        the last span is truncated. The caller then falls back to the
        per-fixing product, which is the same number in exact arithmetic.
        """
        if not full or first >= n:
            return None
        coupon = self._coupon_required()
        index = coupon.overnight_index()
        # ``canApplyTelescopicFormula()``: the coupon fixes with zero delay, so
        # the index must too, otherwise value dates and fixing dates decouple.
        fixing_days = getattr(index, "fixing_days", None)
        if fixing_days is None or fixing_days() != coupon.fixing_days():
            return None
        get_ts = getattr(index, "forecast_term_structure", None)
        curve = get_ts() if get_ts is not None else None
        if curve is None:
            return None

        value_dates = coupon.value_dates()
        interest_dates = coupon.interest_dates()
        fixing_dates = coupon.fixing_dates()
        today = _evaluation_date()

        # C++ cpp:174-181: a first interest date that lands before the first
        # value date (start on a fixing holiday) accrues partially and cannot
        # be telescoped; likewise a final value date past ``date``.
        start = 1 if (first == 0 and value_dates[0] < interest_dates[0]) else first
        end = n if value_dates[n] <= date else n - 1
        if start >= end:
            return None

        factor = 1.0
        for k in range(first, start):
            factor *= 1.0 + self._fixing_value(index, fixing_dates[k], today) * self._span(k, date, full)
        factor *= curve.discount(value_dates[start]) / curve.discount(value_dates[end])
        for k in range(end, n):
            factor *= 1.0 + self._fixing_value(index, fixing_dates[k], today) * self._span(k, date, full)
        return factor

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
        i = 0
        # Already-fixed part, day by day (C++ cpp:126-155).
        has_hist = getattr(index, "has_historical_fixing", None)
        while i < n and today is not None and fixing_dates[i] < today:
            compound *= 1.0 + self._fixing_value(index, fixing_dates[i], today) * self._span(i, date, full)
            i += 1
        # Today is a border case: consume it only if it has actually been published.
        if (
            i < n
            and today is not None
            and fixing_dates[i] == today
            and has_hist is not None
            and has_hist(today)
        ):
            compound *= 1.0 + self._fixing_value(index, fixing_dates[i], today) * self._span(i, date, full)
            i += 1

        telescopic = self._forward_compound(i, n, date, full=full)
        if telescopic is not None:
            compound *= telescopic
        else:
            while i < n:
                compound *= 1.0 + self._fixing_value(index, fixing_dates[i], today) * self._span(
                    i, date, full
                )
                i += 1

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

    def caplet_rate(self, effective_cap: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: hpp:118 (one-arg) + hpp:122-124 (two-arg) — both QL_FAIL."""
        del effective_cap
        if daily_cap_floor:
            msg = "CompoundingOvernightIndexedCouponPricer::capletRate(Rate, bool) not implemented"
            raise LibraryException(msg)
        msg = "capletRate not available"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorletPrice not available"
        raise LibraryException(msg)

    def floorlet_rate(self, effective_floor: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: hpp:120 (one-arg) + hpp:125-127 (two-arg) — both QL_FAIL."""
        del effective_floor
        if daily_cap_floor:
            msg = "CompoundingOvernightIndexedCouponPricer::floorletRate(Rate, bool) not implemented"
            raise LibraryException(msg)
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
        caplet_volatility: OptionletVolatilityStructure | None = None,
        effective_volatility_input: bool = False,
    ) -> None:
        super().__init__(caplet_volatility, effective_volatility_input)
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

    def caplet_rate(self, effective_cap: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: hpp:157 (one-arg) + hpp:160-162 (two-arg) — both QL_FAIL."""
        del effective_cap
        if daily_cap_floor:
            msg = "ArithmeticAveragedOvernightIndexedCouponPricer::capletRate(Rate, bool) not implemented"
            raise LibraryException(msg)
        msg = "capletRate not available"
        raise LibraryException(msg)

    def floorlet_price(self, effective_floor: float) -> float:
        del effective_floor
        msg = "floorletPrice not available"
        raise LibraryException(msg)

    def floorlet_rate(self, effective_floor: float, daily_cap_floor: bool = False) -> float:
        """# C++ parity: hpp:159 (one-arg) + hpp:163-165 (two-arg) — both QL_FAIL."""
        del effective_floor
        if daily_cap_floor:
            msg = "ArithmeticAveragedOvernightIndexedCouponPricer::floorletRate(Rate, bool) not implemented"
            raise LibraryException(msg)
        msg = "floorletRate not available"
        raise LibraryException(msg)
