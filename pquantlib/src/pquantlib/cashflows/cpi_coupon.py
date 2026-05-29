"""CPICoupon + CPICashFlow — coupons / cashflows paying a CPI ratio.

# C++ parity: ql/cashflows/cpicoupon.{hpp,cpp} (v1.42.1).

The C++ ``CPICoupon`` provides three overloaded constructors:

1. ``CPICoupon(baseCPI, ...)`` — explicit base CPI (no base date).
2. ``CPICoupon(baseDate, ...)`` — base CPI looked up via the index history
   at ``baseDate``.
3. ``CPICoupon(baseCPI, baseDate, ...)`` — both supplied; the explicit
   ``baseCPI`` wins.

The Python port collapses these to a single constructor that accepts
``base_cpi`` and/or ``base_date`` (both optional, at least one must be
non-null). The dispatch logic + null-check mirrors the C++
``CPICoupon::CPICoupon(Real baseCPI, const Date& baseDate, ...)``
forwarded path.

``CPICashFlow`` ports analogously — it's a single cash flow that pays
``notional * (CPI(observation_date) / baseFixing)`` (optionally minus 1
for swap-style).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.indexed_cashflow import IndexedCashFlow
from pquantlib.cashflows.inflation_coupon import InflationCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType, lagged_fixing
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.cashflows.inflation_coupon_pricer import InflationCouponPricer

# Module-level null Date for defaults (avoids B008).
_NULL_DATE: Date = Date()

# Numerical guard mirroring the C++ ``|baseCPI_| < 1e-16`` check.
_CPI_EPSILON: float = 1e-16


class CPICoupon(InflationCoupon):
    """Coupon paying ``nominal * fixedRate * (I_end / I_base) * accrualPeriod``.

    # C++ parity: ``CPICoupon`` in cpicoupon.hpp. The rate is set by a
    # ``CPICouponPricer`` to ``fixedRate * indexRatio(accrualEndDate)``,
    # so the amount works out to the formula in the docstring.
    """

    def __init__(
        self,
        payment_date: Date,
        nominal: float,
        accrual_start_date: Date,
        accrual_end_date: Date,
        index: ZeroInflationIndex,
        observation_lag: Period,
        observation_interpolation: InterpolationType,
        day_counter: DayCounter,
        fixed_rate: float,
        base_cpi: float | None = None,
        base_date: Date | None = None,
        ref_period_start: Date | None = None,
        ref_period_end: Date | None = None,
        ex_coupon_date: Date | None = None,
    ) -> None:
        # C++ parity: ql/cashflows/cpicoupon.cpp:80-91 — the unified
        # constructor takes both baseCPI and baseDate (either may be null,
        # but not both). fixing_days is hard-coded to 0 by C++ at
        # cpicoupon.cpp:81.
        super().__init__(
            payment_date=payment_date,
            nominal=nominal,
            accrual_start_date=accrual_start_date,
            accrual_end_date=accrual_end_date,
            fixing_days=0,
            index=index,
            observation_lag=observation_lag,
            day_counter=day_counter,
            ref_period_start=ref_period_start,
            ref_period_end=ref_period_end,
            ex_coupon_date=ex_coupon_date,
        )
        qassert.require(
            base_cpi is not None or (base_date is not None and base_date != _NULL_DATE),
            "baseCPI and baseDate can not be both null, "
            "provide a valid baseCPI or baseDate",
        )
        if base_cpi is not None:
            qassert.require(
                abs(base_cpi) > _CPI_EPSILON,
                "|baseCPI_| < 1e-16, future divide-by-zero problem",
            )
        self._base_cpi: float | None = base_cpi
        self._fixed_rate: float = fixed_rate
        self._observation_interpolation: InterpolationType = observation_interpolation
        self._base_date: Date = base_date if base_date is not None else _NULL_DATE
        self._cpi_index: ZeroInflationIndex = index

    # ---- inspectors --------------------------------------------------

    def fixed_rate(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.hpp:259-261 (inline)."""
        return self._fixed_rate

    def base_cpi(self) -> float | None:
        """C++ parity: ql/cashflows/cpicoupon.hpp:271-273 (inline).

        Returns ``None`` (== C++ ``Null<Rate>()``) if only a base date
        was supplied at construction.
        """
        return self._base_cpi

    def base_date(self) -> Date:
        """C++ parity: ql/cashflows/cpicoupon.hpp:275-277 (inline)."""
        return self._base_date

    def observation_interpolation(self) -> InterpolationType:
        """C++ parity: ql/cashflows/cpicoupon.hpp:279-281 (inline)."""
        return self._observation_interpolation

    def cpi_index(self) -> ZeroInflationIndex:
        """C++ parity: ql/cashflows/cpicoupon.hpp:283-285 (inline)."""
        return self._cpi_index

    # ---- InflationCoupon overrides -----------------------------------

    def index_fixing(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.hpp:267-269 (inline).

        Returns the lagged fixing observed at the accrual end date.
        """
        return lagged_fixing(
            self._cpi_index,
            self._accrual_end_date,
            self._observation_lag,
            self._observation_interpolation,
        )

    def check_pricer_impl(self, pricer: InflationCouponPricer) -> bool:
        """C++ parity: ql/cashflows/cpicoupon.cpp:131-135.

        Accepts only ``CPICouponPricer`` subtypes.
        """
        # Local import to avoid the coupon ↔ pricer cycle.
        from pquantlib.cashflows.cpi_coupon_pricer import CPICouponPricer  # noqa: PLC0415

        return isinstance(pricer, CPICouponPricer)

    def accrued_amount(self, d: Date) -> float:
        """C++ parity: ql/cashflows/cpicoupon.cpp:101-110.

        Uses the pricer's ``accrued_rate(d)`` rather than the abstract
        ``rate()`` (which would discount through the whole accrual
        period). For a CPI coupon, the rate-at-d ≠ rate-at-end except in
        the trivial growth-period-ends-today case.
        """
        if d <= self._accrual_start_date or d > self._payment_date:
            return 0.0
        # Local import for the cycle.
        from pquantlib.cashflows.cpi_coupon_pricer import CPICouponPricer  # noqa: PLC0415

        pricer = self._pricer
        qassert.require(pricer is not None, "pricer not set or of wrong type")
        assert pricer is not None
        qassert.require(
            isinstance(pricer, CPICouponPricer),
            "pricer not set or of wrong type",
        )
        assert isinstance(pricer, CPICouponPricer)
        pricer.initialize(self)
        return self._nominal * pricer.accrued_rate(d) * self.accrued_period(d)

    # ---- CPI-specific accessors --------------------------------------

    def index_ratio(self, d: Date) -> float:
        """``laggedFixing(d) / baseFixing``.

        # C++ parity: ql/cashflows/cpicoupon.cpp:112-129.

        If a base CPI was supplied, divide by it. Otherwise look up the
        base fixing at ``base_date + observation_lag`` with the same
        interpolation.
        """
        i0 = self._base_cpi
        if i0 is None:
            i0 = lagged_fixing(
                self._cpi_index,
                self._base_date + self._observation_lag,
                self._observation_lag,
                self._observation_interpolation,
            )
        i1 = lagged_fixing(
            self._cpi_index,
            d,
            self._observation_lag,
            self._observation_interpolation,
        )
        return i1 / i0

    def adjusted_index_growth(self) -> float:
        """``rate / fixedRate`` — index growth after pricer adjustments.

        # C++ parity: ql/cashflows/cpicoupon.hpp:263-265 (inline).
        """
        return self.rate() / self._fixed_rate


class CPICashFlow(IndexedCashFlow):
    """Single CPI cash flow — ``notional * I(obs) / I(base)`` (or growth-only).

    # C++ parity: ``CPICashFlow`` in cpicoupon.hpp:164-200. Unlike
    # ``ZeroInflationCashFlow``, the base date and base fixing are taken
    # *separately* (so the base fixing can be a quoted/explicit value
    # rather than read from the index history).
    """

    def __init__(
        self,
        notional: float,
        index: ZeroInflationIndex,
        base_date: Date,
        base_fixing: float | None,
        observation_date: Date,
        observation_lag: Period,
        interpolation: InterpolationType,
        payment_date: Date,
        growth_only: bool = False,
    ) -> None:
        # C++ parity: ql/cashflows/cpicoupon.cpp:139-150 — base date is
        # passed as-is, fixing date is ``observationDate - observationLag``.
        super().__init__(
            notional=notional,
            index=index,
            base_date=base_date,
            fixing_date=observation_date - observation_lag,
            payment_date=payment_date,
            growth_only=growth_only,
        )
        qassert.require(
            base_fixing is not None or base_date != _NULL_DATE,
            "baseCPI and baseDate can not be both null, "
            "provide a valid baseCPI or baseDate",
        )
        if base_fixing is not None:
            qassert.require(
                abs(base_fixing) > _CPI_EPSILON,
                "|baseCPI_| < 1e-16, future divide-by-zero problem",
            )
        self._base_fixing: float | None = base_fixing
        self._observation_date: Date = observation_date
        self._observation_lag: Period = observation_lag
        self._interpolation: InterpolationType = interpolation
        self._frequency = index.frequency()
        self._cpi_index: ZeroInflationIndex = index

    # ---- inspectors --------------------------------------------------

    def observation_date(self) -> Date:
        """C++ parity: ql/cashflows/cpicoupon.hpp:182 (inline)."""
        return self._observation_date

    def observation_lag(self) -> Period:
        return self._observation_lag

    def interpolation(self) -> InterpolationType:
        return self._interpolation

    def cpi_index(self) -> ZeroInflationIndex:
        return self._cpi_index

    def base_date(self) -> Date:
        """C++ parity: ql/cashflows/cpicoupon.cpp:159-166 — raises if no
        base date was specified.
        """
        d = self._base_date
        qassert.require(d != _NULL_DATE, "no base date specified")
        return d

    # ---- IndexedCashFlow overrides -----------------------------------

    def base_fixing(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.cpp:168-173.

        If an explicit base fixing was provided, return it. Otherwise
        look up the lagged fixing at the base date with the configured
        interpolation (and zero lag, since the lag has already been
        consumed by the fact that the base date is *the* base reference).
        """
        if self._base_fixing is not None:
            return self._base_fixing
        return lagged_fixing(
            self._cpi_index,
            self.base_date(),
            Period(0, TimeUnit.Months),
            self._interpolation,
        )

    def index_fixing(self) -> float:
        """C++ parity: ql/cashflows/cpicoupon.cpp:175-177."""
        return lagged_fixing(
            self._cpi_index,
            self._observation_date,
            self._observation_lag,
            self._interpolation,
        )
