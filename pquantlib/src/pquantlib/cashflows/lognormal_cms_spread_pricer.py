"""LognormalCmsSpreadPricer — bivariate-lognormal CMS-spread coupon pricer.

# C++ parity: ql/experimental/coupons/lognormalcmsspreadpricer.hpp + .cpp
# (v1.42.1, 099987f0).

Prices a CMS-spread coupon (the spread between two CMS rates) using the
bivariate model of Brigo & Mercurio, *Interest Rate Models* 2nd ed., ch.
13.6.2, with the shifted-lognormal / normal extensions of
http://ssrn.com/abstract=2686998. The two underlying CMS rates are
convexity-adjusted by an inner :class:`CmsCouponPricer` (Hagan), and the
spread option is integrated by Gauss-Hermite quadrature.

# C++ parity divergences:
# - The C++ pricer special-cases a ``SwaptionVolatilityCube`` (to convert
#   between vol types via per-strike smile sections). PQuantLib's swaption-vol
#   surfaces used here are ATM surfaces (no cube), so the pricer takes the
#   inherited-vol-type branch (``swcub == nullptr``): the volatilities come
#   straight from ``swaptionVolatility().volatility(date, tenor, rate)`` and
#   the vol type must be inherited (matching the C++ ``QL_REQUIRE`` on that
#   branch). The full cube-conversion path lands if/when a vol cube with a
#   3-arg ``smileSection(strike, volType, shift)`` smile is wired in.
# - The ``optionletPrice`` cap/floor path (used by capletRate / floorletRate)
#   is ported in full; only ``swapletRate`` (the canonical CMS-spread coupon
#   rate) is exercised by the probe.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cms_coupon import CmsCoupon
from pquantlib.cashflows.cms_spread_coupon import CmsSpreadCoupon
from pquantlib.cashflows.cms_spread_coupon_pricer import CmsSpreadCouponPricer
from pquantlib.math.constants import M_SQRT2
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.integrals.gaussian_quadrature import GaussHermiteIntegration
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import bachelier_black_formula
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

if TYPE_CHECKING:
    from pquantlib.cashflows.cms_coupon_pricer import CmsCouponPricer
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.indexes.swap_spread_index import SwapSpreadIndex
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.date import Date

_M_SQRTPI: float = math.sqrt(math.pi)


class LognormalCmsSpreadPricer(CmsSpreadCouponPricer):
    """CMS-spread coupon pricer (Brigo-Mercurio 13.6.2 bivariate model).

    C++ parity: lognormalcmsspreadpricer.hpp/.cpp.
    """

    def __init__(  # noqa: PLR0915 (faithful port of the C++ member-init list)
        self,
        cms_pricer: CmsCouponPricer,
        correlation: Quote,
        coupon_discount_curve: YieldTermStructureProtocol | None = None,
        integration_points: int = 16,
        volatility_type: VolatilityType | None = None,
        shift1: float | None = None,
        shift2: float | None = None,
    ) -> None:
        super().__init__(correlation)
        self._cms_pricer = cms_pricer
        self._coupon_discount_curve = coupon_discount_curve

        qassert.require(
            integration_points >= 4,
            f"at least 4 integration points should be used ({integration_points})",
        )
        self._integrator = GaussHermiteIntegration(integration_points)
        self._cnd = CumulativeNormalDistribution(0.0, 1.0)

        if volatility_type is None:
            qassert.require(
                shift1 is None and shift2 is None,
                "if volatility type is inherited, no shifts should be specified",
            )
            self._inherited_volatility_type = True
            vol = cms_pricer.swaption_volatility()
            assert vol is not None
            self._vol_type = vol.volatility_type()
            self._shift1 = 0.0
            self._shift2 = 0.0
        else:
            self._shift1 = 0.0 if shift1 is None else shift1
            self._shift2 = 0.0 if shift2 is None else shift2
            self._inherited_volatility_type = False
            self._vol_type = volatility_type

        # per-coupon state, populated by initialize()
        self._coupon: CmsSpreadCoupon | None = None
        self._today: Date | None = None
        self._fixing_date: Date | None = None
        self._payment_date: Date | None = None
        self._fixing_time = 0.0
        self._gearing = 1.0
        self._spread = 0.0
        self._spread_leg_value = 0.0
        self._discount = 1.0
        self._index: SwapSpreadIndex | None = None
        self._swap_rate1 = 0.0
        self._swap_rate2 = 0.0
        self._gearing1 = 0.0
        self._gearing2 = 0.0
        self._adjusted_rate1 = 0.0
        self._adjusted_rate2 = 0.0
        self._vol1 = 0.0
        self._vol2 = 0.0
        self._mu1 = 0.0
        self._mu2 = 0.0
        self._rho = 0.0
        self._c1: CmsCoupon | None = None
        self._c2: CmsCoupon | None = None
        # mutable integrand scratch (C++ mutable members)
        self._phi = 0.0
        self._a = 0.0
        self._b = 0.0
        self._s1 = 0.0
        self._s2 = 0.0
        self._m1 = 0.0
        self._m2 = 0.0
        self._v1 = 0.0
        self._v2 = 0.0
        self._k = 0.0

    # --- integrands ----------------------------------------------------

    def _integrand(self, x: float) -> float:
        # Brigo, 13.16.2 with x = v / sqrt(2). C++ parity: lognormal...cpp:83-109.
        v = M_SQRT2 * x
        ft = self._fixing_time
        sqrt_ft = math.sqrt(ft)
        rho = self._rho
        h = self._k - self._b * self._s2 * math.exp(
            (self._m2 - 0.5 * self._v2 * self._v2) * ft + self._v2 * sqrt_ft * v
        )
        denom = self._v1 * math.sqrt(ft * (1.0 - rho * rho))
        phi1 = self._cnd(
            self._phi
            * (
                math.log(self._a * self._s1 / h)
                + (self._m1 + (0.5 - rho * rho) * self._v1 * self._v1) * ft
                + rho * self._v1 * sqrt_ft * v
            )
            / denom
        )
        phi2 = self._cnd(
            self._phi
            * (
                math.log(self._a * self._s1 / h)
                + (self._m1 - 0.5 * self._v1 * self._v1) * ft
                + rho * self._v1 * sqrt_ft * v
            )
            / denom
        )
        f = (
            self._a
            * self._phi
            * self._s1
            * math.exp(
                self._m1 * ft
                - 0.5 * rho * rho * self._v1 * self._v1 * ft
                + rho * self._v1 * sqrt_ft * v
            )
            * phi1
            - self._phi * h * phi2
        )
        return math.exp(-x * x) * f

    # --- pricer wiring -------------------------------------------------

    def initialize(self, coupon: FloatingRateCoupon) -> None:
        # C++ parity: lognormalcmsspreadpricer.cpp:131-249.
        qassert.require(isinstance(coupon, CmsSpreadCoupon), "CMS spread coupon needed")
        assert isinstance(coupon, CmsSpreadCoupon)
        self._coupon = coupon
        index = coupon.swap_spread_index()
        self._index = index
        self._gearing = coupon.gearing()
        self._spread = coupon.spread()
        self._fixing_date = coupon.fixing_date()
        self._payment_date = coupon.date()

        self._today = ObservableSettings().evaluation_date_or_today()

        discount_curve = self._coupon_discount_curve
        if discount_curve is None:
            s1 = index.swap_index1()
            discount_curve = (
                s1.discounting_term_structure()
                if s1.exogenous_discount()
                else s1.forwarding_term_structure()
            )
            self._coupon_discount_curve = discount_curve
        assert discount_curve is not None

        self._discount = (
            discount_curve.discount(self._payment_date)
            if self._payment_date > discount_curve.reference_date()
            else 1.0
        )
        self._spread_leg_value = self._spread * coupon.accrual_period() * self._discount

        self._gearing1 = index.gearing1()
        self._gearing2 = index.gearing2()
        qassert.require(
            self._gearing1 > 0.0 and self._gearing2 < 0.0,
            f"gearing1 ({self._gearing1}) should be positive while gearing2 "
            f"({self._gearing2}) should be negative",
        )

        self._c1 = CmsCoupon(
            coupon.date(),
            coupon.nominal(),
            coupon.accrual_start_date(),
            coupon.accrual_end_date(),
            coupon.fixing_days(),
            index.swap_index1(),
            1.0,
            0.0,
            coupon.reference_period_start(),
            coupon.reference_period_end(),
            coupon.day_counter(),
            coupon.is_in_arrears(),
        )
        self._c2 = CmsCoupon(
            coupon.date(),
            coupon.nominal(),
            coupon.accrual_start_date(),
            coupon.accrual_end_date(),
            coupon.fixing_days(),
            index.swap_index2(),
            1.0,
            0.0,
            coupon.reference_period_start(),
            coupon.reference_period_end(),
            coupon.day_counter(),
            coupon.is_in_arrears(),
        )
        self._c1.set_pricer(self._cms_pricer)
        self._c2.set_pricer(self._cms_pricer)

        assert self._today is not None
        if self._fixing_date > self._today:
            vol = self._cms_pricer.swaption_volatility()
            assert vol is not None
            self._fixing_time = vol.time_from_reference(self._fixing_date)
            self._swap_rate1 = self._c1.index_fixing()
            self._swap_rate2 = self._c2.index_fixing()
            self._adjusted_rate1 = self._c1.adjusted_fixing()
            self._adjusted_rate2 = self._c2.adjusted_fixing()

            if self._inherited_volatility_type and self._vol_type == VolatilityType.ShiftedLognormal:
                self._shift1 = vol.shift(self._fixing_date, index.swap_index1().tenor())
                self._shift2 = vol.shift(self._fixing_date, index.swap_index2().tenor())

            # PQuantLib uses ATM surfaces (no cube): inherited-vol-type branch.
            qassert.require(
                self._inherited_volatility_type,
                "if only an atm surface is given, the volatility type must be inherited",
            )
            self._vol1 = vol.volatility(
                self._fixing_date, index.swap_index1().tenor(), self._swap_rate1
            )
            self._vol2 = vol.volatility(
                self._fixing_date, index.swap_index2().tenor(), self._swap_rate2
            )

            if self._vol_type == VolatilityType.ShiftedLognormal:
                self._mu1 = (
                    1.0
                    / self._fixing_time
                    * math.log(
                        (self._adjusted_rate1 + self._shift1)
                        / (self._swap_rate1 + self._shift1)
                    )
                )
                self._mu2 = (
                    1.0
                    / self._fixing_time
                    * math.log(
                        (self._adjusted_rate2 + self._shift2)
                        / (self._swap_rate2 + self._shift2)
                    )
                )
            corr = self.correlation()
            assert corr is not None
            self._rho = max(min(corr.value(), 0.9999), -0.9999)
        else:
            self._adjusted_rate1 = self._c1.index_fixing()
            self._adjusted_rate2 = self._c2.index_fixing()

    def _optionlet_price(self, option_type: OptionType, strike: float) -> float:
        # C++ parity: lognormalcmsspreadpricer.cpp:251-297 (future fixings only).
        self._phi = 1.0 if option_type == OptionType.Call else -1.0
        res = 0.0
        if self._vol_type == VolatilityType.ShiftedLognormal:
            if strike >= 0.0:
                self._a = self._gearing1
                self._b = self._gearing2
                self._s1 = self._swap_rate1 + self._shift1
                self._s2 = self._swap_rate2 + self._shift2
                self._m1 = self._mu1
                self._m2 = self._mu2
                self._v1 = self._vol1
                self._v2 = self._vol2
                self._k = strike + self._gearing1 * self._shift1 + self._gearing2 * self._shift2
            else:
                self._a = -self._gearing2
                self._b = -self._gearing1
                self._s1 = self._swap_rate2 + self._shift1
                self._s2 = self._swap_rate1 + self._shift2
                self._m1 = self._mu2
                self._m2 = self._mu1
                self._v1 = self._vol2
                self._v2 = self._vol1
                self._k = -strike - self._gearing1 * self._shift1 - self._gearing2 * self._shift2
                res += self._phi * (
                    self._gearing1 * self._adjusted_rate1
                    + self._gearing2 * self._adjusted_rate2
                    - strike
                )
            res += 1.0 / _M_SQRTPI * self._integrator(self._integrand)
        else:
            forward = (
                self._gearing1 * self._adjusted_rate1 + self._gearing2 * self._adjusted_rate2
            )
            stddev = math.sqrt(
                self._fixing_time
                * (
                    self._gearing1 * self._gearing1 * self._vol1 * self._vol1
                    + self._gearing2 * self._gearing2 * self._vol2 * self._vol2
                    + 2.0
                    * self._gearing1
                    * self._gearing2
                    * self._rho
                    * self._vol1
                    * self._vol2
                )
            )
            res = bachelier_black_formula(option_type, strike, forward, stddev, 1.0)
        assert self._coupon is not None
        return res * self._discount * self._coupon.accrual_period()

    # --- CouponPricer interface ----------------------------------------

    def swaplet_price(self) -> float:
        # C++ parity: lognormalcmsspreadpricer.cpp:341-345.
        assert self._coupon is not None
        return (
            self._gearing
            * self._coupon.accrual_period()
            * self._discount
            * (self._gearing1 * self._adjusted_rate1 + self._gearing2 * self._adjusted_rate2)
            + self._spread_leg_value
        )

    def swaplet_rate(self) -> float:
        assert self._coupon is not None
        return self.swaplet_price() / (self._coupon.accrual_period() * self._discount)

    def caplet_price(self, effective_cap: float) -> float:
        # C++ parity: lognormalcmsspreadpricer.cpp:303-315.
        assert self._coupon is not None
        assert self._fixing_date is not None
        assert self._today is not None
        if self._fixing_date <= self._today:
            rs = max(self._coupon.index().fixing(self._fixing_date) - effective_cap, 0.0)
            return self._gearing * rs * self._coupon.accrual_period() * self._discount
        return self._gearing * self._optionlet_price(OptionType.Call, effective_cap)

    def caplet_rate(self, effective_cap: float) -> float:
        assert self._coupon is not None
        return self.caplet_price(effective_cap) / (
            self._coupon.accrual_period() * self._discount
        )

    def floorlet_price(self, effective_floor: float) -> float:
        # C++ parity: lognormalcmsspreadpricer.cpp:322-334.
        assert self._coupon is not None
        assert self._fixing_date is not None
        assert self._today is not None
        if self._fixing_date <= self._today:
            rs = max(effective_floor - self._coupon.index().fixing(self._fixing_date), 0.0)
            return self._gearing * rs * self._coupon.accrual_period() * self._discount
        return self._gearing * self._optionlet_price(OptionType.Put, effective_floor)

    def floorlet_rate(self, effective_floor: float) -> float:
        assert self._coupon is not None
        return self.floorlet_price(effective_floor) / (
            self._coupon.accrual_period() * self._discount
        )


__all__ = ["LognormalCmsSpreadPricer"]
