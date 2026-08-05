"""LinearTsrPricer — CMS-coupon pricing with a linear terminal-swap-rate model.

# C++ parity: ql/cashflows/lineartsrpricer.hpp + .cpp (v1.43).

Prices a CMS coupon by static replication under a linear terminal swap-rate
model whose slope is linked to a Gaussian (GSR) short-rate model. Reference:
Andersen & Piterbarg, *Interest Rate Modeling*, 16.3.2.

The cut-off point for the replication integral can be set

* by explicitly specifying the lower and upper rate bound
  (:class:`Strategy.RateBound`),
* by defining the bounds to be the strike where a vanilla swaption has a given
  fraction of the ATM swaption's vega (:class:`Strategy.VegaRatio`),
* by defining the bounds to be the strike where undeflated payer / receiver
  prices fall below a threshold (:class:`Strategy.PriceThreshold`),
* by specifying a number of standard deviations of a Black-Scholes process at
  the ATM volatility (:class:`Strategy.BSStdDevs`).

In every case the lower and upper bound are applied as well. For a shifted
lognormal smile the bounds are applied to ``strike + shift``; for normal
volatility input the lower bound is pulled to ``min(-upper, lower)` unless the
bounds were set explicitly.

Module layout vs C++
--------------------
C++ nests ``Settings`` (with its ``Strategy`` enum) and the two solver
objectives ``VegaRatioHelper`` / ``PriceHelper`` inside ``LinearTsrPricer``.
Following the house convention for nested C++ types (cf. ``RateAveraging``,
``Thirty360.Convention``), they are flattened to module level here:
:class:`Strategy`, :class:`Settings`, :class:`VegaRatioHelper`,
:class:`PriceHelper`.

# C++ parity divergences
# ----------------------
# - Handles: C++ threads ``Handle<SwaptionVolatilityStructure>``,
#   ``Handle<Quote>`` and ``Handle<YieldTermStructure>``. PQuantLib threads the
#   objects directly (project-wide convention — see ``CmsCouponPricer``); an
#   *empty* handle is spelled ``None``. Dereferencing an empty handle raises in
#   C++, and ``None`` raises a ``LibraryException`` here, at the same points.
# - ``OvernightIndexedSwapIndex``: C++ ``initialize`` dynamic_casts the swap
#   index to ``OvernightIndexedSwapIndex`` to pick up its own
#   ``underlyingSwap``. PQuantLib has no ``OvernightIndexedSwapIndex`` yet, so
#   the branch is unreachable and only ``SwapIndex.underlying_swap`` is called
#   — which is exactly what the C++ else-branch does.
#
# KNOWN UPSTREAM DEFECTS, REPRODUCED DELIBERATELY (see ``_optionlet_price``
# and ``_strike_from_price``): in v1.43 the ``PriceThreshold`` strategy is
# inert. Both quirks are pinned by the cross-validation probe
# ``v143/cf/lineartsr``.
"""

from __future__ import annotations

import contextlib
import math
from enum import IntEnum
from typing import TYPE_CHECKING, Final

from pquantlib import qassert
from pquantlib.cashflows.cms_coupon import CmsCoupon
from pquantlib.cashflows.cms_coupon_pricer import CmsCouponPricer, MeanRevertingPricer
from pquantlib.cashflows.coupon import Coupon
from pquantlib.math.integrals.integrator import Integrator
from pquantlib.math.integrals.kronrod import GaussKronrodNonAdaptive
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.termstructures.volatility.atm_smile_section import AtmSmileSection
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

if TYPE_CHECKING:
    from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
        SwaptionVolatilityStructure,
    )
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period

# C++ parity: lineartsrpricer.cpp:52-53 —
# ``const Real LinearTsrPricer::defaultLowerBound = 0.0001,
#              LinearTsrPricer::defaultUpperBound = 2.0000;``
DEFAULT_LOWER_BOUND: Final[float] = 0.0001
DEFAULT_UPPER_BOUND: Final[float] = 2.0000

# C++ parity: lineartsrpricer.cpp:66-67 — the integrator installed when the
# caller supplies none.
_DEFAULT_INTEGRATOR_ABS_ACCURACY: Final[float] = 1.0e-10
_DEFAULT_INTEGRATOR_MAX_EVALUATIONS: Final[int] = 5000
_DEFAULT_INTEGRATOR_REL_ACCURACY: Final[float] = 1.0e-10

# C++ parity: lineartsrpricer.cpp:73 — below this |mean reversion| GsrG
# degenerates to the plain year fraction.
_MEAN_REVERSION_CUTOFF: Final[float] = 1.0e-4

# C++ parity: lineartsrpricer.cpp:216 / 243 — Brent accuracy in both
# strike-search helpers.
_STRIKE_SOLVER_ACCURACY: Final[float] = 1.0e-5


class Strategy(IntEnum):
    """How the replication-integral cut-off is chosen.

    # C++ parity: ``LinearTsrPricer::Settings::Strategy``
    # (lineartsrpricer.hpp:141-146). Integer values match the C++
    # declaration order.
    """

    RateBound = 0
    VegaRatio = 1
    PriceThreshold = 2
    BSStdDevs = 3


class Settings:
    """Cut-off configuration for :class:`LinearTsrPricer`.

    # C++ parity: ``LinearTsrPricer::Settings`` (lineartsrpricer.hpp:70-153).

    The C++ ``with*`` members mutate and return ``*this``; the Python
    ``with_*`` methods do the same and return ``self``, so they chain::

        Settings().with_vega_ratio(0.05, 0.005, 0.045)

    Each strategy setter is a full reset of the strategy-relevant state: the
    one-argument overloads restore the default bounds AND set
    ``default_bounds = True``; the three-argument overloads install explicit
    bounds and clear the flag. Only ``with_rate_bound`` leaves the ratio /
    threshold / std-devs fields alone, exactly as C++ does.
    """

    def __init__(self) -> None:
        # C++ parity: the in-class member initialisers plus the default ctor,
        # which sets lower/upper to the two default bounds.
        self.strategy: Strategy = Strategy.RateBound
        self.vega_ratio: float = 0.01
        self.price_threshold: float = 1.0e-8
        self.std_devs: float = 3.0
        self.lower_rate_bound: float = DEFAULT_LOWER_BOUND
        self.upper_rate_bound: float = DEFAULT_UPPER_BOUND
        self.default_bounds: bool = True

    def with_rate_bound(
        self,
        lower_rate_bound: float = DEFAULT_LOWER_BOUND,
        upper_rate_bound: float = DEFAULT_UPPER_BOUND,
    ) -> Settings:
        """# C++ parity: ``Settings::withRateBound`` (lineartsrpricer.hpp:74-81)."""
        self.strategy = Strategy.RateBound
        self.lower_rate_bound = lower_rate_bound
        self.upper_rate_bound = upper_rate_bound
        self.default_bounds = False
        return self

    def with_vega_ratio(
        self,
        vega_ratio: float = 0.01,
        lower_rate_bound: float | None = None,
        upper_rate_bound: float | None = None,
    ) -> Settings:
        """# C++ parity: ``Settings::withVegaRatio`` — BOTH overloads.

        C++ has a one-argument form (default bounds, ``defaultBounds_ = true``)
        and a three-argument form (explicit bounds, ``defaultBounds_ = false``).
        Python collapses them onto optional bounds: passing neither bound
        reproduces the one-argument overload, passing both reproduces the
        three-argument one.
        """
        self.strategy = Strategy.VegaRatio
        self.vega_ratio = vega_ratio
        return self._apply_bounds(lower_rate_bound, upper_rate_bound)

    def with_price_threshold(
        self,
        price_threshold: float = 1.0e-8,
        lower_rate_bound: float | None = None,
        upper_rate_bound: float | None = None,
    ) -> Settings:
        """# C++ parity: ``Settings::withPriceThreshold`` — BOTH overloads.

        .. warning:: ``price_threshold`` is stored but never read by the
           v1.43 model; see :meth:`LinearTsrPricer._optionlet_price`.
        """
        self.strategy = Strategy.PriceThreshold
        self.price_threshold = price_threshold
        return self._apply_bounds(lower_rate_bound, upper_rate_bound)

    def with_bs_std_devs(
        self,
        std_devs: float = 3.0,
        lower_rate_bound: float | None = None,
        upper_rate_bound: float | None = None,
    ) -> Settings:
        """# C++ parity: ``Settings::withBSStdDevs`` — BOTH overloads."""
        self.strategy = Strategy.BSStdDevs
        self.std_devs = std_devs
        return self._apply_bounds(lower_rate_bound, upper_rate_bound)

    def _apply_bounds(self, lower: float | None, upper: float | None) -> Settings:
        """Shared tail of the non-RateBound setters (one- vs three-arg form)."""
        qassert.require(
            (lower is None) == (upper is None),
            "lower_rate_bound and upper_rate_bound must be given together",
        )
        if lower is None or upper is None:
            self.lower_rate_bound = DEFAULT_LOWER_BOUND
            self.upper_rate_bound = DEFAULT_UPPER_BOUND
            self.default_bounds = True
        else:
            self.lower_rate_bound = lower
            self.upper_rate_bound = upper
            self.default_bounds = False
        return self


class VegaRatioHelper:
    """Brent objective: ``section.vega(strike) - target_vega``.

    # C++ parity: ``LinearTsrPricer::VegaRatioHelper``
    # (lineartsrpricer.hpp:188-198).
    """

    def __init__(self, section: SmileSection, target_vega: float) -> None:
        self.section: SmileSection = section
        self.target_vega: float = target_vega

    def __call__(self, strike: float, /) -> float:
        return self.section.vega(strike) - self.target_vega


class PriceHelper:
    """Brent objective: ``section.option_price(strike, type) - target_price``.

    # C++ parity: ``LinearTsrPricer::PriceHelper``
    # (lineartsrpricer.hpp:200-212).
    """

    def __init__(self, section: SmileSection, option_type: OptionType, target_price: float) -> None:
        self.section: SmileSection = section
        self.option_type: OptionType = option_type
        self.target_price: float = target_price

    def __call__(self, strike: float, /) -> float:
        return self.section.option_price(strike, int(self.option_type)) - self.target_price


class LinearTsrPricer(CmsCouponPricer, MeanRevertingPricer):
    """CMS-coupon pricer using a linear terminal swap-rate model.

    # C++ parity: ``LinearTsrPricer`` (lineartsrpricer.hpp:64-236 +
    # lineartsrpricer.cpp).
    """

    def __init__(
        self,
        swaption_vol: SwaptionVolatilityStructure | None,
        mean_reversion: Quote | None,
        coupon_discount_curve: YieldTermStructureProtocol | None = None,
        settings: Settings | None = None,
        integrator: Integrator | None = None,
    ) -> None:
        # C++ parity: lineartsrpricer.cpp:55-68.
        super().__init__(swaption_vol)
        qassert.require(
            swaption_vol is not None,
            "LinearTsrPricer requires a swaption volatility structure",
        )
        assert swaption_vol is not None
        self._mean_reversion: Quote | None = mean_reversion
        self._coupon_discount_curve: YieldTermStructureProtocol | None = coupon_discount_curve
        self._settings: Settings = settings if settings is not None else Settings()
        self._vol_day_counter: DayCounter = swaption_vol.day_counter()
        self._integrator: Integrator = (
            integrator
            if integrator is not None
            else GaussKronrodNonAdaptive(
                _DEFAULT_INTEGRATOR_ABS_ACCURACY,
                _DEFAULT_INTEGRATOR_MAX_EVALUATIONS,
                _DEFAULT_INTEGRATOR_REL_ACCURACY,
            )
        )
        if self._mean_reversion is not None:
            self._mean_reversion.register_with(self)
        if self._coupon_discount_curve is not None:
            self._coupon_discount_curve.register_with(self)  # type: ignore[attr-defined]

        # --- state filled by initialize() ---
        self._a: float = 0.0
        self._b: float = 0.0
        self._coupon: CmsCoupon | None = None
        self._forward_curve: YieldTermStructureProtocol | None = None
        self._discount_curve: YieldTermStructureProtocol | None = None
        self._today: Date | None = None
        self._payment_date: Date | None = None
        self._fixing_date: Date | None = None
        self._gearing: float = 1.0
        self._spread: float = 0.0
        self._swap_tenor: Period | None = None
        self._spread_leg_value: float = 0.0
        self._swap_rate_value: float = 0.0
        self._coupon_discount_ratio: float = 1.0
        self._discount_curve_payment_discount: float = 1.0
        self._annuity: float = 0.0
        self._swap_index: SwapIndex | None = None
        self._smile_section: SmileSection | None = None
        self._adjusted_lower_bound: float = self._settings.lower_rate_bound
        self._adjusted_upper_bound: float = self._settings.upper_rate_bound

    # --- inspectors ----------------------------------------------------

    def settings(self) -> Settings:
        """The cut-off configuration in force (C++ ``settings_``)."""
        return self._settings

    def integrator(self) -> Integrator:
        """The quadrature rule in force (C++ ``integrator_``)."""
        return self._integrator

    def coupon_discount_curve(self) -> YieldTermStructureProtocol | None:
        """The coupon-discount curve, or ``None`` for C++'s empty handle."""
        return self._coupon_discount_curve

    # --- MeanRevertingPricer -------------------------------------------

    def mean_reversion(self) -> float:
        """# C++ parity: lineartsrpricer.cpp:342 — dereferences the handle."""
        return self._require_mean_reversion().value()

    def set_mean_reversion(self, mean_reversion: Quote) -> None:
        """# C++ parity: lineartsrpricer.hpp:172-177."""
        if self._mean_reversion is not None:
            self._mean_reversion.unregister_with(self)
        self._mean_reversion = mean_reversion
        self._mean_reversion.register_with(self)
        self.update()

    def _require_mean_reversion(self) -> Quote:
        # C++ raises "empty Handle cannot be dereferenced"; the Python
        # equivalent of an empty handle is ``None``.
        qassert.require(
            self._mean_reversion is not None,
            "empty mean reversion quote cannot be dereferenced",
        )
        assert self._mean_reversion is not None
        return self._mean_reversion

    # --- model internals -----------------------------------------------

    def _gsr_g(self, d: Date) -> float:
        """GSR ``G(fixing, d)``.

        # C++ parity: lineartsrpricer.cpp:70-79 — for |mean reversion| below
        # 1e-4 the closed form ``(1 - exp(-a*t)) / a`` degenerates numerically,
        # so C++ returns the year fraction itself.
        """
        assert self._fixing_date is not None
        yf = self._vol_day_counter.year_fraction(self._fixing_date, d)
        a = self._require_mean_reversion().value()
        if abs(a) < _MEAN_REVERSION_CUTOFF:
            return yf
        return (1.0 - math.exp(-a * yf)) / a

    def _singular_terms(self, option_type: OptionType, strike: float) -> float:
        """# C++ parity: lineartsrpricer.cpp:81-94."""
        assert self._smile_section is not None
        omega = 1.0 if option_type == OptionType.Call else -1.0
        s1 = max(omega * (self._swap_rate_value - strike), 0.0) * (self._a * self._swap_rate_value + self._b)
        inner = OptionType.Put if strike < self._swap_rate_value else OptionType.Call
        s2 = (self._a * strike + self._b) * self._smile_section.option_price(strike, int(inner))
        return s1 + s2

    def _integrand(self, strike: float, /) -> float:
        """# C++ parity: lineartsrpricer.cpp:96-100 (``integrand_f``)."""
        assert self._smile_section is not None
        inner = OptionType.Put if strike < self._swap_rate_value else OptionType.Call
        return 2.0 * self._a * self._smile_section.option_price(strike, int(inner))

    # --- initialization -------------------------------------------------

    def initialize(self, coupon: FloatingRateCoupon) -> None:  # noqa: PLR0915  (faithful C++ port — LinearTsrPricer::initialize is one long straight-line function)
        """# C++ parity: lineartsrpricer.cpp:102-205."""
        qassert.require(isinstance(coupon, CmsCoupon), "CMS coupon needed")
        assert isinstance(coupon, CmsCoupon)
        self._coupon = coupon
        self._gearing = coupon.gearing()
        self._spread = coupon.spread()

        self._fixing_date = coupon.fixing_date()
        self._payment_date = coupon.date()
        swap_index = coupon.swap_index()
        self._swap_index = swap_index

        self._forward_curve = swap_index.forwarding_term_structure()
        if swap_index.exogenous_discount():
            self._discount_curve = swap_index.discounting_term_structure()
        else:
            self._discount_curve = self._forward_curve
        qassert.require(
            self._discount_curve is not None,
            "no term structure set to the swap index of this CMS coupon",
        )
        discount_curve = self._discount_curve
        assert discount_curve is not None

        # If no coupon-discount curve is given just use the discounting curve
        # from the swap index. For rate calculation this curve cancels out in
        # the computation, so e.g. the discounting swap engine will produce
        # correct results even if the coupon-discount curve is not set here;
        # only the price members depend on it.
        today = ObservableSettings().evaluation_date_or_today()
        self._today = today

        cdc = self._coupon_discount_curve
        if cdc is not None and self._payment_date > cdc.reference_date():
            coupon_curve_payment_discount = cdc.discount(self._payment_date)
        else:
            coupon_curve_payment_discount = 1.0

        if self._payment_date > discount_curve.reference_date():
            self._discount_curve_payment_discount = discount_curve.discount(self._payment_date)
        else:
            self._discount_curve_payment_discount = 1.0

        self._coupon_discount_ratio = coupon_curve_payment_discount / self._discount_curve_payment_discount

        self._spread_leg_value = (
            self._spread
            * coupon.accrual_period()
            * self._discount_curve_payment_discount
            * self._coupon_discount_ratio
        )

        if self._fixing_date > today:
            self._swap_tenor = swap_index.tenor()
            # # C++ parity divergence: C++ additionally dynamic_casts the index
            # # to OvernightIndexedSwapIndex; PQuantLib has no such class, so
            # # only the plain SwapIndex branch exists.
            swap = swap_index.underlying_swap(self._fixing_date)
            self._swap_rate_value = swap.fair_rate()
            self._annuity = 1.0e4 * abs(swap.fixed_leg_bps())
            swap_fixed_leg = swap.fixed_leg()

            vol = self.swaption_volatility()
            assert vol is not None
            section_tmp = vol.smile_section(self._fixing_date, self._swap_tenor)

            self._adjusted_lower_bound = self._settings.lower_rate_bound
            self._adjusted_upper_bound = self._settings.upper_rate_bound

            if section_tmp.volatility_type() == VolatilityType.Normal:
                # adjust lower bound if it was not set explicitly
                if self._settings.default_bounds:
                    self._adjusted_lower_bound = min(self._adjusted_lower_bound, -self._adjusted_upper_bound)
            else:
                # adjust bounds by the section's shift
                self._adjusted_lower_bound -= section_tmp.shift()
                self._adjusted_upper_bound -= section_tmp.shift()

            # If the section does not provide an ATM level, enhance it to have
            # one — no need to exit with an exception.
            # (C++ tests ``atmLevel() == Null<Real>()``; PQuantLib's
            # not-supplied sentinel is NaN.)
            if math.isnan(section_tmp.atm_level()):
                self._smile_section = AtmSmileSection(base=section_tmp, atm=self._swap_rate_value)
            else:
                self._smile_section = section_tmp

            # compute the linear model's parameters
            gx = 0.0
            gy = 0.0
            for cf in swap_fixed_leg:
                qassert.require(isinstance(cf, Coupon), "fixed leg cash flow is not a Coupon")
                assert isinstance(cf, Coupon)
                yf = cf.accrual_period()
                d = cf.date()
                pv = yf * discount_curve.discount(d)
                gx += pv * self._gsr_g(d)
                gy += pv

            gamma = gx / gy
            lastd = swap_fixed_leg[-1].date()

            self._a = (
                discount_curve.discount(self._payment_date)
                * (gamma - self._gsr_g(self._payment_date))
                / (discount_curve.discount(lastd) * self._gsr_g(lastd) + self._swap_rate_value * gy * gamma)
            )
            self._b = discount_curve.discount(self._payment_date) / gy - self._a * self._swap_rate_value

    # --- strike search --------------------------------------------------

    def _strike_from_vega_ratio(
        self, ratio: float, option_type: OptionType, reference_strike: float
    ) -> float:
        """# C++ parity: lineartsrpricer.cpp:207-235."""
        section = self._smile_section
        assert section is not None
        if option_type == OptionType.Call:
            a = self._swap_rate_value
            lo = reference_strike
            b = hi = k = min(section.max_strike(), self._adjusted_upper_bound)
        else:
            a = lo = k = max(section.min_strike(), self._adjusted_lower_bound)
            b = self._swap_rate_value
            hi = reference_strike

        h = VegaRatioHelper(section, section.vega(self._swap_rate_value) * ratio)
        solver = Brent()
        # C++ parity: `try { ... } catch (...) { /* use default value set above */ }`.
        with contextlib.suppress(Exception):
            k = solver.solve(h, _STRIKE_SOLVER_ACCURACY, (a + b) / 2.0, a, b)
        return min(max(k, lo), hi)

    def _strike_from_price(self, price: float, option_type: OptionType, reference_strike: float) -> float:
        """# C++ parity: lineartsrpricer.cpp:237-262.

        .. warning:: UPSTREAM DEFECT, reproduced deliberately. C++ passes the
           bracket end ``swapRateValue_`` as Brent's *guess*: for a call the
           bracket is ``(swapRateValue_, upper)`` and for a put it is
           ``(lower, swapRateValue_)``, so the guess always coincides with one
           end. ``Solver1D::solve`` requires ``xMin < guess < xMax`` and
           therefore always throws, the blanket ``catch (...)`` swallows it,
           and ``k`` keeps the plain rate bound assigned above. The whole
           ``PriceThreshold`` strategy consequently degenerates to
           ``RateBound``. Both PQuantLib's ``Brent``/``Solver1D`` and C++'s
           enforce the same guess check, so simply porting the code faithfully
           reproduces the behaviour — pinned by the probe cases
           ``price_threshold_1arg`` (≡ ``default``) and ``price_threshold_3arg``
           (≡ ``rate_bound_narrow``).
        """
        section = self._smile_section
        assert section is not None
        if option_type == OptionType.Call:
            a = self._swap_rate_value
            lo = reference_strike
            b = hi = k = min(section.max_strike(), self._adjusted_upper_bound)
        else:
            a = lo = k = max(section.min_strike(), self._adjusted_lower_bound)
            b = self._swap_rate_value
            hi = reference_strike

        h = PriceHelper(section, option_type, price)
        solver = Brent()
        # C++ parity: `try { ... } catch (...) { /* use default value set above */ }`.
        # This suppression is load-bearing — see the warning above.
        with contextlib.suppress(Exception):
            k = solver.solve(h, _STRIKE_SOLVER_ACCURACY, self._swap_rate_value, a, b)
        return min(max(k, lo), hi)

    # --- optionlet ------------------------------------------------------

    def _optionlet_price(self, option_type: OptionType, strike: float) -> float:  # noqa: PLR0915  (faithful C++ port — one branch per Settings::Strategy, then the integral)
        """# C++ parity: lineartsrpricer.cpp:264-338."""
        if option_type == OptionType.Call and strike >= self._adjusted_upper_bound:
            return 0.0
        if option_type == OptionType.Put and strike <= self._adjusted_lower_bound:
            return 0.0

        section = self._smile_section
        assert section is not None
        assert self._coupon is not None

        # determine lower or upper integration bound (depending on option type)
        lower = strike
        upper = strike
        strategy = self._settings.strategy

        if strategy == Strategy.RateBound:
            if option_type == OptionType.Call:
                upper = self._adjusted_upper_bound
            else:
                lower = self._adjusted_lower_bound
        elif strategy == Strategy.VegaRatio:
            # _strike_from_vega_ratio ensures that the returned strike is on
            # the expected side of ``strike``.
            bound = self._strike_from_vega_ratio(self._settings.vega_ratio, option_type, strike)
            if option_type == OptionType.Call:
                upper = min(bound, self._adjusted_upper_bound)
            else:
                lower = max(bound, self._adjusted_lower_bound)
        elif strategy == Strategy.PriceThreshold:
            # UPSTREAM DEFECT, reproduced deliberately: C++ passes
            # ``settings_.vegaRatio_`` here, NOT ``settings_.priceThreshold_``
            # (lineartsrpricer.cpp:283). ``price_threshold`` is stored and
            # never read. Combined with the guess/bracket defect documented on
            # ``_strike_from_price``, this branch always returns the plain rate
            # bound. Do not "fix" it: C++ v1.43 is the source of truth and the
            # reference JSON pins this exact behaviour.
            bound = self._strike_from_price(self._settings.vega_ratio, option_type, strike)
            if option_type == OptionType.Call:
                upper = min(bound, self._adjusted_upper_bound)
            else:
                lower = max(bound, self._adjusted_lower_bound)
        elif strategy == Strategy.BSStdDevs:
            atm = section.atm_level()
            atm_vol = section.volatility(atm)
            shift = section.shift()
            if section.volatility_type() == VolatilityType.ShiftedLognormal:
                upper_tmp = (atm + shift) * math.exp(
                    self._settings.std_devs * atm_vol - 0.5 * atm_vol * atm_vol * section.exercise_time()
                ) - shift
                lower_tmp = (atm + shift) * math.exp(
                    -self._settings.std_devs * atm_vol - 0.5 * atm_vol * atm_vol * section.exercise_time()
                ) - shift
            else:
                tmp = self._settings.std_devs * atm_vol * math.sqrt(section.exercise_time())
                upper_tmp = atm + tmp
                lower_tmp = atm - tmp
            upper = min(upper_tmp - shift, self._adjusted_upper_bound)
            lower = max(lower_tmp - shift, self._adjusted_lower_bound)
        else:
            qassert.fail(f"Unknown strategy ({int(strategy)})")

        # compute the relevant integral
        result = 0.0
        if upper > lower:
            tmp_bound = min(upper, self._swap_rate_value)
            if tmp_bound > lower:
                result += self._integrator(self._integrand, lower, tmp_bound)
            tmp_bound = max(lower, self._swap_rate_value)
            if upper > tmp_bound:
                result += self._integrator(self._integrand, tmp_bound, upper)
            result *= 1.0 if option_type == OptionType.Call else -1.0

        result += self._singular_terms(option_type, strike)

        return self._annuity * result * self._coupon_discount_ratio * self._coupon.accrual_period()

    # --- CouponPricer interface -----------------------------------------

    def _deflator(self) -> float:
        """``accrualPeriod * discountCurvePaymentDiscount * couponDiscountRatio``.

        The common divisor of every ``*Rate`` member
        (lineartsrpricer.cpp:344-346, 366-369, 383-386).
        """
        assert self._coupon is not None
        return (
            self._coupon.accrual_period()
            * self._discount_curve_payment_discount
            * self._coupon_discount_ratio
        )

    def swaplet_price(self) -> float:
        """# C++ parity: lineartsrpricer.cpp:388-403."""
        assert self._coupon is not None
        assert self._fixing_date is not None
        assert self._today is not None
        if self._fixing_date <= self._today:
            # the fixing is determined
            rs = self._coupon.swap_index().fixing(self._fixing_date)
            return (self._gearing * rs + self._spread) * self._deflator()
        atm_caplet_price = self._optionlet_price(OptionType.Call, self._swap_rate_value)
        atm_floorlet_price = self._optionlet_price(OptionType.Put, self._swap_rate_value)
        return (
            self._gearing
            * (
                self._coupon.accrual_period()
                * self._discount_curve_payment_discount
                * self._swap_rate_value
                * self._coupon_discount_ratio
                + atm_caplet_price
                - atm_floorlet_price
            )
            + self._spread_leg_value
        )

    def swaplet_rate(self) -> float:
        """# C++ parity: lineartsrpricer.cpp:344-347."""
        return self.swaplet_price() / self._deflator()

    def caplet_price(self, effective_cap: float) -> float:
        """# C++ parity: lineartsrpricer.cpp:349-364."""
        assert self._coupon is not None
        assert self._fixing_date is not None
        assert self._today is not None
        if self._fixing_date <= self._today:
            # the fixing is determined — caplet is a call on the fixing
            rs = max(self._coupon.swap_index().fixing(self._fixing_date) - effective_cap, 0.0)
            return (self._gearing * rs) * self._deflator()
        return self._gearing * self._optionlet_price(OptionType.Call, effective_cap)

    def caplet_rate(self, effective_cap: float) -> float:
        """# C++ parity: lineartsrpricer.cpp:366-369."""
        return self.caplet_price(effective_cap) / self._deflator()

    def floorlet_price(self, effective_floor: float) -> float:
        """# C++ parity: lineartsrpricer.cpp:371-381."""
        assert self._coupon is not None
        assert self._fixing_date is not None
        assert self._today is not None
        if self._fixing_date <= self._today:
            # the fixing is determined — floorlet is a put on the fixing
            rs = max(effective_floor - self._coupon.swap_index().fixing(self._fixing_date), 0.0)
            return (self._gearing * rs) * self._deflator()
        return self._gearing * self._optionlet_price(OptionType.Put, effective_floor)

    def floorlet_rate(self, effective_floor: float) -> float:
        """# C++ parity: lineartsrpricer.cpp:383-386."""
        return self.floorlet_price(effective_floor) / self._deflator()


__all__ = [
    "DEFAULT_LOWER_BOUND",
    "DEFAULT_UPPER_BOUND",
    "LinearTsrPricer",
    "PriceHelper",
    "Settings",
    "Strategy",
    "VegaRatioHelper",
]
