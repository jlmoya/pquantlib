"""Base class for engines that can generate a calibration basket.

# C++ parity: ql/pricingengines/swaption/basketgeneratingengine.{hpp,cpp}
# @ v1.43 (6b57206e0).

The Gaussian1d non-standard / float-float swaption engines derive from
this base so that an exotic Bermudan can produce a basket of standard
European swaptions to calibrate the model against. Two flavours:

``Naive``
    One at-the-money :class:`~pquantlib.models.swaption_helper.SwaptionHelper`
    per alive exercise date, running to ``underlying_last_date()``.

``MaturityStrikeByDeltaGamma``
    A Levenberg-Marquardt fit of ``(nominal, maturity, rate)`` so that the
    standard swap matches the exotic's NPV, delta and gamma in the model
    state ``y`` at ``y = 0``. The residual vector is supplied by
    :class:`MatchHelper`.

Warnings carried over from the C++ header:

* generated calibrating swaptions have their strike floored at 0.1bp
  (minus the lognormal shift, if any); this is not true of the ATM
  swaptions, whose strike is generated inside the swaption helper;
* ``standard_swap_base`` must have associated forward and discount
  curves — the market price of the calibration instrument is computed
  with them, so the model price must use the same ones.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.end_criteria import EndCriteria
from pquantlib.math.optimization.end_criteria import Type as EndCriteriaType
from pquantlib.math.optimization.levenberg_marquardt import LevenbergMarquardt
from pquantlib.math.optimization.problem import Problem
from pquantlib.models.calibration_helper import CalibrationErrorType
from pquantlib.models.swaption_helper import SwaptionHelper
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.exercise import Exercise
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.instruments.swap import SwapType
    from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
        SwaptionVolatilityStructure,
    )

_FD_STEP: float = 0.0001
"""# C++ parity: basketgeneratingengine.cpp:111 — ``const Real h = 0.0001``,
the finite-difference step in the model state ``y``."""

_STRIKE_FLOOR: float = 0.00001
"""# C++ parity: basketgeneratingengine.cpp:209-210 — floor the calibrated
strike at 0.1bp minus the smile's shift."""

_NOMINAL_FLOOR: float = 0.000001
"""# C++ parity: basketgeneratingengine.cpp:213-214 — floor the calibrated
nominal at 0.01bp (the comment in C++ says "float at 0.01bp")."""

_MONTHS_PER_YEAR: int = 12
_INITIAL_GUESS_SIZE: int = 3


class CalibrationBasketType(IntEnum):
    """Which basket-generation algorithm to run.

    # C++ parity: ``BasketGeneratingEngine::CalibrationBasketType`` in
    # basketgeneratingengine.hpp:61-64 (v1.43).
    """

    Naive = 0
    MaturityStrikeByDeltaGamma = 1


class MatchHelper(CostFunction):
    """Residuals for matching a standard swap's (NPV, delta, gamma).

    # C++ parity: ``BasketGeneratingEngine::MatchHelper`` — a private
    # nested class declared in basketgeneratingengine.hpp:101-231 (v1.43),
    # deriving from ``CostFunction``.

    The free vector is ``v = (nominal, maturity_in_years, fixed_rate)``:

    * ``|v[0]|`` is the nominal and its SIGN flips the payer/receiver type;
    * ``|v[1]|`` is capped at ``max_maturity`` and split into whole years +
      whole months, with the fractional month handled by interpolating
      linearly (weight ``alpha``) between the lower- and upper-month swap;
      a maturity rounding to zero is bumped to one month with ``alpha = 1``
      so that only the lower swap is looked at;
    * ``v[2]`` is the fixed rate, explicitly allowed to be negative.

    The three residuals are ``(npv - npv_) / delta_``, ``(delta - delta_) /
    delta_`` and ``(gamma - gamma_) / gamma_`` — note the FIRST one is
    divided by ``delta_``, not by ``npv_``.
    """

    def __init__(
        self,
        swap_type: SwapType,
        npv: float,
        delta: float,
        gamma: float,
        model: Gaussian1dModel,
        index_base: SwapIndex,
        expiry: Date,
        max_maturity: float,
        h: float,
    ) -> None:
        # # C++ parity: basketgeneratingengine.hpp:103-114.
        self._type: int = int(swap_type)
        self._mdl: Gaussian1dModel = model
        self._index_base: SwapIndex = index_base
        self._expiry: Date = expiry
        self._max_maturity: float = max_maturity
        self._npv: float = npv
        self._delta: float = delta
        self._gamma: float = gamma
        self._h: float = h

    def npv(
        self, swap: object, fixed_rate: float, nominal: float, y: float, type_: int
    ) -> float:
        """NPV of ``swap`` at model state ``y``, seen from ``expiry``.

        # C++ parity: ``MatchHelper::NPV`` (basketgeneratingengine.hpp:116-140).
        """
        discount_ts = self._index_base.discounting_term_structure()
        value = 0.0
        for cf in swap.fixed_leg():  # type: ignore[attr-defined]
            assert isinstance(cf, FixedRateCoupon)
            value -= (
                fixed_rate
                * cf.accrual_period()
                * nominal
                * self._mdl.zerobond_date(cf.date(), self._expiry, y, discount_ts)
            )
        for cf in swap.floating_leg():  # type: ignore[attr-defined]
            assert isinstance(cf, IborCoupon)
            ibor_index = cf.ibor_index()
            assert isinstance(ibor_index, IborIndex)
            value += (
                self._mdl.forward_rate(cf.fixing_date(), self._expiry, y, ibor_index)
                * cf.accrual_period()
                * nominal
                * self._mdl.zerobond_date(cf.date(), self._expiry, y, discount_ts)
            )
        return float(type_) * value

    def value(self, x: npt.NDArray[np.float64]) -> float:
        """# C++ parity: basketgeneratingengine.hpp:142-149 — RMS of ``values``."""
        vals = self.values(x)
        return float(math.sqrt(float(np.sum(vals * vals)) / vals.size))

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # # C++ parity: basketgeneratingengine.hpp:151-223.
        # 1 means payer, -1 receiver — same as the exotic underlying.
        type_ = self._type
        nominal = abs(float(x[0]))
        if x[0] < 0.0:
            type_ *= -1
        maturity = min(abs(float(x[1])), self._max_maturity)

        fixed_rate = float(x[2])  # negative rates allowed explicitly
        years = math.floor(maturity)
        maturity -= float(years)
        maturity *= float(_MONTHS_PER_YEAR)
        months = math.floor(maturity)
        alpha = 1.0 - (maturity - float(months))
        if years == 0 and months == 0:
            months = 1  # ensure a maturity of at least one month ...
            alpha = 1.0  # ... but then look only at the lower-maturity swap

        lower_period = Period(years, TimeUnit.Years) + Period(months, TimeUnit.Months)
        upper_period = lower_period + Period(1, TimeUnit.Months)
        swap_lower = self._index_base.clone_with_tenor(lower_period).underlying_swap(
            self._expiry
        )
        swap_upper = self._index_base.clone_with_tenor(upper_period).underlying_swap(
            self._expiry
        )

        h = self._h
        npvm = alpha * self.npv(swap_lower, fixed_rate, nominal, -h, type_) + (
            1.0 - alpha
        ) * self.npv(swap_upper, fixed_rate, nominal, -h, type_)
        npv = alpha * self.npv(swap_lower, fixed_rate, nominal, 0.0, type_) + (
            1.0 - alpha
        ) * self.npv(swap_upper, fixed_rate, nominal, 0.0, type_)
        npvu = alpha * self.npv(swap_lower, fixed_rate, nominal, h, type_) + (
            1.0 - alpha
        ) * self.npv(swap_upper, fixed_rate, nominal, h, type_)
        delta = (npvu - npvm) / (2.0 * h)
        gamma = (npvu - 2.0 * npv + npvm) / (h * h)

        return np.array(
            [
                (npv - self._npv) / self._delta,
                (delta - self._delta) / self._delta,
                (gamma - self._gamma) / self._gamma,
            ],
            dtype=np.float64,
        )


class BasketGeneratingEngine(ABC):
    """Mixin for engines that can produce a swaption calibration basket.

    # C++ parity: ``class BasketGeneratingEngine`` in
    # basketgeneratingengine.hpp:57-232 (v1.43).
    """

    CalibrationBasketType = CalibrationBasketType

    def __init__(
        self,
        model: Gaussian1dModel,
        oas: Quote | None = None,
        discount_curve: YieldTermStructureProtocol | None = None,
    ) -> None:
        # # C++ parity: basketgeneratingengine.hpp:75-84 (both protected ctors).
        self._onefactormodel: Gaussian1dModel = model
        self._oas: Quote | None = oas
        self._basket_discount_curve: YieldTermStructureProtocol | None = discount_curve

    # --- hooks the deriving engine must supply ------------------------

    @abstractmethod
    def underlying_npv(self, expiry: Date, y: float) -> float:
        """# C++ parity: ``virtual Real underlyingNpv(const Date&, Real) const``."""

    @abstractmethod
    def underlying_type(self) -> SwapType:
        """# C++ parity: ``virtual Swap::Type underlyingType() const``."""

    @abstractmethod
    def underlying_last_date(self) -> Date:
        """# C++ parity: ``virtual const Date underlyingLastDate() const``."""

    @abstractmethod
    def initial_guess(self, expiry: Date) -> npt.NDArray[np.float64]:
        """Return ``(nominal, maturity, rate)``.

        # C++ parity: ``virtual const Array initialGuess(const Date&) const``.
        """

    # --- basket generation --------------------------------------------

    def calibration_basket(  # noqa: PLR0915 (one-shot port of the C++ switch)
        self,
        exercise: Exercise,
        standard_swap_base: SwapIndex,
        swaption_volatility: SwaptionVolatilityStructure,
        basket_type: CalibrationBasketType = CalibrationBasketType.MaturityStrikeByDeltaGamma,
    ) -> list[SwaptionHelper]:
        """Build one calibration helper per alive exercise date.

        # C++ parity: ``BasketGeneratingEngine::calibrationBasket`` in
        # basketgeneratingengine.cpp:34-243 (v1.43).
        """
        qassert.require(
            standard_swap_base.forwarding_term_structure() is not None,
            "standard swap base forwarding term structure must not be empty.",
        )
        qassert.require(
            not standard_swap_base.exogenous_discount()
            or standard_swap_base.discounting_term_structure() is not None,
            "standard swap base discounting term structure must not be empty.",
        )

        result: list[SwaptionHelper] = []
        today = ObservableSettings().evaluation_date
        dates = exercise.dates()
        min_idx_alive = 0
        while min_idx_alive < len(dates) and dates[min_idx_alive] <= today:
            min_idx_alive += 1

        # RebatedExercise is not ported; C++ dynamic_pointer_casts to it and
        # falls back to rebate = 0 / rebateDate = expiry when the cast fails,
        # which is the branch every non-rebated exercise takes.
        rebate = 0.0

        helper_curve = (
            standard_swap_base.discounting_term_structure()
            if standard_swap_base.exogenous_discount()
            else standard_swap_base.forwarding_term_structure()
        )
        assert helper_curve is not None
        ibor_index = standard_swap_base.ibor_index()

        for i in range(min_idx_alive, len(dates)):
            expiry = exercise.date(i)
            rebate_date = expiry

            if basket_type == CalibrationBasketType.Naive:
                # # C++ parity: basketgeneratingengine.cpp:74-104.
                swap_length = swaption_volatility.day_counter().year_fraction(
                    standard_swap_base.value_date(expiry), self.underlying_last_date()
                )
                tenor = Period(
                    int(_lround(swap_length * float(_MONTHS_PER_YEAR))), TimeUnit.Months
                )
                sec = swaption_volatility.smile_section(expiry, tenor, True)
                atm_strike = sec.atm_level()
                # C++ tests ``atmStrike == Null<Real>()``; PQuantLib's
                # SmileSection uses NaN as the "no ATM level" sentinel.
                atm_vol = (
                    sec.volatility(0.03)
                    if math.isnan(atm_strike)
                    else sec.volatility(atm_strike)
                )
                shift = sec.shift()
                helper = SwaptionHelper(
                    expiry,
                    self.underlying_last_date(),
                    SimpleQuote(atm_vol),
                    ibor_index,
                    standard_swap_base.fixed_leg_tenor(),
                    standard_swap_base.day_counter(),
                    ibor_index.day_counter(),
                    helper_curve,
                    CalibrationErrorType.RelativePriceError,
                    None,
                    1.0,
                    swaption_volatility.volatility_type(),
                    shift,
                )
            elif basket_type == CalibrationBasketType.MaturityStrikeByDeltaGamma:
                # # C++ parity: basketgeneratingengine.cpp:106-232.
                h = _FD_STEP
                model = self._onefactormodel
                z_spread_dsc = (
                    1.0
                    if self._oas is None
                    else math.exp(
                        -self._oas.value()
                        * model.term_structure.day_counter().year_fraction(
                            expiry, rebate_date
                        )
                    )
                )

                npvm = (
                    self.underlying_npv(expiry, -h)
                    + rebate
                    * model.zerobond_date(
                        rebate_date, expiry, -h, self._basket_discount_curve
                    )
                    * z_spread_dsc
                )
                npv = (
                    self.underlying_npv(expiry, 0.0)
                    + rebate
                    * model.zerobond_date(
                        rebate_date, expiry, 0.0, self._basket_discount_curve
                    )
                    * z_spread_dsc
                )
                npvp = (
                    self.underlying_npv(expiry, h)
                    + rebate
                    * model.zerobond_date(
                        rebate_date, expiry, h, self._basket_discount_curve
                    )
                    * z_spread_dsc
                )

                delta = (npvp - npvm) / (2.0 * h)
                gamma = (npvp - 2.0 * npv + npvm) / (h * h)
                qassert.require(
                    npv * npv + delta * delta + gamma * gamma > 0.0,
                    "(npv,delta,gamma) must have a positive norm",
                )

                # Restrict the maximum maturity so it fits inside Date's range.
                max_maturity = swaption_volatility.day_counter().year_fraction(
                    expiry, Date.max_date() - 365
                )

                match_helper = MatchHelper(
                    self.underlying_type(),
                    npv,
                    delta,
                    gamma,
                    model,
                    standard_swap_base,
                    expiry,
                    max_maturity,
                    h,
                )

                initial = self.initial_guess(expiry)
                qassert.require(
                    initial.size == _INITIAL_GUESS_SIZE,
                    f"initial guess must have size 3 (but is {initial.size})",
                )

                # # C++ parity: basketgeneratingengine.cpp:173-186 —
                # # EndCriteria(1000, 200, 1e-8, 1e-8, 1e-8), NoConstraint,
                # # LevenbergMarquardt with default (epsfcn, xtol, gtol).
                end_criteria = EndCriteria(1000, 200, 1e-8, 1e-8, 1e-8)
                problem = Problem(match_helper, NoConstraint(), initial)
                ret = LevenbergMarquardt().minimize(problem, end_criteria)
                qassert.require(
                    ret
                    not in (
                        EndCriteriaType.None_,
                        EndCriteriaType.Unknown,
                        EndCriteriaType.MaxIterations,
                    ),
                    f"optimizer returns error ({ret})",
                )
                solution = problem.current_value.copy()

                maturity = abs(float(solution[1]))
                years = math.floor(maturity)
                maturity -= float(years)
                maturity *= float(_MONTHS_PER_YEAR)
                # NOTE: round-half-UP here, unlike MatchHelper.values which
                # floors. C++ uses floor(maturity + 0.5) at cpp:194.
                months = math.floor(maturity + 0.5)
                if years == 0 and months == 0:
                    months = 1  # ensure a maturity of at least one month
                mat_period = Period(years, TimeUnit.Years) + Period(
                    months, TimeUnit.Months
                )

                sec = swaption_volatility.smile_section(expiry, mat_period, True)
                shift = sec.shift()
                # Floor the strike (see the header warning) and the nominal.
                solution[2] = max(float(solution[2]), _STRIKE_FLOOR - shift)
                solution[0] = max(float(solution[0]), _NOMINAL_FLOOR)
                vol = sec.volatility(float(solution[2]))

                helper = SwaptionHelper(
                    expiry,
                    mat_period,
                    SimpleQuote(vol),
                    ibor_index,
                    standard_swap_base.fixed_leg_tenor(),
                    standard_swap_base.day_counter(),
                    ibor_index.day_counter(),
                    helper_curve,
                    CalibrationErrorType.RelativePriceError,
                    float(solution[2]),
                    abs(float(solution[0])),
                    swaption_volatility.volatility_type(),
                    shift,
                )
            else:
                qassert.fail(f"Calibration basket type not known ({basket_type})")
                raise AssertionError  # unreachable, satisfies the type checker

            result.append(helper)

        return result


def _lround(x: float) -> int:
    """``std::lround`` — round half AWAY FROM ZERO, not banker's rounding.

    # C++ parity: basketgeneratingengine.cpp:80 uses ``std::lround``;
    # Python's built-in ``round`` is round-half-to-even and would differ
    # at exact .5 boundaries.
    """
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


__all__ = ["BasketGeneratingEngine", "CalibrationBasketType", "MatchHelper"]
