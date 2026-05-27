"""Black-style swaption engine (lognormal Black-76 + normal Bachelier).

# C++ parity: ql/pricingengines/swaption/blackswaptionengine.{hpp,cpp}
# (v1.42.1).

C++ uses a templated base ``BlackStyleSwaptionEngine<Spec>`` where the
``Spec`` is one of ``Black76Spec`` (shifted-lognormal) or
``BachelierSpec`` (normal). The concrete classes ``BlackSwaptionEngine``
and ``BachelierSwaptionEngine`` are thin typedef-like subclasses with
their own constructors.

PQuantLib mirrors that with two flavour modes — ``LOGNORMAL`` and
``NORMAL`` — selected via ``volatility_type`` on a single
``BlackStyleSwaptionEngine`` class. The user-facing ``BlackSwaptionEngine``
and ``BachelierSwaptionEngine`` are thin classes that fix the
volatility-type at construction.

Divergences from C++:

- ``SwaptionVolatilityStructure``-driven constructors are deferred —
  no L4-E test path needs the term/strike-structure variant. Only the
  flat-volatility constructors (float and ``Quote``) are exposed.
- C++ supports two cash-annuity models (``DiscountCurve`` /
  ``SwapRate``). We only port ``DiscountCurve`` (the default and the
  only one the L4-E tests exercise). The ``model`` enum is preserved
  for forward compatibility, defaulting to ``DiscountCurve``.
- C++ writes a number of fields into ``additionalResults`` (annuity,
  stdDev, vega, delta, impliedVolatility, ...). PQuantLib writes the
  same keys so calibration helpers' implied-vol Brent solvers work.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    bachelier_black_formula_std_dev_derivative,
    black_formula,
    black_formula_std_dev_derivative,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class CashAnnuityModel(IntEnum):
    """Annuity-computation mode for cash-settled swaptions.

    # C++ parity: ``BlackStyleSwaptionEngine::CashAnnuityModel`` in
    # blackswaptionengine.hpp:56 (v1.42.1).
    """

    SwapRate = 0
    DiscountCurve = 1


class BlackStyleSwaptionEngine(GenericEngine[SwaptionArguments, SwaptionResults]):
    """Templated Black-style swaption engine (lognormal or normal).

    # C++ parity: ``detail::BlackStyleSwaptionEngine`` in
    # blackswaptionengine.hpp:53-78 (v1.42.1).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        day_counter: DayCounter | None = None,
        displacement: float = 0.0,
        model: CashAnnuityModel = CashAnnuityModel.DiscountCurve,
    ) -> None:
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._discount_curve: YieldTermStructureProtocol = discount_curve
        # Store the vol as a callable returning a scalar.
        self._vol: float | Quote = vol
        self._volatility_type: VolatilityType = volatility_type
        self._day_counter: DayCounter = (
            day_counter if day_counter is not None else Actual365Fixed()
        )
        self._displacement: float = displacement
        self._model: CashAnnuityModel = model

    # --- helpers ------------------------------------------------------

    def _vol_value(self) -> float:
        if isinstance(self._vol, Quote):
            return self._vol.value()
        return float(self._vol)

    @property
    def volatility_type(self) -> VolatilityType:
        return self._volatility_type

    @property
    def displacement(self) -> float:
        return self._displacement

    # --- engine -------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 (one-shot port of C++ template body)
        # # C++ parity: blackswaptionengine.hpp:220-327 (v1.42.1).
        # Basis-point factor used to convert leg-BPS to annuity.
        basis_point = 1.0e-4

        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type().name == "European",
            "not a European option",
        )
        qassert.require(args.swap is not None, "swap not set")
        swap = args.swap
        assert swap is not None

        # Re-price the swap on the engine's discount curve. PQuantLib
        # has no observer-pause primitive (C++ uses
        # ObservableSettings::disableUpdates); the swap rebuilds its
        # cache when set_pricing_engine is called.
        swap.set_pricing_engine(DiscountingSwapEngine(self._discount_curve))

        fixed_leg = swap.fixed_leg()
        first_coupon = fixed_leg[0]
        exercise_date = args.exercise.date(0)
        # C++ asserts swap_start >= exercise_date — accrual_start_date
        # of the first fixed coupon.
        from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon  # noqa: PLC0415

        assert isinstance(first_coupon, FixedRateCoupon)
        qassert.require(
            first_coupon.accrual_start_date() >= exercise_date,
            f"swap start ({first_coupon.accrual_start_date()}) before "
            f"exercise date ({exercise_date}) not supported in Black swaption engine",
        )

        strike = swap.fixed_rate()
        atm_forward = swap.fair_rate()

        # Volatilities are quoted for zero-spreaded swaps; adjust if
        # the swap has a floating-leg spread (matches C++).
        spread = swap.spread()
        if spread != 0.0:
            correction = spread * abs(swap.floating_leg_bps() / swap.fixed_leg_bps())
            strike -= correction
            atm_forward -= correction
            results.additional_results["spreadCorrection"] = correction
        else:
            results.additional_results["spreadCorrection"] = 0.0
        results.additional_results["strike"] = strike
        results.additional_results["atmForward"] = atm_forward

        # Annuity computation — Physical or Cash/CollateralizedCashPrice
        # both use |fixed_leg_bps| / basis_point.
        if args.settlement_type == SettlementType.Physical or (
            args.settlement_type == SettlementType.Cash
            and args.settlement_method == SettlementMethod.CollateralizedCashPrice
        ):
            annuity = abs(swap.fixed_leg_bps()) / basis_point
        else:
            # Cash / ParYieldCurve — C++ uses the cash-BPS approximation
            # via CashFlows::bps under an InterestRate. We defer that
            # path (no L4-E test exercises Cash settlement) but raise
            # a clear error rather than silently producing zero.
            qassert.fail(
                "Cash/ParYieldCurve settlement not yet ported in BlackSwaptionEngine"
            )
        results.additional_results["annuity"] = annuity

        # Compute the swap "length" = year_fraction(floating_start, floating_end).
        # C++ uses ``vol_->swapLength(...)`` which rounds to whole months;
        # we use the day_counter directly — close enough for analytic
        # pricing under a flat vol (the variance is just vol^2 * time
        # so the integer-month rounding doesn't change the answer).
        floating_schedule = swap.floating_schedule()
        floating_start = floating_schedule.date(0)
        floating_end = floating_schedule.date(len(floating_schedule) - 1)
        swap_length = self._day_counter.year_fraction(floating_start, floating_end)
        swap_length = max(swap_length, 1.0 / 12.0)
        results.additional_results["swapLength"] = swap_length

        # Total variance + std dev = vol^2 * t.
        exercise_time = self._day_counter.year_fraction(
            self._discount_curve.reference_date(), exercise_date
        )
        sigma = self._vol_value()
        std_dev = sigma * math.sqrt(exercise_time)
        results.additional_results["stdDev"] = std_dev

        # Effective displacement: ShiftedLognormal uses the configured
        # shift; Normal forces it to 0.0.
        eff_disp = (
            self._displacement
            if self._volatility_type == VolatilityType.ShiftedLognormal
            else 0.0
        )

        # Swap type → Call (Payer) or Put (Receiver).
        w = OptionType.Call if swap.swap_type() == SwapType.Payer else OptionType.Put

        # Compute the value via the appropriate Black formula.
        if self._volatility_type == VolatilityType.ShiftedLognormal:
            value = black_formula(w, strike, atm_forward, std_dev, annuity, eff_disp)
            vega_per_stddev = black_formula_std_dev_derivative(
                strike, atm_forward, std_dev, annuity, eff_disp
            )
        elif self._volatility_type == VolatilityType.Normal:
            value = bachelier_black_formula(w, strike, atm_forward, std_dev, annuity)
            vega_per_stddev = bachelier_black_formula_std_dev_derivative(
                strike, atm_forward, std_dev, annuity
            )
        else:
            qassert.fail(f"unknown VolatilityType ({self._volatility_type})")

        results.value = value
        # vega = dV/dstdDev * sqrt(t); C++ matches.
        results.additional_results["vega"] = vega_per_stddev * math.sqrt(exercise_time)
        results.additional_results["timeToExpiry"] = exercise_time
        results.additional_results["impliedVolatility"] = (
            std_dev / math.sqrt(exercise_time) if exercise_time > 0.0 else 0.0
        )
        df_exercise = self._discount_curve.discount(exercise_date)
        if df_exercise > 0.0:
            results.additional_results["forwardPrice"] = value / df_exercise


class BlackSwaptionEngine(BlackStyleSwaptionEngine):
    """Shifted-lognormal (Black 76) swaption engine.

    # C++ parity: ``class BlackSwaptionEngine`` in
    # blackswaptionengine.hpp:136-152 (v1.42.1).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote,
        day_counter: DayCounter | None = None,
        displacement: float = 0.0,
        model: CashAnnuityModel = CashAnnuityModel.DiscountCurve,
    ) -> None:
        super().__init__(
            discount_curve,
            vol,
            volatility_type=VolatilityType.ShiftedLognormal,
            day_counter=day_counter,
            displacement=displacement,
            model=model,
        )


class BachelierSwaptionEngine(BlackStyleSwaptionEngine):
    """Normal (Bachelier) swaption engine.

    # C++ parity: ``class BachelierSwaptionEngine`` in
    # blackswaptionengine.hpp:161-175 (v1.42.1).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote,
        day_counter: DayCounter | None = None,
        model: CashAnnuityModel = CashAnnuityModel.DiscountCurve,
    ) -> None:
        super().__init__(
            discount_curve,
            vol,
            volatility_type=VolatilityType.Normal,
            day_counter=day_counter,
            displacement=0.0,
            model=model,
        )


__all__ = [
    "BachelierSwaptionEngine",
    "BlackStyleSwaptionEngine",
    "BlackSwaptionEngine",
    "CashAnnuityModel",
]
