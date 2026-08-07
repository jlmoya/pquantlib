"""Black-style swaption engine (lognormal Black-76 + normal Bachelier).

# C++ parity: ql/pricingengines/swaption/blackswaptionengine.{hpp,cpp}
# @ v1.43 (6b57206e0).

C++ uses a templated base ``detail::BlackStyleSwaptionEngine<Spec>``
where ``Spec`` is one of ``detail::Black76Spec`` (shifted-lognormal) or
``detail::BachelierSpec`` (normal). The two Specs are not template tags:
each carries a ``VolatilityType`` and three functions —
``value``/``vega``/``delta`` — and the concrete engines
``BlackSwaptionEngine`` / ``BachelierSwaptionEngine`` differ ONLY by
which Spec they instantiate.

PQuantLib ports the Specs as real classes (:class:`Black76Spec`,
:class:`BachelierSpec`) and passes an instance to
:class:`BlackStyleSwaptionEngine`, which is the closest Python analogue
of the C++ template parameter.

Note the deliberate asymmetry in the C++ signatures, reproduced here:
``BachelierSpec::value/vega/delta`` take a trailing unnamed ``Real``
displacement argument and IGNORE it. Passing a non-zero displacement to
:class:`BachelierSpec` therefore changes nothing, exactly as in C++.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, Final

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exercise import Exercise
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SettlementType,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.interest_rate import InterestRate
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    bachelier_black_formula_forward_derivative,
    bachelier_black_formula_std_dev_derivative,
    black_formula,
    black_formula_forward_derivative,
    black_formula_std_dev_derivative,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    ConstantSwaptionVolatility,
)
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol

_BASIS_POINT: Final[float] = 1.0e-4
"""# C++ parity: blackswaptionengine.hpp:222 — ``static const Spread
basisPoint = 1.0e-4``."""

_MIN_SWAP_LENGTH: Final[float] = 1.0 / 12.0
"""# C++ parity: blackswaptionengine.hpp:305 — swapLength is floored at one
month so a variance and a shift can always be read off the vol surface."""


class Black76Spec:
    """Shifted-lognormal policy for :class:`BlackStyleSwaptionEngine`.

    # C++ parity: ``struct detail::Black76Spec`` in
    # blackswaptionengine.hpp:81-102 (v1.43).
    """

    type: Final[VolatilityType] = VolatilityType.ShiftedLognormal

    def value(
        self,
        option_type: OptionType,
        strike: float,
        atm_forward: float,
        std_dev: float,
        annuity: float,
        displacement: float,
    ) -> float:
        """# C++ parity: blackswaptionengine.hpp:83-88 — ``blackFormula``."""
        return black_formula(
            option_type, strike, atm_forward, std_dev, annuity, displacement
        )

    def vega(
        self,
        strike: float,
        atm_forward: float,
        std_dev: float,
        exercise_time: float,
        annuity: float,
        displacement: float,
    ) -> float:
        """# C++ parity: blackswaptionengine.hpp:89-95 —
        # ``sqrt(exerciseTime) * blackFormulaStdDevDerivative(...)``.
        """
        return math.sqrt(exercise_time) * black_formula_std_dev_derivative(
            strike, atm_forward, std_dev, annuity, displacement
        )

    def delta(
        self,
        option_type: OptionType,
        strike: float,
        atm_forward: float,
        std_dev: float,
        annuity: float,
        displacement: float,
    ) -> float:
        """# C++ parity: blackswaptionengine.hpp:96-101 —
        # ``blackFormulaForwardDerivative``.
        """
        return black_formula_forward_derivative(
            option_type, strike, atm_forward, std_dev, annuity, displacement
        )


class BachelierSpec:
    """Normal policy for :class:`BlackStyleSwaptionEngine`.

    # C++ parity: ``struct detail::BachelierSpec`` in
    # blackswaptionengine.hpp:105-125 (v1.43).

    Every method takes a trailing displacement argument that C++ leaves
    unnamed and never uses; it is accepted and discarded here for the
    same reason — the two Specs must be interchangeable.
    """

    type: Final[VolatilityType] = VolatilityType.Normal

    def value(
        self,
        option_type: OptionType,
        strike: float,
        atm_forward: float,
        std_dev: float,
        annuity: float,
        displacement: float,  # C++ leaves this parameter unnamed; ignored on purpose
    ) -> float:
        """# C++ parity: blackswaptionengine.hpp:107-112 —
        # ``bachelierBlackFormula`` (displacement ignored).
        """
        _ = displacement
        return bachelier_black_formula(
            option_type, strike, atm_forward, std_dev, annuity
        )

    def vega(
        self,
        strike: float,
        atm_forward: float,
        std_dev: float,
        exercise_time: float,
        annuity: float,
        displacement: float,  # C++ leaves this parameter unnamed; ignored on purpose
    ) -> float:
        """# C++ parity: blackswaptionengine.hpp:113-118 (displacement ignored)."""
        _ = displacement
        return math.sqrt(exercise_time) * bachelier_black_formula_std_dev_derivative(
            strike, atm_forward, std_dev, annuity
        )

    def delta(
        self,
        option_type: OptionType,
        strike: float,
        atm_forward: float,
        std_dev: float,
        annuity: float,
        displacement: float,  # C++ leaves this parameter unnamed; ignored on purpose
    ) -> float:
        """# C++ parity: blackswaptionengine.hpp:119-124 (displacement ignored)."""
        _ = displacement
        return bachelier_black_formula_forward_derivative(
            option_type, strike, atm_forward, std_dev, annuity
        )


class CashAnnuityModel(IntEnum):
    """Annuity-computation mode for cash-settled swaptions.

    # C++ parity: ``BlackStyleSwaptionEngine::CashAnnuityModel`` in
    # blackswaptionengine.hpp:56 (v1.43).
    """

    SwapRate = 0
    DiscountCurve = 1


class BlackStyleSwaptionEngine(GenericEngine[SwaptionArguments, SwaptionResults]):
    """Black-style swaption engine parameterised by a volatility Spec.

    # C++ parity: ``detail::BlackStyleSwaptionEngine<Spec>`` in
    # blackswaptionengine.hpp:53-78 + 181-327 (v1.43).

    ``vol`` accepts the three C++ constructor shapes:

    * a ``float`` volatility — wrapped in a ``ConstantSwaptionVolatility``
      built with ``(0, NullCalendar(), Following, vol, dc, Spec.type,
      displacement)``, exactly as C++ does;
    * a :class:`~pquantlib.quotes.quote.Quote` — likewise;
    * a :class:`SwaptionVolatilityStructure` — used directly, in which case
      ``day_counter`` and ``displacement`` are ignored (they belong to the
      surface) and must be left at their defaults.
    """

    def __init__(
        self,
        spec: Black76Spec | BachelierSpec,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote | SwaptionVolatilityStructure,
        day_counter: DayCounter | None = None,
        displacement: float = 0.0,
        model: CashAnnuityModel = CashAnnuityModel.DiscountCurve,
    ) -> None:
        # # C++ parity: blackswaptionengine.hpp:181-218 — three ctors.
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._spec: Black76Spec | BachelierSpec = spec
        self._discount_curve: YieldTermStructureProtocol = discount_curve
        self._model: CashAnnuityModel = model

        if isinstance(vol, SwaptionVolatilityStructure):
            qassert.require(
                day_counter is None and displacement == 0.0,
                "day_counter / displacement belong to the volatility structure; "
                "do not pass them alongside a SwaptionVolatilityStructure",
            )
            self._vol: SwaptionVolatilityStructure = vol
        else:
            dc: DayCounter = day_counter if day_counter is not None else Actual365Fixed()
            self._vol = ConstantSwaptionVolatility(
                settlement_days=0,
                calendar=NullCalendar(),
                business_day_convention=BusinessDayConvention.Following,
                volatility=vol,
                day_counter=dc,
                volatility_type=spec.type,
                shift=displacement,
            )

    # --- inspectors ---------------------------------------------------

    def term_structure(self) -> YieldTermStructureProtocol:
        """# C++ parity: ``BlackStyleSwaptionEngine::termStructure()``."""
        return self._discount_curve

    def volatility(self) -> SwaptionVolatilityStructure:
        """# C++ parity: ``BlackStyleSwaptionEngine::volatility()``."""
        return self._vol

    @property
    def volatility_type(self) -> VolatilityType:
        return self._vol.volatility_type()

    # --- engine -------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 (one-shot port of the C++ template body)
        # # C++ parity: blackswaptionengine.hpp:220-327 (v1.43).
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not a European option",
        )
        exercise_date = args.exercise.date(0)

        qassert.require(args.swap is not None, "swap not set")
        swap = args.swap
        assert swap is not None

        fixed_leg = swap.fixed_leg()
        first_coupon = fixed_leg[0]
        from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon  # noqa: PLC0415

        assert isinstance(first_coupon, FixedRateCoupon)
        qassert.require(
            first_coupon.accrual_start_date() >= exercise_date,
            f"swap start ({first_coupon.accrual_start_date()}) before exercise date "
            f"({exercise_date}) not supported in Black swaption engine",
        )

        strike = swap.fixed_rate()

        # Re-price the underlying on the ENGINE's discount curve — the
        # index's forwarding curve may well be a different one.
        # # C++ parity: blackswaptionengine.hpp:246-249. PQuantLib has no
        # observer-pause primitive (C++ brackets this with
        # ObservableSettings::disable/enableUpdates purely to avoid
        # notifying the swaption); the swap simply recalculates.
        swap.set_pricing_engine(DiscountingSwapEngine(self._discount_curve))
        valuation_date = swap.valuation_date()
        results.valuation_date = valuation_date
        atm_forward = swap.fair_rate()

        # Volatilities are quoted for zero-spreaded swaps; any floating-leg
        # spread is removed with a matching correction on the fixed leg.
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

        if args.settlement_type == SettlementType.Physical or (
            args.settlement_type == SettlementType.Cash
            and args.settlement_method == SettlementMethod.CollateralizedCashPrice
        ):
            annuity = abs(swap.fixed_leg_bps()) / _BASIS_POINT
        elif (
            args.settlement_type == SettlementType.Cash
            and args.settlement_method == SettlementMethod.ParYieldCurve
        ):
            # # C++ parity: blackswaptionengine.hpp:275-293.
            day_count = first_coupon.day_counter()
            # C++ assumes the cash-settlement date equals the swap start date.
            discount_date = (
                first_coupon.accrual_start_date()
                if self._model == CashAnnuityModel.DiscountCurve
                else valuation_date
            )
            freq = Frequency.Annual
            fixed_schedule = swap.fixed_schedule()
            if fixed_schedule.has_tenor():
                freq = fixed_schedule.tenor.frequency()
            rate = InterestRate(atm_forward, day_count, Compounding.Compounded, freq)
            # # C++ parity: CashFlows::bps(leg, InterestRate, ...) at
            # # cashflows.cpp:870-890 builds
            # # ``FlatForward(settlementDate, y.rate(), y.dayCounter(),
            # #               y.compounding(), y.frequency())``
            # # and delegates to the YieldTermStructure overload with
            # # settlementDate == npvDate == discountDate.
            flat_rate = FlatForward.from_rate(
                discount_date,
                rate.rate(),
                rate.day_counter(),
                rate.compounding(),
                rate.frequency(),
            )
            fixed_leg_cash_bps = CashFlows.bps(
                fixed_leg,
                flat_rate,
                False,
                discount_date,
                discount_date,
            )
            annuity = abs(fixed_leg_cash_bps / _BASIS_POINT) * self._discount_curve.discount(
                discount_date
            )
        else:
            qassert.fail("invalid (settlementType, settlementMethod) pair")
            raise AssertionError  # unreachable, satisfies the type checker
        results.additional_results["annuity"] = annuity

        # swapLength is the vol structure's whole-month rounding of the
        # floating schedule's span, NOT a day-count year fraction.
        floating_schedule = swap.floating_schedule()
        swap_length = self._vol.swap_length(
            floating_schedule.date(0), floating_schedule.date(len(floating_schedule) - 1)
        )
        swap_length = max(swap_length, _MIN_SWAP_LENGTH)
        results.additional_results["swapLength"] = swap_length

        variance = self._vol.black_variance(exercise_date, swap_length, strike)
        displacement = (
            self._vol.shift(exercise_date, swap_length)
            if self._vol.volatility_type() == VolatilityType.ShiftedLognormal
            else 0.0
        )

        std_dev = math.sqrt(variance)
        results.additional_results["stdDev"] = std_dev
        w = OptionType.Call if swap.swap_type() == SwapType.Payer else OptionType.Put
        results.value = self._spec.value(
            w, strike, atm_forward, std_dev, annuity, displacement
        )

        exercise_time = self._vol.time_from_reference(exercise_date)
        results.additional_results["vega"] = self._spec.vega(
            strike, atm_forward, std_dev, exercise_time, annuity, displacement
        )
        results.additional_results["delta"] = self._spec.delta(
            w, strike, atm_forward, std_dev, annuity, displacement
        )
        results.additional_results["timeToExpiry"] = exercise_time
        # ALIGN (v1.43 bondswap wave): C++ blackswaptionengine.hpp:325 is a bare
        # ``Real(stdDev / std::sqrt(exerciseTime))`` with no guard, and IEEE-754
        # double division makes that NaN (0/0) or +/-inf rather than an error.
        # Python raises ZeroDivisionError instead, which made a swaption
        # expiring TODAY unpriceable — the first swaptionlet of every
        # ``CounterpartyAdjSwapEngine`` strip is exactly that.
        sqrt_exercise_time = math.sqrt(exercise_time)
        if sqrt_exercise_time == 0.0:
            implied_volatility = (
                math.nan if std_dev == 0.0 else math.copysign(math.inf, std_dev)
            )
        else:
            implied_volatility = std_dev / sqrt_exercise_time
        results.additional_results["impliedVolatility"] = implied_volatility
        results.additional_results["forwardPrice"] = results.value / self._discount_curve.discount(
            exercise_date
        )


class BlackSwaptionEngine(BlackStyleSwaptionEngine):
    """Shifted-lognormal (Black 76) swaption engine.

    # C++ parity: ``class BlackSwaptionEngine`` in
    # blackswaptionengine.hpp:136-152 + blackswaptionengine.cpp:30-55 (v1.43).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote | SwaptionVolatilityStructure,
        day_counter: DayCounter | None = None,
        displacement: float = 0.0,
        model: CashAnnuityModel = CashAnnuityModel.DiscountCurve,
    ) -> None:
        super().__init__(
            Black76Spec(),
            discount_curve,
            vol,
            day_counter=day_counter,
            displacement=displacement,
            model=model,
        )
        # # C++ parity: blackswaptionengine.cpp:52-54 — only the
        # # SwaptionVolatilityStructure ctor validates the vol type.
        if isinstance(vol, SwaptionVolatilityStructure):
            qassert.require(
                vol.volatility_type() == VolatilityType.ShiftedLognormal,
                "BlackSwaptionEngine requires (shifted) lognormal input volatility",
            )


class BachelierSwaptionEngine(BlackStyleSwaptionEngine):
    """Normal (Bachelier) swaption engine.

    # C++ parity: ``class BachelierSwaptionEngine`` in
    # blackswaptionengine.hpp:161-175 + blackswaptionengine.cpp:58-77 (v1.43).
    """

    def __init__(
        self,
        discount_curve: YieldTermStructureProtocol,
        vol: float | Quote | SwaptionVolatilityStructure,
        day_counter: DayCounter | None = None,
        model: CashAnnuityModel = CashAnnuityModel.DiscountCurve,
    ) -> None:
        super().__init__(
            BachelierSpec(),
            discount_curve,
            vol,
            day_counter=day_counter,
            displacement=0.0,
            model=model,
        )
        # # C++ parity: blackswaptionengine.cpp:75-76.
        if isinstance(vol, SwaptionVolatilityStructure):
            qassert.require(
                vol.volatility_type() == VolatilityType.Normal,
                "BachelierSwaptionEngine requires normal input volatility",
            )


__all__ = [
    "BachelierSpec",
    "BachelierSwaptionEngine",
    "Black76Spec",
    "BlackStyleSwaptionEngine",
    "BlackSwaptionEngine",
    "CashAnnuityModel",
]
