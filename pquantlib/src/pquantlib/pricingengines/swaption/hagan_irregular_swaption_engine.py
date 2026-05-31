"""HaganIrregularSwaptionEngine — Hagan linear-TSR engine for irregular swaptions.

# C++ parity: ql/experimental/swaptions/haganirregularswaptionengine.hpp + .cpp
# (v1.42.1, 099987f0).

Prices an irregular (e.g. step-down notional) European swaption by
super-replication: the irregular swap is decomposed into a basket of standard
vanilla swaps (Hagan's methodology), and the irregular swaption price is the
weighted sum of the corresponding vanilla-swaption prices.

References:
  1. P.S. Hagan, "Methodology for Callable Swaps and Bermudan 'Exercise into
     Swaptions'".
  2. P.J. Hunt, J.E. Kennedy, "Implied interest rate pricing models",
     Finance Stochast. 2, 275-293 (1998).

# C++ parity divergence (SVD): the C++ Basket solves the replication linear
# system with ``SVD::solveFor`` (a least-squares solve via singular-value
# decomposition). PQuantLib delegates to ``numpy.linalg.lstsq`` (also an
# SVD-based least-squares solver), per the project delegation philosophy. See
# docs/carve-outs.md (Category 3 — ecosystem tooling).

# C++ parity divergence (vol structure): the C++ HKPrice passes the whole
# SwaptionVolatilityStructure to BachelierSwaptionEngine; PQuantLib's
# BachelierSwaptionEngine takes a flat normal vol, so each member swaption's
# vol is looked up from the structure at its (exercise, tenor, ATM-strike)
# point and passed as a flat Quote.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.coupon_pricer import BlackIborCouponPricer
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.ibor_coupon import IborCoupon
from pquantlib.exercise import Exercise
from pquantlib.instruments.irregular_swap import IrregularSwap
from pquantlib.instruments.irregular_swaption import (
    IrregularSwaptionArguments,
)
from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap
from pquantlib.instruments.swaption import Swaption, SwaptionResults
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.pricingengines.swaption.black_swaption_engine import (
    BachelierSwaptionEngine,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class _Basket:
    """Replication of an irregular swap as a basket of standard vanilla swaps.

    # C++ parity: ``HaganIrregularSwaptionEngine::Basket`` (haganirregular...cpp).
    """

    def __init__(
        self,
        swap: IrregularSwap,
        term_structure: YieldTermStructureProtocol,
        volatility_structure: SwaptionVolatilityStructure,
    ) -> None:
        self._swap: IrregularSwap = swap
        self._term_structure: YieldTermStructureProtocol = term_structure
        self._volatility_structure: SwaptionVolatilityStructure = volatility_structure
        self._lambda: float = 0.0

        engine = DiscountingSwapEngine(term_structure)
        swap.set_pricing_engine(engine)
        self._target_npv: float = swap.npv()

        fixed_leg = swap.fixed_leg()
        float_leg = swap.floating_leg()

        self._fair_rates: list[float] = []
        self._annuities: list[float] = []
        self._expiries: list[Date] = []

        fixed_cfs: list[FixedRateCoupon] = []

        for i in range(len(fixed_leg)):
            coupon = fixed_leg[i]
            qassert.require(
                isinstance(coupon, FixedRateCoupon),
                "dynamic cast of fixed leg coupon failed.",
            )
            assert isinstance(coupon, FixedRateCoupon)

            self._expiries.append(coupon.date())

            new_cpn = FixedRateCoupon.from_rate(
                coupon.date(),
                1.0,
                coupon.rate(),
                coupon.day_counter(),
                coupon.accrual_start_date(),
                coupon.accrual_end_date(),
                coupon.reference_period_start(),
                coupon.reference_period_end(),
            )
            fixed_cfs.append(new_cpn)

            self._annuities.append(
                10000.0
                * CashFlows.bps(fixed_cfs, term_structure, include_settlement_date_flows=True)
            )

            float_cfs: list[IborCoupon] = []
            for cf in float_leg:
                qassert.require(
                    isinstance(cf, IborCoupon),
                    "dynamic cast of float leg coupon failed.",
                )
                assert isinstance(cf, IborCoupon)
                if cf.date() <= self._expiries[i]:
                    new_float = IborCoupon(
                        cf.date(),
                        1.0,
                        cf.accrual_start_date(),
                        cf.accrual_end_date(),
                        cf.fixing_days(),
                        cf.ibor_index(),
                        1.0,
                        cf.spread(),
                        cf.reference_period_start(),
                        cf.reference_period_end(),
                        cf.day_counter(),
                        cf.is_in_arrears(),
                    )
                    if not new_float.is_in_arrears():
                        new_float.set_pricer(BlackIborCouponPricer())
                    float_cfs.append(new_float)

            float_leg_npv = CashFlows.npv_curve(
                float_cfs, term_structure, include_settlement_date_flows=True
            )
            self._fair_rates.append(float_leg_npv / self._annuities[i])

    def compute(self, lambda_: float = 0.0) -> np.ndarray:
        """Replication weights — solve the Hagan linear system.

        # C++ parity: ``Basket::compute`` (haganirregular...cpp:107-164).
        """
        self._lambda = lambda_
        n = len(self._swap.fixed_leg())

        arr = np.zeros((n, n))
        rhs = np.zeros(n)

        fixed_leg = self._swap.fixed_leg()

        for r in range(n):
            cpn_r = fixed_leg[r]
            assert isinstance(cpn_r, FixedRateCoupon)
            for c in range(r, n):
                arr[r][c] = (self._fair_rates[c] + lambda_) * cpn_r.accrual_period()
            arr[r][r] += 1.0

        for r in range(n):
            cpn_r = fixed_leg[r]
            assert isinstance(cpn_r, FixedRateCoupon)
            n_r = cpn_r.nominal()
            if r < n - 1:
                cpn_rp1 = fixed_leg[r + 1]
                assert isinstance(cpn_rp1, FixedRateCoupon)
                n_rp1 = cpn_rp1.nominal()
                rhs[r] = n_r * cpn_r.rate() * cpn_r.accrual_period() + (n_r - n_rp1)
            else:
                rhs[r] = n_r * cpn_r.rate() * cpn_r.accrual_period() + n_r

        # # C++ uses SVD::solveFor — least-squares solve via SVD.
        solution, *_ = np.linalg.lstsq(arr, rhs, rcond=None)
        return solution

    def __call__(self, lambda_: float) -> float:
        """Defect of the replication at a given lambda (root => calibrated).

        # C++ parity: ``Basket::operator()`` (haganirregular...cpp:167-177).
        """
        weights = self.compute(lambda_)
        defect = -self._target_npv
        type_sign = int(self._swap.type())
        for i in range(len(weights)):
            defect -= type_sign * lambda_ * weights[i] * self._annuities[i]
        return defect

    def weights(self) -> np.ndarray:
        return self.compute(self._lambda)

    def lambda_value(self) -> float:
        return self._lambda

    def swap(self) -> IrregularSwap:
        return self._swap

    def component(self, i: int) -> VanillaSwap:
        """Build the i-th replicating standard vanilla swap.

        # C++ parity: ``Basket::component`` (haganirregular...cpp:181-213).
        """
        ibor_cpn = self._swap.floating_leg()[0]
        qassert.require(
            isinstance(ibor_cpn, IborCoupon),
            "dynamic cast of float leg coupon failed. Can't find index.",
        )
        assert isinstance(ibor_cpn, IborCoupon)
        ibor_index = ibor_cpn.ibor_index()

        dummy_swap_length = Period(1, TimeUnit.Years)

        member_swap = make_vanilla_swap(
            swap_tenor=dummy_swap_length,
            ibor_index=ibor_index,  # type: ignore[arg-type]
            swap_type=self._swap.type(),
            effective_date=self._swap.start_date(),
            termination_date=self._expiries[i],
            fixed_leg_rule=DateGeneration.Backward,
            discount_curve=self._term_structure,
        )

        std_annuity = 10000.0 * CashFlows.bps(
            member_swap.fixed_leg(), self._term_structure, include_settlement_date_flows=True
        )

        transformed_rate = (self._fair_rates[i] + self._lambda) * self._annuities[i] / std_annuity

        return make_vanilla_swap(
            swap_tenor=dummy_swap_length,
            ibor_index=ibor_index,  # type: ignore[arg-type]
            fixed_rate=transformed_rate,
            swap_type=self._swap.type(),
            effective_date=self._swap.start_date(),
            termination_date=self._expiries[i],
            fixed_leg_rule=DateGeneration.Backward,
            discount_curve=self._term_structure,
        )


class HaganIrregularSwaptionEngine(
    GenericEngine[IrregularSwaptionArguments, SwaptionResults]
):
    """Hagan super-replication engine for European irregular swaptions."""

    def __init__(
        self,
        volatility_structure: SwaptionVolatilityStructure,
        term_structure: YieldTermStructureProtocol,
    ) -> None:
        super().__init__(IrregularSwaptionArguments(), SwaptionResults())
        self._term_structure: YieldTermStructureProtocol = term_structure
        self._volatility_structure: SwaptionVolatilityStructure = volatility_structure

    def calculate(self) -> None:
        """# C++ parity: haganirregular...cpp:231-326."""
        exercise = self._arguments.exercise
        assert exercise is not None
        qassert.require(
            exercise.type() == Exercise.Type.European, "swaption must be european"
        )

        swap = self._arguments.swap
        assert swap is not None

        # Reshuffle spread from float to fixed (remove spread from float side by
        # adjusting the fixed coupon so the swap NPV is unchanged).
        fixed_leg = swap.fixed_leg()
        fxd_lg_bps = CashFlows.bps(
            fixed_leg, self._term_structure, include_settlement_date_flows=True
        )

        float_leg = swap.floating_leg()
        flt_lg_npv = CashFlows.npv_curve(
            float_leg, self._term_structure, include_settlement_date_flows=True
        )
        flt_lg_bps = CashFlows.bps(
            float_leg, self._term_structure, include_settlement_date_flows=True
        )

        float_cfs: list[IborCoupon] = []
        for cf in float_leg:
            qassert.require(
                isinstance(cf, IborCoupon), "dynamic cast of float leg coupon failed."
            )
            assert isinstance(cf, IborCoupon)
            new_cpn = IborCoupon(
                cf.date(),
                cf.nominal(),
                cf.accrual_start_date(),
                cf.accrual_end_date(),
                cf.fixing_days(),
                cf.ibor_index(),
                cf.gearing(),
                0.0,
                cf.reference_period_start(),
                cf.reference_period_end(),
                cf.day_counter(),
                cf.is_in_arrears(),
            )
            if not new_cpn.is_in_arrears():
                new_cpn.set_pricer(BlackIborCouponPricer())
            float_cfs.append(new_cpn)

        sprd_lg_npv = flt_lg_npv - CashFlows.npv_curve(
            float_cfs, self._term_structure, include_settlement_date_flows=True
        )
        avg_spread = sprd_lg_npv / flt_lg_bps / 10000.0
        cpn_adjustment = avg_spread * flt_lg_bps / fxd_lg_bps

        fixed_cfs: list[FixedRateCoupon] = []
        for cf in fixed_leg:
            qassert.require(
                isinstance(cf, FixedRateCoupon), "dynamic cast of fixed leg coupon failed."
            )
            assert isinstance(cf, FixedRateCoupon)
            new_fixed = FixedRateCoupon.from_rate(
                cf.date(),
                cf.nominal(),
                cf.rate() - cpn_adjustment,
                cf.day_counter(),
                cf.accrual_start_date(),
                cf.accrual_end_date(),
                cf.reference_period_start(),
                cf.reference_period_end(),
            )
            fixed_cfs.append(new_fixed)

        # the irregular swap with spread removed
        spread_free_swap = IrregularSwap(swap.type(), fixed_cfs, float_cfs)

        basket = _Basket(spread_free_swap, self._term_structure, self._volatility_structure)

        # find lambda by root-finding (C++ uses Bisection over [-0.5, 0.5]).
        from pquantlib.math.solvers1d.bisection import Bisection  # noqa: PLC0415

        s1d = Bisection()
        min_lambda = -0.5
        max_lambda = 0.5
        s1d.set_max_evaluations(10000)
        s1d.set_lower_bound(min_lambda)
        s1d.set_upper_bound(max_lambda)
        s1d.solve(basket, 1.0e-8, 0.01, min_lambda, max_lambda)

        self._results.value = self._hk_price(basket, exercise)

    def _hk_price(self, basket: _Basket, exercise: Exercise) -> float:
        """Hunt-Kennedy price = sum of weighted vanilla-swaption prices.

        # C++ parity: ``HKPrice`` (haganirregular...cpp:334-359).
        """
        qassert.require(
            self._volatility_structure.volatility_type() == VolatilityType.Normal,
            "swaptionEngine: only normal volatility implemented.",
        )

        weights = basket.weights()
        npv = 0.0
        for i in range(len(weights)):
            pv_swap = basket.component(i)
            swaption = Swaption(pv_swap, exercise)

            # Look up the flat normal vol for this member swaption from the
            # structure at (exercise, swap tenor, ATM strike).
            exercise_date = exercise.dates()[-1]
            swap_tenor = self._member_tenor(pv_swap)
            atm_strike = pv_swap.fair_rate()
            flat_vol = self._volatility_structure.volatility(
                exercise_date, swap_tenor, atm_strike, True
            )

            swaption.set_pricing_engine(
                BachelierSwaptionEngine(self._term_structure, flat_vol)
            )
            npv += weights[i] * swaption.npv()
        return npv

    @staticmethod
    def _member_tenor(swap: VanillaSwap) -> Period:
        """Approximate swap tenor (years) from the fixed leg span."""
        fixed = swap.fixed_leg()
        first = fixed[0]
        last = fixed[-1]
        assert isinstance(first, FixedRateCoupon)
        assert isinstance(last, FixedRateCoupon)
        start = first.accrual_start_date()
        end = last.accrual_end_date()
        years = max(1, round((end.serial_number() - start.serial_number()) / 365.25))
        return Period(years, TimeUnit.Years)


__all__ = [
    "HaganIrregularSwaptionEngine",
]
