"""AnalyticCompoundOptionEngine — closed-form compound option pricing.

# C++ parity:
# ql/pricingengines/exotic/analyticcompoundoptionengine.{hpp,cpp}
# (v1.42.1).

Wystup 2002 closed-form for compound options under Black-Scholes
dynamics. The algorithm:

1. Find the implicit "trigger" spot ``S*`` at the mother expiry such
   that the daughter Black-Scholes value at ``(S*, T_M)`` equals the
   mother strike. Brent solver inverts the BSM call value.
2. Compute the compound NPV via a bivariate-normal CDF formula
   involving ``transformX(S*)`` and the daughter d_+ / d_-.
3. Greeks (delta, gamma, vega, theta) follow analogous bivariate-normal
   expressions.

Only ``PlainVanillaPayoff`` is supported on both legs (mirrors C++
``dynamic_pointer_cast<PlainVanillaPayoff>``).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.instruments.compound_option import CompoundOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.bivariate_normal_distribution import (
    BivariateCumulativeNormalDistributionDr78,
)
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class _ImpliedSpotHelper:
    """Closure: ``BSM(spot) - mother_strike`` — root is the trigger spot.

    # C++ parity: anonymous namespace ``ImpliedSpotHelper`` in
    # analyticcompoundoptionengine.cpp:31-52.
    """

    def __init__(
        self,
        *,
        dividend_discount: float,
        risk_free_discount: float,
        std_dev: float,
        payoff: PlainVanillaPayoff,
        mother_strike: float,
    ) -> None:
        self._dividend_discount = dividend_discount
        self._risk_free_discount = risk_free_discount
        self._std_dev = std_dev
        self._payoff = payoff
        self._mother_strike = mother_strike

    def __call__(self, spot: float) -> float:
        forward = spot * self._dividend_discount / self._risk_free_discount
        value = black_formula(
            self._payoff.option_type(),
            self._payoff.strike(),
            forward,
            self._std_dev,
            self._risk_free_discount,
        )
        return value - self._mother_strike


class AnalyticCompoundOptionEngine(
    GenericEngine[CompoundOptionArguments, OneAssetOptionResults]
):
    """Wystup 2002 closed-form for compound options.

    # C++ parity: ``AnalyticCompoundOptionEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(CompoundOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._n: CumulativeNormalDistribution = CumulativeNormalDistribution()
        self._n_pdf: NormalDistribution = NormalDistribution()
        process.register_with(self)

    # --- helpers --------------------------------------------------------

    def _payoff_mother(self) -> PlainVanillaPayoff:
        payoff = self._arguments.payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff

    def _payoff_daughter(self) -> PlainVanillaPayoff:
        payoff = self._arguments.daughter_payoff
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        return payoff

    def _strike_mother(self) -> float:
        return self._payoff_mother().strike()

    def _strike_daughter(self) -> float:
        return self._payoff_daughter().strike()

    def _spot(self) -> float:
        return self._process.state_variable().value()

    def _type_mother(self) -> float:
        return float(int(self._payoff_mother().option_type()))

    def _type_daughter(self) -> float:
        return float(int(self._payoff_daughter().option_type()))

    def _maturity_mother(self) -> Date:
        ex = self._arguments.exercise
        assert ex is not None
        return ex.last_date()

    def _maturity_daughter(self) -> Date:
        ex = self._arguments.daughter_exercise
        assert ex is not None
        return ex.last_date()

    def _residual_mother(self) -> float:
        return self._process.time(self._maturity_mother())

    def _residual_daughter(self) -> float:
        return self._process.time(self._maturity_daughter())

    def _residual_mother_daughter(self) -> float:
        return self._residual_daughter() - self._residual_mother()

    def _vol_daughter(self) -> float:
        return self._process.black_volatility().black_vol(
            self._maturity_daughter(), self._strike_daughter(), extrapolate=True
        )

    def _vol_mother(self) -> float:
        return self._process.black_volatility().black_vol(
            self._maturity_mother(), self._strike_mother(), extrapolate=True
        )

    def _std_dev_daughter(self) -> float:
        return self._vol_daughter() * math.sqrt(self._residual_daughter())

    def _std_dev_mother(self) -> float:
        return self._vol_mother() * math.sqrt(self._residual_mother())

    def _risk_free_discount_daughter(self) -> float:
        return self._process.risk_free_rate().discount(self._residual_daughter())

    def _risk_free_discount_mother(self) -> float:
        return self._process.risk_free_rate().discount(self._residual_mother())

    def _risk_free_discount_md(self) -> float:
        return self._process.risk_free_rate().discount(
            self._residual_mother_daughter()
        )

    def _dividend_discount_daughter(self) -> float:
        return self._process.dividend_yield().discount(self._residual_daughter())

    def _dividend_discount_mother(self) -> float:
        return self._process.dividend_yield().discount(self._residual_mother())

    def _dividend_discount_md(self) -> float:
        return self._process.dividend_yield().discount(
            self._residual_mother_daughter()
        )

    def _d_plus(self) -> float:
        forward = (
            self._spot()
            * self._dividend_discount_daughter()
            / self._risk_free_discount_daughter()
        )
        sd = self._std_dev_daughter()
        return math.log(forward / self._strike_daughter()) / sd + 0.5 * sd

    def _d_minus(self) -> float:
        return self._d_plus() - self._std_dev_daughter()

    def _d_plus_tau12(self, s: float) -> float:
        forward = s * self._dividend_discount_md() / self._risk_free_discount_md()
        sd = self._vol_daughter() * math.sqrt(self._residual_mother_daughter())
        return math.log(forward / self._strike_daughter()) / sd + 0.5 * sd

    def _transform_x(self, x: float) -> float:
        sd = self._std_dev_mother()
        res_x = self._risk_free_discount_mother() * x / (
            self._spot() * self._dividend_discount_mother()
        )
        res_x = res_x * math.exp(0.5 * sd * sd)
        res_x = math.log(res_x)
        return res_x / sd

    def _e(self, x: float) -> float:
        rtm = self._residual_mother()
        rtd = self._residual_daughter()
        return (x * math.sqrt(rtd) + math.sqrt(rtm) * self._d_minus()) / math.sqrt(
            rtd - rtm
        )

    # --- main ----------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915  (faithful C++ port — Wystup formula is intrinsically one long routine)
        """Wystup 2002 closed-form NPV + Greeks.

        # C++ parity: ``AnalyticCompoundOptionEngine::calculate``.
        """
        qassert.require(
            self._strike_daughter() > 0.0, "Daughter strike must be positive"
        )
        qassert.require(
            self._strike_mother() > 0.0, "Mother strike must be positive"
        )
        qassert.require(self._spot() > 0.0, "negative or null underlying given")

        # Mother-to-daughter residual: use process.time on a helper date
        # to mirror C++ exactly (different from passing time directly
        # because day-counter rounding can drift). We use a residual
        # time-based proxy: vol at K_d times sqrt(T_D - T_M).
        rt_md = self._residual_mother_daughter()
        vol_helper = self._vol_daughter()
        help_std = vol_helper * math.sqrt(rt_md)

        dividend_discount = self._dividend_discount_md()
        risk_free_discount = self._risk_free_discount_md()

        helper = _ImpliedSpotHelper(
            dividend_discount=dividend_discount,
            risk_free_discount=risk_free_discount,
            std_dev=help_std,
            payoff=self._payoff_daughter(),
            mother_strike=self._strike_mother(),
        )

        solver = Brent()
        solver.set_max_evaluations(1000)
        accuracy = 1.0e-6

        s_solved = solver.solve(
            helper,
            accuracy,
            self._strike_daughter(),
            1.0e-6,
            self._strike_daughter() * 1000.0,
        )
        x = self._transform_x(s_solved)

        phi = self._type_daughter()  # -1 or +1
        w = self._type_mother()  # -1 or +1

        rho = math.sqrt(self._residual_mother() / self._residual_daughter())
        n2 = BivariateCumulativeNormalDistributionDr78(w * rho)

        dd_d = self._dividend_discount_daughter()
        rd_d = self._risk_free_discount_daughter()
        rd_m = self._risk_free_discount_mother()

        x_msm = x - self._std_dev_mother()
        spot = self._spot()
        d_plus = self._d_plus()
        d_plus_t12 = self._d_plus_tau12(s_solved)
        v_d = self._vol_daughter()

        d_minus = self._d_minus()
        str_d = self._strike_daughter()
        str_m = self._strike_mother()
        rtm = self._residual_mother()
        rtd = self._residual_daughter()

        rd = self._zero_rate_at(rtd, self._process.risk_free_rate())
        dd = self._zero_rate_at(rtd, self._process.dividend_yield())

        n2_xmsm = n2(-phi * w * x_msm, phi * d_plus)
        n2_x = n2(-phi * w * x, phi * d_minus)
        ne_x = self._n(-phi * w * self._e(x))
        n_x = self._n(-phi * w * x)
        n_t12 = self._n(phi * d_plus_t12)
        n_dp = self._n_pdf(d_plus)
        n_xm = self._n_pdf(x_msm)
        inv_m_time = 1.0 / math.sqrt(rtm)
        inv_d_time = 1.0 / math.sqrt(rtd)

        temp_res = (
            phi * w * spot * dd_d * n2_xmsm
            - phi * w * str_d * rd_d * n2_x
            - w * str_m * rd_m * n_x
        )
        temp_delta = phi * w * dd_d * n2_xmsm
        temp_gamma = (dd_d / (v_d * spot)) * (
            inv_m_time * n_xm * n_t12 + w * inv_d_time * n_dp * ne_x
        )
        temp_vega = dd_d * spot * (
            (1.0 / inv_m_time) * n_xm * n_t12
            + w * (1.0 / inv_d_time) * n_dp * ne_x
        )
        temp_theta = (
            phi * w * dd * spot * dd_d * n2_xmsm
            - phi * w * rd * str_d * rd_d * n2_x
            - w * rd * str_m * rd_m * n_x
        )
        temp_theta -= 0.5 * v_d * spot * dd_d * (
            inv_m_time * n_xm * n_t12 + w * inv_d_time * n_dp * ne_x
        )

        results = self._results
        results.reset()
        results.value = temp_res
        results.delta = temp_delta
        results.gamma = temp_gamma
        results.vega = temp_vega
        results.theta = temp_theta

    @staticmethod
    def _zero_rate_at(t: float, ts: YieldTermStructure) -> float:
        """Continuous zero rate at time ``t``.

        # C++ parity: ``ts->zeroRate(t, Continuous, NoFrequency)``.
        """
        return ts.zero_rate(
            t, Compounding.Continuous, Frequency.NoFrequency
        ).rate()

    def update(self) -> None:
        self.notify_observers()


__all__ = ["AnalyticCompoundOptionEngine"]
