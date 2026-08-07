"""JumpDiffusionEngine — Merton (1976) jump-diffusion European option engine.

# C++ parity: ql/pricingengines/vanilla/jumpdiffusionengine.{hpp,cpp} (v1.43)
# — ``class JumpDiffusionEngine : public VanillaOption::engine``.

Merton's series: a jump-diffusion price is a Poisson-weighted sum of
Black-Scholes prices, term ``i`` using the variance and rate the option
would have *conditional on exactly i jumps*::

    v_i = sqrt((variance + i sigma_J^2) / t)
    r_i = r - lambda_0 k + i (mu_J + sigma_J^2/2) / t

with ``k = exp(mu_J + sigma_J^2/2) - 1`` the mean jump size and
``lambda = (k + 1) lambda_0`` the risk-adjusted intensity that drives the
Poisson weights ``p(i; lambda t)``.

Two things a port gets wrong if it simplifies
---------------------------------------------
1. The loop condition has an **or** in it::

       for (i = 0; (lastContribution > relativeAccuracy_ && i < maxIterations_)
                   || i < Size(lambda*t); i++)

   so the series always runs at least ``floor(lambda t)`` terms no matter how
   fast it converges.  A plain while-not-converged loop diverges from C++ as
   soon as ``lambda t >= 1``.

2. The per-term curves are rebuilt with the **volatility** term structure's
   day counter and the **risk-free** curve's reference date::

       new FlatForward(rateRefDate, r, voldc)
       new BlackConstantVol(rateRefDate, volcal, v, voldc)

   — a deliberate-looking mismatch in the C++ source that changes the answer
   whenever the two curves use different day counters or reference dates.
   Reproduced verbatim.

The theta accumulation is likewise idiosyncratic: each term adds
``theta_i + correction_i + lambda * value_i`` and then subtracts
``p(i-1) lambda value_i`` for ``i > 0`` (a telescoping of the intensity
term), where ``correction_i`` mixes the term's vega and rho.

Results filled: ``value``, ``delta``, ``gamma``, ``theta``, ``vega``,
``rho``, ``dividend_rho``.  Nothing else — no ``theta_per_day``, no
``strike_sensitivity``, no ``itm_cash_probability``.

C++ relinks a ``RelinkableHandle`` per term; Python has no handle
indirection, so a fresh :class:`GeneralizedBlackScholesProcess` and
:class:`AnalyticEuropeanEngine` are built per term instead.  Numerically
identical — the C++ engine re-runs ``baseEngine.calculate()`` from scratch
each iteration anyway.
"""

from __future__ import annotations

import math
import sys
from typing import Final

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.poisson_distribution import PoissonDistribution
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import AnalyticEuropeanEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.merton76_process import Merton76Process
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward

# C++ ``QL_EPSILON``.
_QL_EPSILON: Final[float] = sys.float_info.epsilon


def _required(value: float | None, name: str) -> float:
    """Unwrap a base-engine result the C++ code reads unconditionally."""
    if value is None:
        raise LibraryException(f"{name} not provided by the base European engine")
    return value


class JumpDiffusionEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Jump-diffusion engine for vanilla options.

    # C++ parity: ``class JumpDiffusionEngine``.
    """

    def __init__(
        self,
        process: Merton76Process,
        relative_accuracy: float = 1e-4,
        max_iterations: int = 100,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: Merton76Process = process
        self._relative_accuracy: float = relative_accuracy
        self._max_iterations: int = max_iterations
        process.register_with(self)

    def calculate(self) -> None:  # noqa: PLR0915 - one-to-one with the C++ body
        """Sum Merton's Poisson series of Black-Scholes prices.

        # C++ parity: ``JumpDiffusionEngine::calculate``.
        """
        args = self._arguments
        results = self._results
        process = self._process

        jump_square_vol = (
            process.log_jump_volatility().value() * process.log_jump_volatility().value()
        )
        mu_plus_half_square_vol = process.log_mean_jump().value() + 0.5 * jump_square_vol
        # mean jump size
        k = math.exp(mu_plus_half_square_vol) - 1.0
        lambda_ = (k + 1.0) * process.jump_intensity().value()

        qassert.require(args.payoff is not None, "no payoff given")
        assert args.payoff is not None
        qassert.require(isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        last_date = args.exercise.last_date()

        variance = process.black_volatility().black_variance(
            last_date, payoff.strike(), extrapolate=True
        )

        voldc = process.black_volatility().day_counter()
        volcal = process.black_volatility().calendar()
        vol_ref_date = process.black_volatility().reference_date()
        t = voldc.year_fraction(vol_ref_date, last_date)
        risk_free_rate = -math.log(process.risk_free_rate().discount(last_date)) / t
        rate_ref_date = process.risk_free_rate().reference_date()

        p = PoissonDistribution(lambda_ * t)

        state_variable = process.state_variable()
        dividend_ts = process.dividend_yield()

        results.value = 0.0
        results.delta = 0.0
        results.gamma = 0.0
        results.theta = 0.0
        results.vega = 0.0
        results.rho = 0.0
        results.dividend_rho = 0.0

        last_contribution = 1.0
        i = 0
        # C++: for (i=0; (lastContribution>relativeAccuracy_ && i<maxIterations_)
        #                || i < Size(lambda*t); i++)
        min_terms = int(lambda_ * t)
        while (
            last_contribution > self._relative_accuracy and i < self._max_iterations
        ) or i < min_terms:
            # constant vol/rate assumption. It should be relaxed
            v = math.sqrt((variance + i * jump_square_vol) / t)
            r = (
                risk_free_rate
                - process.jump_intensity().value() * k
                + i * mu_plus_half_square_vol / t
            )
            # NOTE: the rate curve is built with the VOLATILITY day counter and
            # the RISK-FREE reference date, exactly as in C++.
            risk_free_ts = FlatForward.from_rate(
                reference_date=rate_ref_date, forward_rate=r, day_counter=voldc
            )
            vol_ts = BlackConstantVol(
                reference_date=rate_ref_date,
                calendar=volcal,
                day_counter=voldc,
                volatility=v,
            )
            bs_process = GeneralizedBlackScholesProcess(
                x0=state_variable,
                dividend_ts=dividend_ts,
                risk_free_ts=risk_free_ts,
                black_vol_ts=vol_ts,
            )
            base_engine = AnalyticEuropeanEngine(bs_process)
            base_arguments = base_engine.get_arguments()
            base_arguments.payoff = args.payoff
            base_arguments.exercise = args.exercise
            base_arguments.validate()
            base_engine.calculate()
            base_results = base_engine.get_results()

            base_value = _required(base_results.value, "value")
            base_delta = _required(base_results.delta, "delta")
            base_gamma = _required(base_results.gamma, "gamma")
            base_theta = _required(base_results.theta, "theta")
            base_vega = _required(base_results.vega, "vega")
            base_rho = _required(base_results.rho, "rho")
            base_dividend_rho = _required(base_results.dividend_rho, "dividendRho")

            weight = p(i)
            results.value += weight * base_value
            results.delta += weight * base_delta
            results.gamma += weight * base_gamma
            results.vega += weight * (math.sqrt(variance / t) / v) * base_vega
            # theta modified
            theta_correction = (
                base_vega * ((i * jump_square_vol) / (2.0 * v * t * t))
                + base_rho * i * mu_plus_half_square_vol / (t * t)
            )
            results.theta += weight * (base_theta + theta_correction + lambda_ * base_value)
            if i != 0:
                results.theta -= p(i - 1) * lambda_ * base_value
            # end theta calculation
            results.rho += weight * base_rho
            results.dividend_rho += weight * base_dividend_rho

            last_contribution = abs(
                base_value / (results.value if abs(results.value) > _QL_EPSILON else 1.0)
            )
            last_contribution = max(
                last_contribution,
                abs(base_delta / (results.delta if abs(results.delta) > _QL_EPSILON else 1.0)),
            )
            last_contribution = max(
                last_contribution,
                abs(base_gamma / (results.gamma if abs(results.gamma) > _QL_EPSILON else 1.0)),
            )
            last_contribution = max(
                last_contribution,
                abs(base_theta / (results.theta if abs(results.theta) > _QL_EPSILON else 1.0)),
            )
            last_contribution = max(
                last_contribution,
                abs(base_vega / (results.vega if abs(results.vega) > _QL_EPSILON else 1.0)),
            )
            last_contribution = max(
                last_contribution,
                abs(base_rho / (results.rho if abs(results.rho) > _QL_EPSILON else 1.0)),
            )
            last_contribution = max(
                last_contribution,
                abs(
                    base_dividend_rho
                    / (
                        results.dividend_rho
                        if abs(results.dividend_rho) > _QL_EPSILON
                        else 1.0
                    )
                ),
            )
            last_contribution *= weight

            i += 1

        qassert.require(
            i < self._max_iterations,
            f"{i} iterations have been not enough to reach the required "
            f"{self._relative_accuracy} accuracy. The {i} addendum was "
            f"{last_contribution} while the running sum was {results.value}",
        )


__all__ = ["JumpDiffusionEngine"]
