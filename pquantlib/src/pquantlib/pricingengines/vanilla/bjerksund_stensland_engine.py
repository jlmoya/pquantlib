"""BjerksundStenslandApproximationEngine — Bjerksund-Stensland (1993/2002).

# C++ parity: ql/pricingengines/vanilla/bjerksundstenslandengine.{hpp,cpp}
# (v1.43) — ``class BjerksundStenslandApproximationEngine :
# public VanillaOption::engine``.

American option approximation with a flat exercise boundary ``I``.  Only the
*call* case is implemented; puts go through the put-call symmetry transform

    swap(spot, strike);  swap(risk_free_discount, dividend_discount)

and the results are un-transformed afterwards by swapping
``delta <-> strike_sensitivity``, ``gamma <-> additional_results["strikeGamma"]``
and ``rho <-> dividend_rho``, then rescaling ``rho *= t_r / t_q`` and
``dividend_rho *= t_q / t_r``.  A port that skips any part of that dance gets
the right NPV and the wrong greeks.

Three result shapes, tagged by ``additional_results["exerciseType"]``:

``"European"``
    ``dividend_discount >= 1.0 and dividend_discount >= risk_free_discount``
    — early exercise never optimal; also the fallback when the exercise
    boundary runs away (``q = ln(I/fwd)/sqrt(variance) > 12.5``, where the
    European greeks are numerically better) and when the approximation
    undercuts the European value.

``"Immediate"``
    ``spot >= I`` at the outset, or the final ``value < (S - X)(1 + 10 eps)``
    check.  Value ``max(0, S - X)``, delta 0 or 1, every other greek zero.

``"American"``
    The approximation proper.  Value, delta, gamma, rho, dividend_rho and
    vega are all analytic, built from the ``_phi*`` family below — ``_phi``
    itself plus its partial derivatives in ``S`` (twice), ``gamma``, ``H``,
    ``I``, ``rT``, ``bT`` and the variance.  ``theta_per_day`` is *derived*
    from vega/rho/dividend_rho by the BSM relation rather than computed
    directly, and ``theta = 365 * theta_per_day``.

Guard: ``dividend_discount > 1.0 and risk_free_discount > dividend_discount``
raises — the double-boundary case ``r < q < 0``, which this approximation
cannot represent.

Documented divergence (C++ defect)
----------------------------------
C++ builds a *local* ``OneAssetOption::results`` in each of the three result
builders and assigns the whole struct into ``results_``.  ``Greeks`` and
``MoreGreeks`` have no default member initialisers, so
``itmCashProbability``, ``deltaForward`` and ``elasticity`` are never
written and the assignment copies indeterminate values — the C++ accessors
then return garbage (observed: ``deltaForward == -1.03e+80``) instead of
signalling "not computed".  Python leaves those three unset, so the
accessors raise.  Reading uninitialised memory has no faithful Python
analogue and reproducing garbage would be worse than useless; the probe
marks those slots ``"indeterminate"`` and the cross-validation skips them.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.exercise import AmericanExercise, Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.date import Date

# The C99 <math.h> constants the C++ source uses, written as the same decimal
# literals so both languages round to the identical double.
_M_SQRT2: Final[float] = 1.41421356237309504880168872420969808
_M_SQRTPI: Final[float] = 1.77245385090551602729816748334114518
_M_PI: Final[float] = 3.14159265358979323846264338327950288

# C++ ``QL_EPSILON``.
_QL_EPSILON: Final[float] = 2.2204460492503131e-16

_CUM_NORMAL_DIST: Final[CumulativeNormalDistribution] = CumulativeNormalDistribution()

# Above this the exercise boundary has run away and the European greeks are
# numerically better. C++ literal: ``else if (q > 12.5)``.
_RUNAWAY_BOUNDARY_Q: Final[float] = 12.5


def _sq(x: float) -> float:
    """``QuantLib::squared``."""
    return x * x


# ---------------------------------------------------------------------------
# The anonymous-namespace helper family from bjerksundstenslandengine.cpp.
# Transcribed expression-for-expression: the groupings matter, floating-point
# addition is not associative, and these are the only place the engine's
# greeks come from.
# ---------------------------------------------------------------------------


def _phi(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi(Real S, Real gamma, Real H, Real I, ...)``."""
    lambda_ = -r_t + gamma * b_t + 0.5 * gamma * (gamma - 1.0) * v
    d = -(math.log(s / h) + (b_t + (gamma - 0.5) * v)) / math.sqrt(v)
    kappa = 2.0 * b_t / v + (2.0 * gamma - 1.0)
    return math.exp(lambda_) * (
        _CUM_NORMAL_DIST(d)
        - math.pow(i / s, kappa) * _CUM_NORMAL_DIST(d - 2.0 * math.log(i / s) / math.sqrt(v))
    )


def _phi_s(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi_S(...)`` — d(phi)/dS."""
    lsh = math.log(s / h)
    lis = math.log(i / s)
    sv = math.sqrt(v)
    p = math.pow(i / s, 2 * (gamma + b_t / v))
    return math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2.0) * (
        (
            -(p / (math.exp(_sq(2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (8.0 * v)) * i))
            - 1 / (math.exp(_sq(2 * b_t - v + 2 * gamma * v + 2 * lsh) / (8.0 * v)) * s)
        )
        / (_M_SQRT2 * _M_SQRTPI * sv)
        + (
            p
            * (2 * b_t + (-1 + 2 * gamma) * v)
            * math.erfc((2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
        )
        / (2.0 * i * v)
    )


def _phi_ss(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi_SS(...)`` — d2(phi)/dS2."""
    lsh = math.log(s / h)
    lis = math.log(i / s)
    sv = math.sqrt(v)
    ex = math.exp(_sq(2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (8.0 * v))
    ey = math.exp(_sq(2 * b_t + (-1 + 2 * gamma) * v + 2 * lsh) / (8.0 * v))
    p = math.pow(i / s, 2 * (gamma + b_t / v))
    return (
        math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2.0)
        * (
            (_M_SQRT2 * i * v * sv) / ey
            + (2 * _M_SQRT2 * p * s * sv * (2 * b_t + (-1 + 2 * gamma) * v)) / ex
            - 2
            * math.sqrt(_M_PI)
            * p
            * s
            * (b_t + gamma * v)
            * (2 * b_t + (-1 + 2 * gamma) * v)
            * math.erfc((2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
            + (_M_SQRT2 * i * sv * (b_t + (-0.5 + gamma) * v + lsh)) / ey
            - (p * s * sv * (2 * b_t - 3 * v + 2 * gamma * v + 4 * lis + 2 * lsh))
            / (_M_SQRT2 * ex)
        )
    ) / (2.0 * i * _M_SQRTPI * _sq(s * v))


def _phi_gamma(
    s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float
) -> float:
    """# C++ parity: ``Real phi_gamma(...)`` — d(phi)/d(gamma)."""
    lsh = math.log(s / h)
    lis = math.log(i / s)
    sv = math.sqrt(v)
    p = math.pow(i / s, -1 + 2 * gamma + (2 * b_t) / v)
    return math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2) * (
        (
            (
                -math.exp(-_sq(2 * b_t - v + 2 * gamma * v + 2 * lsh) / (8 * v))
                + p / math.exp(_sq(2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (8 * v))
            )
            * sv
        )
        / (_M_SQRT2 * _M_SQRTPI)
        + (
            (2 * b_t + (-1 + 2 * gamma) * v)
            * math.erfc((2 * b_t + (-1 + 2 * gamma) * v + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
        )
        / 4.0
        - (
            p
            * math.erfc((2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
            * (2 * b_t + (-1 + 2 * gamma) * v + 4 * lis)
        )
        / 4.0
    )


def _phi_h(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi_H(...)`` — d(phi)/dH."""
    lsh = math.log(s / h)
    return (
        math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2.0)
        * (
            i / math.exp(_sq(2 * b_t - v + 2 * gamma * v + 2 * lsh) / (8.0 * v))
            - (math.pow(i / s, 2 * (gamma + b_t / v)) * s)
            / math.exp(
                _sq(2 * b_t - v + 2 * gamma * v + 4 * math.log(i / s) + 2 * lsh) / (8.0 * v)
            )
        )
    ) / (h * i * math.sqrt(2 * _M_PI) * math.sqrt(v))


def _phi_i(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi_I(...)`` — d(phi)/dI."""
    lsh = math.log(s / h)
    lis = math.log(i / s)
    sv = math.sqrt(v)
    return (
        math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2.0)
        * math.pow(i / s, 2 * (gamma + b_t / v))
        * s
        * (
            (2 * math.sqrt(2 / _M_PI))
            / (math.exp(_sq(2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (8.0 * v)) * sv)
            + (1 - 2 * gamma - (2 * b_t) / v)
            * math.erfc((2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
        )
    ) / (2.0 * i * i)


def _phi_rt(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi_rt(...)`` — d(phi)/d(rT)."""
    lsh = math.log(s / h)
    return (
        math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2.0)
        * (
            -(i * math.erfc((2 * b_t - v + 2 * gamma * v + 2 * lsh) / (2.0 * math.sqrt(2 * v))))
            + math.pow(i / s, 2 * (gamma + b_t / v))
            * s
            * math.erfc(
                (2 * b_t - v + 2 * gamma * v + 4 * math.log(i / s) + 2 * lsh)
                / (2.0 * math.sqrt(2 * v))
            )
        )
    ) / (2.0 * i)


def _phi_bt(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi_bt(...)`` — d(phi)/d(bT)."""
    lsh = math.log(s / h)
    lis = math.log(i / s)
    sv = math.sqrt(v)
    p = math.pow(i / s, 2 * (gamma + b_t / v))
    return (
        math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2.0)
        * (
            _M_SQRT2
            * (
                -(i / math.exp(_sq(2 * b_t - v + 2 * gamma * v + 2 * lsh) / (8.0 * v)))
                + (p * s)
                / math.exp(_sq(2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (8.0 * v))
            )
            * sv
            + gamma
            * i
            * math.sqrt(_M_PI)
            * v
            * math.erfc((2 * b_t - v + 2 * gamma * v + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
            - _M_SQRTPI
            * p
            * s
            * math.erfc((2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
            * (gamma * v + 2 * lis)
        )
    ) / (2.0 * i * math.sqrt(_M_PI) * v)


def _phi_v(s: float, gamma: float, h: float, i: float, r_t: float, b_t: float, v: float) -> float:
    """# C++ parity: ``Real phi_v(...)`` — d(phi)/d(variance)."""
    lsh = math.log(s / h)
    lis = math.log(i / s)
    sv = math.sqrt(v)
    er = math.erfc((2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
    p2 = math.pow(i / s, 2 * (gamma + b_t / v))
    p1 = math.pow(i / s, -1 + 2 * gamma + (2 * b_t) / v)
    return (
        math.exp(b_t * gamma - r_t + ((-1 + gamma) * gamma * v) / 2.0)
        * (
            (
                (-1 + gamma)
                * gamma
                * (
                    i * math.erfc((2 * b_t - v + 2 * gamma * v + 2 * lsh) / (2.0 * _M_SQRT2 * sv))
                    - p2 * s * er
                )
            )
            / (2.0 * i)
            + (2 * b_t * p1 * er * lis) / (v * v)
            + (2 * b_t + v - 2 * gamma * v + 2 * lsh)
            / (
                2.0
                * math.exp(math.pow(2 * b_t + (-1 + 2 * gamma) * v + 2 * lsh, 2) / (8.0 * v))
                * _M_SQRT2
                * _M_SQRTPI
                * v
                * sv
            )
            - (p1 * (2 * b_t + v - 2 * gamma * v + 4 * lis + 2 * lsh))
            / (
                2.0
                * math.exp(_sq(2 * b_t - v + 2 * gamma * v + 4 * lis + 2 * lsh) / (8.0 * v))
                * _M_SQRT2
                * _M_SQRTPI
                * v
                * sv
            )
        )
    ) / 2.0


def _copy_results(dst: OneAssetOptionResults, src: OneAssetOptionResults) -> None:
    """Stand-in for the C++ ``results_ = <local results>`` struct assignment."""
    dst.value = src.value
    dst.error_estimate = src.error_estimate
    dst.delta = src.delta
    dst.gamma = src.gamma
    dst.theta = src.theta
    dst.vega = src.vega
    dst.rho = src.rho
    dst.dividend_rho = src.dividend_rho
    dst.itm_cash_probability = src.itm_cash_probability
    dst.delta_forward = src.delta_forward
    dst.elasticity = src.elasticity
    dst.theta_per_day = src.theta_per_day
    dst.strike_sensitivity = src.strike_sensitivity
    dst.additional_results = dict(src.additional_results)


class BjerksundStenslandApproximationEngine(
    GenericEngine[OptionArguments, OneAssetOptionResults]
):
    """Bjerksund-Stensland American option approximation engine.

    # C++ parity: ``class BjerksundStenslandApproximationEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    # -- result builders ---------------------------------------------------

    def european_call_results(
        self, s: float, x: float, rf_d: float, d_d: float, variance: float
    ) -> OneAssetOptionResults:
        """European call value + greeks via :class:`BlackCalculator`.

        # C++ parity: ``BjerksundStenslandApproximationEngine::europeanCallResults``.
        """
        args = self._arguments
        assert args.exercise is not None
        results = OneAssetOptionResults()

        forward_price = s * d_d / rf_d
        black = BlackCalculator.from_type_strike(
            OptionType.Call, x, forward_price, math.sqrt(variance), rf_d
        )

        results.value = black.value()
        results.delta = black.delta(s)
        results.gamma = black.gamma(s)

        process = self._process
        rfdc = process.risk_free_rate().day_counter()
        divdc = process.dividend_yield().day_counter()
        voldc = process.black_volatility().day_counter()
        last_date = args.exercise.last_date()

        t = rfdc.year_fraction(process.risk_free_rate().reference_date(), last_date)
        results.rho = black.rho(t)

        t = divdc.year_fraction(process.dividend_yield().reference_date(), last_date)
        results.dividend_rho = black.dividend_rho(t)

        t = voldc.year_fraction(process.black_volatility().reference_date(), last_date)
        results.vega = black.vega(t)
        results.theta = black.theta(s, t)
        results.theta_per_day = black.theta_per_day(s, t)

        results.strike_sensitivity = black.strike_sensitivity()
        assert results.gamma is not None
        results.additional_results["strikeGamma"] = results.gamma * _sq(s / x)
        results.additional_results["exerciseType"] = "European"

        return results

    @staticmethod
    def immediate_exercise(s: float, x: float) -> OneAssetOptionResults:
        """Intrinsic value with all sensitivities zeroed.

        # C++ parity: ``BjerksundStenslandApproximationEngine::immediateExercise``.
        """
        results = OneAssetOptionResults()
        results.value = max(0.0, s - x)
        results.delta = 1.0 if s >= x else 0.0
        results.gamma = 0.0
        results.rho = 0.0
        results.dividend_rho = 0.0
        results.vega = 0.0
        results.theta = 0.0
        results.theta_per_day = 0.0

        results.strike_sensitivity = -results.delta
        results.additional_results["strikeGamma"] = 0.0
        results.additional_results["exerciseType"] = "Immediate"

        return results

    def american_call_approximation(  # noqa: PLR0915 - one-to-one with C++
        self, s: float, x: float, rf_d: float, d_d: float, variance: float
    ) -> OneAssetOptionResults:
        """The approximation proper.

        # C++ parity:
        # ``BjerksundStenslandApproximationEngine::americanCallApproximation``.
        """
        args = self._arguments
        assert args.exercise is not None
        european_results = self.european_call_results(s, x, rf_d, d_d, variance)

        b_t = math.log(d_d / rf_d)
        r_t = math.log(1.0 / rf_d)

        beta = (0.5 - b_t / variance) + math.sqrt(
            _sq(b_t / variance - 0.5) + 2.0 * r_t / variance
        )

        b_infinity = beta / (beta - 1.0) * x
        b0 = x if b_t == r_t else max(x, r_t / (r_t - b_t) * x)
        ht = -(b_t + 2.0 * math.sqrt(variance)) * b0 / (b_infinity - b0)

        i = b0 + (b_infinity - b0) * (1 - math.exp(ht))

        fwd = s * d_d / rf_d
        q = math.log(i / fwd) / math.sqrt(variance)

        if s >= i:
            results = self.immediate_exercise(s, x)
        elif q > _RUNAWAY_BOUNDARY_Q:
            # Run-away exercise boundary: the European greeks are numerically
            # more accurate here.
            results = european_results
        else:
            results = OneAssetOptionResults()

            phi_s_beta_i_i = _phi(s, beta, i, i, r_t, b_t, variance)
            phi_s_1_i_i = _phi(s, 1.0, i, i, r_t, b_t, variance)
            phi_s_1_x_i = _phi(s, 1.0, x, i, r_t, b_t, variance)
            results.value = (
                (i - x) * math.pow(s / i, beta) * (1 - phi_s_beta_i_i)
                + s * phi_s_1_i_i
                - s * phi_s_1_x_i
                - x * _phi(s, 0.0, i, i, r_t, b_t, variance)
                + x * _phi(s, 0.0, x, i, r_t, b_t, variance)
            )

            phi_s_s_beta_i_i = _phi_s(s, beta, i, i, r_t, b_t, variance)
            phi_s_s_1_i_i = _phi_s(s, 1.0, i, i, r_t, b_t, variance)
            phi_s_s_1_x_i = _phi_s(s, 1.0, x, i, r_t, b_t, variance)
            results.delta = (
                (i - x) * math.pow(s / i, beta - 1) * beta / i * (1 - phi_s_beta_i_i)
                - (i - x) * math.pow(s / i, beta) * phi_s_s_beta_i_i
                + phi_s_1_i_i
                + s * phi_s_s_1_i_i
                - phi_s_1_x_i
                - s * phi_s_s_1_x_i
                - x * _phi_s(s, 0.0, i, i, r_t, b_t, variance)
                + x * _phi_s(s, 0.0, x, i, r_t, b_t, variance)
            )

            process = self._process
            ref_date = process.risk_free_rate().reference_date()
            exercise_date = args.exercise.last_date()
            qdc = process.dividend_yield().day_counter()
            tq = qdc.year_fraction(ref_date, exercise_date)

            beta_dq = tq * (
                1 / variance
                - 1
                / (2 * math.sqrt(_sq(b_t / variance - 0.5) + 2.0 * r_t / variance))
                * 2
                * (b_t / variance - 0.5)
                / variance
            )
            b_infinity_dq = -x / _sq(beta - 1.0) * beta_dq
            b0_dq = 0.0 if d_d <= rf_d else x * math.log(rf_d) / _sq(math.log(d_d)) * tq

            ht_dq = tq * b0 / (b_infinity - b0) - (b_t + 2.0 * math.sqrt(variance)) * (
                b0_dq * (b_infinity - b0) - b0 * (b_infinity_dq - b0_dq)
            ) / _sq(b_infinity - b0)
            i_dq = (
                b0_dq
                + (b_infinity_dq - b0_dq) * (1 - math.exp(ht))
                - (b_infinity - b0) * math.exp(ht) * ht_dq
            )

            phi_h_s_beta_i_i = _phi_h(s, beta, i, i, r_t, b_t, variance)
            phi_i_s_beta_i_i = _phi_i(s, beta, i, i, r_t, b_t, variance)
            phi_gamma_s_beta_i_i = _phi_gamma(s, beta, i, i, r_t, b_t, variance)
            phi_bt_s_beta_i_i = _phi_bt(s, beta, i, i, r_t, b_t, variance)
            phi_h_s_1_i_i = _phi_h(s, 1.0, i, i, r_t, b_t, variance)
            phi_i_s_1_i_i = _phi_i(s, 1.0, i, i, r_t, b_t, variance)
            phi_bt_s_1_i_i = _phi_bt(s, 1.0, i, i, r_t, b_t, variance)
            phi_i_s_1_x_i = _phi_i(s, 1.0, x, i, r_t, b_t, variance)
            phi_bt_s_1_x_i = _phi_bt(s, 1.0, x, i, r_t, b_t, variance)
            phi_h_s_0_i_i = _phi_h(s, 0.0, i, i, r_t, b_t, variance)
            phi_i_s_0_i_i = _phi_i(s, 0.0, i, i, r_t, b_t, variance)
            phi_bt_s_0_i_i = _phi_bt(s, 0.0, i, i, r_t, b_t, variance)
            phi_i_s_0_x_i = _phi_i(s, 0.0, x, i, r_t, b_t, variance)
            phi_bt_s_0_x_i = _phi_bt(s, 0.0, x, i, r_t, b_t, variance)

            results.dividend_rho = (
                (
                    i_dq * math.pow(s / i, beta)
                    + (i - x)
                    * math.pow(s / i, beta)
                    * (beta_dq * math.log(s / i) - beta * 1 / i * i_dq)
                )
                * (1 - phi_s_beta_i_i)
                - (i - x)
                * math.pow(s / i, beta)
                * (
                    phi_h_s_beta_i_i * i_dq
                    + phi_i_s_beta_i_i * i_dq
                    + phi_gamma_s_beta_i_i * beta_dq
                    - phi_bt_s_beta_i_i * tq
                )
                + s * (phi_h_s_1_i_i * i_dq + phi_i_s_1_i_i * i_dq - phi_bt_s_1_i_i * tq)
                - s * (phi_i_s_1_x_i * i_dq - phi_bt_s_1_x_i * tq)
                - x * (phi_h_s_0_i_i * i_dq + phi_i_s_0_i_i * i_dq - phi_bt_s_0_i_i * tq)
                + x * (phi_i_s_0_x_i * i_dq - phi_bt_s_0_x_i * tq)
            )

            rdc = process.risk_free_rate().day_counter()
            tr = rdc.year_fraction(ref_date, exercise_date)

            beta_dr = tr * (
                -1 / variance
                + 1
                / (2 * math.sqrt(_sq(b_t / variance - 0.5) + 2.0 * r_t / variance))
                * 2
                * ((b_t / variance - 0.5) / variance + 1 / variance)
            )
            b_infinity_dr = -x / _sq(beta - 1.0) * beta_dr
            b0_dr = 0.0 if d_d <= rf_d else -x * tr / math.log(d_d)
            ht_dr = -tr * b0 / (b_infinity - b0) - (b_t + 2.0 * math.sqrt(variance)) * (
                b0_dr * (b_infinity - b0) - b0 * (b_infinity_dr - b0_dr)
            ) / _sq(b_infinity - b0)
            i_dr = (
                b0_dr
                + (b_infinity_dr - b0_dr) * (1 - math.exp(ht))
                - (b_infinity - b0) * math.exp(ht) * ht_dr
            )

            results.rho = (
                (
                    i_dr * math.pow(s / i, beta)
                    + (i - x)
                    * math.pow(s / i, beta)
                    * (beta_dr * math.log(s / i) - beta / i * i_dr)
                )
                * (1 - phi_s_beta_i_i)
                - (i - x)
                * math.pow(s / i, beta)
                * (
                    phi_h_s_beta_i_i * i_dr
                    + phi_i_s_beta_i_i * i_dr
                    + phi_gamma_s_beta_i_i * beta_dr
                    + _phi_rt(s, beta, i, i, r_t, b_t, variance) * tr
                    + phi_bt_s_beta_i_i * tr
                )
                + s
                * (
                    phi_h_s_1_i_i * i_dr
                    + phi_i_s_1_i_i * i_dr
                    + _phi_rt(s, 1.0, i, i, r_t, b_t, variance) * tr
                    + phi_bt_s_1_i_i * tr
                )
                - s
                * (
                    phi_i_s_1_x_i * i_dr
                    + _phi_rt(s, 1.0, x, i, r_t, b_t, variance) * tr
                    + phi_bt_s_1_x_i * tr
                )
                - x
                * (
                    phi_h_s_0_i_i * i_dr
                    + phi_i_s_0_i_i * i_dr
                    + _phi_rt(s, 0.0, i, i, r_t, b_t, variance) * tr
                    + phi_bt_s_0_i_i * tr
                )
                + x
                * (
                    phi_i_s_0_x_i * i_dr
                    + _phi_rt(s, 0.0, x, i, r_t, b_t, variance) * tr
                    + phi_bt_s_0_x_i * tr
                )
            )

            # C++ re-declares ``const Real beta = ...`` here, shadowing the
            # outer one with an identical expression — a no-op, not repeated.

            vdc = process.black_volatility().day_counter()
            tv = vdc.year_fraction(ref_date, exercise_date)
            variance_dv = 2 * math.sqrt(variance * tv)

            beta_dv = (
                b_t / _sq(variance) * variance_dv
                + -1
                / (2 * math.sqrt(_sq(b_t / variance - 0.5) + 2.0 * r_t / variance))
                * (
                    2 * (b_t / variance - 0.5) * b_t * variance_dv / _sq(variance)
                    + 2 * r_t / _sq(variance) * variance_dv
                )
            )
            b_infinity_dv = -x / _sq(beta - 1.0) * beta_dv
            ht_dv = -1 / math.sqrt(variance) * variance_dv * b0 / (b_infinity - b0) + (
                b_t + 2 * math.sqrt(variance)
            ) * b0 / _sq(b_infinity - b0) * b_infinity_dv

            i_dv = b_infinity_dv * (1 - math.exp(ht)) - (b_infinity - b0) * math.exp(ht) * ht_dv

            results.vega = (
                (
                    i_dv * math.pow(s / i, beta)
                    + (i - x)
                    * math.pow(s / i, beta)
                    * (beta_dv * math.log(s / i) - beta / i * i_dv)
                )
                * (1 - phi_s_beta_i_i)
                - (i - x)
                * math.pow(s / i, beta)
                * (
                    phi_h_s_beta_i_i * i_dv
                    + phi_i_s_beta_i_i * i_dv
                    + phi_gamma_s_beta_i_i * beta_dv
                    + _phi_v(s, beta, i, i, r_t, b_t, variance) * variance_dv
                )
                + s
                * (
                    phi_h_s_1_i_i * i_dv
                    + phi_i_s_1_i_i * i_dv
                    + _phi_v(s, 1.0, i, i, r_t, b_t, variance) * variance_dv
                )
                - s * (phi_i_s_1_x_i * i_dv + _phi_v(s, 1.0, x, i, r_t, b_t, variance) * variance_dv)
                - x
                * (
                    phi_h_s_0_i_i * i_dv
                    + phi_i_s_0_i_i * i_dv
                    + _phi_v(s, 0.0, i, i, r_t, b_t, variance) * variance_dv
                )
                + x * (phi_i_s_0_x_i * i_dv + _phi_v(s, 0.0, x, i, r_t, b_t, variance) * variance_dv)
            )

            results.gamma = (
                (i - x)
                * math.pow(s / i, beta - 2)
                * beta
                * (beta - 1)
                / _sq(i)
                * (1 - phi_s_beta_i_i)
                - 2 * (i - x) * math.pow(s / i, beta - 1) * beta / i * phi_s_s_beta_i_i
                - (i - x) * math.pow(s / i, beta) * _phi_ss(s, beta, i, i, r_t, b_t, variance)
                + 2 * phi_s_s_1_i_i
                + s * _phi_ss(s, 1.0, i, i, r_t, b_t, variance)
                - 2 * phi_s_s_1_x_i
                - s * _phi_ss(s, 1.0, x, i, r_t, b_t, variance)
                - x * _phi_ss(s, 0.0, i, i, r_t, b_t, variance)
                + x * _phi_ss(s, 0.0, x, i, r_t, b_t, variance)
            )

            vol = math.sqrt(variance / tv)

            tomorrow: Date = ref_date + 1
            dtq = qdc.year_fraction(ref_date, exercise_date) - qdc.year_fraction(
                tomorrow, exercise_date
            )
            dtr = rdc.year_fraction(ref_date, exercise_date) - rdc.year_fraction(
                tomorrow, exercise_date
            )
            dtv = vdc.year_fraction(ref_date, exercise_date) - vdc.year_fraction(
                tomorrow, exercise_date
            )

            results.theta_per_day = -(
                0.5 * results.vega * vol / tv * dtv
                + results.rho * r_t / (tr * tr) * dtr
                + results.dividend_rho * (r_t - b_t) / (tq * tq) * dtq
            )
            results.theta = 365 * results.theta_per_day

            results.strike_sensitivity = results.value / x - s / x * results.delta
            results.additional_results["strikeGamma"] = results.gamma * _sq(s / x)
            results.additional_results["exerciseType"] = "American"

        # Check if the European engine gives a higher NPV.
        assert results.value is not None
        assert european_results.value is not None
        if results.value < european_results.value:
            results = european_results

        return results

    # -- engine ------------------------------------------------------------

    def calculate(self) -> None:
        """Price the American option.

        # C++ parity: ``BjerksundStenslandApproximationEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.American,
            "not an American Option",
        )
        qassert.require(isinstance(args.exercise, AmericanExercise), "non-American exercise given")
        assert isinstance(args.exercise, AmericanExercise)
        ex: AmericanExercise = args.exercise
        qassert.require(not ex.payoff_at_expiry(), "payoff at expiry not handled")

        qassert.require(args.payoff is not None, "no payoff given")
        assert args.payoff is not None
        qassert.require(isinstance(args.payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(args.payoff, PlainVanillaPayoff)
        original_payoff: PlainVanillaPayoff = args.payoff

        process = self._process
        last_date = ex.last_date()
        variance = process.black_volatility().black_variance(
            last_date, original_payoff.strike(), extrapolate=True
        )
        dividend_discount = process.dividend_yield().discount(last_date)
        risk_free_discount = process.risk_free_rate().discount(last_date)
        spot = process.state_variable().value()
        qassert.require(spot > 0.0, "negative or null underlying given")
        strike = original_payoff.strike()

        is_put = original_payoff.option_type() == OptionType.Put
        if is_put:
            # Put-call symmetry. (C++ also rebuilds `payoff` as a Call here;
            # nothing downstream reads it — the result builders take S and X.)
            spot, strike = strike, spot
            risk_free_discount, dividend_discount = dividend_discount, risk_free_discount

        qassert.require(
            not (dividend_discount > 1.0 and risk_free_discount > dividend_discount),
            "double-boundary case r<q<0 for a call given",
        )

        if dividend_discount >= 1.0 and dividend_discount >= risk_free_discount:
            _copy_results(
                results,
                self.european_call_results(
                    spot, strike, risk_free_discount, dividend_discount, variance
                ),
            )
        else:
            # Early exercise can be optimal - use the approximation.
            _copy_results(
                results,
                self.american_call_approximation(
                    spot, strike, risk_free_discount, dividend_discount, variance
                ),
            )

        # Check if immediate exercise gives a higher NPV.
        assert results.value is not None
        if results.value < (spot - strike) * (1 + 10 * _QL_EPSILON):
            _copy_results(results, self.immediate_exercise(spot, strike))

        if is_put:
            results.delta, results.strike_sensitivity = (
                results.strike_sensitivity,
                results.delta,
            )

            tmp = results.gamma
            results.gamma = float(results.additional_results["strikeGamma"])
            results.additional_results["strikeGamma"] = tmp

            results.rho, results.dividend_rho = results.dividend_rho, results.rho

            tr = process.risk_free_rate().day_counter().year_fraction(
                process.risk_free_rate().reference_date(), last_date
            )
            tq = process.dividend_yield().day_counter().year_fraction(
                process.dividend_yield().reference_date(), last_date
            )

            assert results.rho is not None
            assert results.dividend_rho is not None
            results.rho *= tr / tq
            results.dividend_rho *= tq / tr


__all__ = ["BjerksundStenslandApproximationEngine"]
