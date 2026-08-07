"""CEVCalculator + AnalyticCEVEngine — closed-form constant-elasticity-of-variance prices.

# C++ parity: ql/pricingengines/vanilla/analyticcevengine.{hpp,cpp} (v1.43) —
# ``class CEVCalculator`` and
# ``class AnalyticCEVEngine : public VanillaOption::engine``.

The CEV process with an absorbing boundary at ``f = 0``::

    df_t = alpha * f_t^beta dW_t

has European option prices expressible through non-central chi-squared
distribution functions. Following the C++, the calculator caches

    delta = (1 - 2*beta) / (1 - beta)
    X(f)  = f^(2*(1-beta)) / (alpha*(1-beta))^2
    x0    = X(f0)

and then splits on ``delta < 2`` — equivalently on ``beta < 1``:

* ``beta < 1`` (``delta < 2``): the process is a true martingale, so
  put-call parity ``C - P = f0 - K`` holds exactly.
* ``beta > 1`` (``delta >= 2``): the process is only a *strict local*
  martingale, ``E[F_t] < F_0``, and put-call parity is genuinely
  violated. The call branch picks up an extra ``gamma_p(delta/2 - 1,
  x0/(2t))`` factor that the put branch does not have. That asymmetry is
  the local-martingale defect and must not be "fixed".

Reference: D.R. Brecher, A.E. Lindsay, *Results on the CEV Process, Past
and Present*.

Numerics
--------
C++ calls **Boost** here, not QuantLib's own distributions:
``boost::math::non_central_chi_squared_distribution`` and
``boost::math::gamma_p``. QuantLib's own
``NonCentralCumulativeChiSquareDistribution`` is a different algorithm with an
*absolute* 1e-12 stopping rule and disagrees with Boost by orders of magnitude
in the far tail, so this port must not reach for it. ``scipy.stats.ncx2.cdf``
and ``scipy.special.gammainc`` are used instead, and the choice is
cross-validated rather than assumed: over the 77 ``cev_calc_*`` points in
``migration-harness/references/v143/pe/hybrid.json`` — which include five
deliberately deep-tail evaluations — they reproduce Boost to 4.4e-15 relative
and 2.2e-16 absolute.
"""

from __future__ import annotations

import math
from typing import cast

from scipy.special import gammainc  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from scipy.stats import ncx2  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


def _nc_chi2_cdf(df: float, ncp: float, x: float) -> float:
    """``boost::math::cdf(non_central_chi_squared(df, ncp), x)``."""
    return float(cast("float", ncx2.cdf(x, df, ncp)))  # pyright: ignore[reportUnknownMemberType]


def _gamma_p(a: float, x: float) -> float:
    """``boost::math::gamma_p(a, x)`` — regularized lower incomplete gamma."""
    return float(cast("float", gammainc(a, x)))


class CEVCalculator:
    """Undiscounted European CEV option values.

    # C++ parity: ``class CEVCalculator`` in analyticcevengine.hpp:47-61.

    A genuine top-level public class in C++, not a nested helper. Its public
    read surface is exactly ``f0()`` / ``alpha()`` / ``beta()`` plus
    :meth:`value`; ``X(f)`` and the cached ``delta_`` are private there and
    stay private here.
    """

    __slots__ = ("_alpha", "_beta", "_delta", "_f0", "_x0")

    def __init__(self, f0: float, alpha: float, beta: float) -> None:
        """Cache ``delta`` and ``x0 = X(f0)``.

        # C++ parity: analyticcevengine.cpp:31-36.
        """
        self._f0: float = float(f0)
        self._alpha: float = float(alpha)
        self._beta: float = float(beta)
        # C++ parity: delta_((1.0-2.0*beta)/(1.0-beta)).
        self._delta: float = (1.0 - 2.0 * self._beta) / (1.0 - self._beta)
        # C++ parity: x0_(X(f0)) — computed after delta_ in the ctor init list.
        self._x0: float = self._x(self._f0)

    # --- inspectors -----------------------------------------------------

    def f0(self) -> float:
        """# C++ parity: ``CEVCalculator::f0`` at analyticcevengine.hpp:53."""
        return self._f0

    def alpha(self) -> float:
        """# C++ parity: ``CEVCalculator::alpha`` at analyticcevengine.hpp:54."""
        return self._alpha

    def beta(self) -> float:
        """# C++ parity: ``CEVCalculator::beta`` at analyticcevengine.hpp:55."""
        return self._beta

    # --- private --------------------------------------------------------

    def _x(self, f: float) -> float:
        """``X(f) = f^(2(1-beta)) / (alpha*(1-beta))^2``.

        # C++ parity: ``CEVCalculator::X`` at analyticcevengine.cpp:38-40
        # (private there; ``squared(alpha_*(1.0-beta_))`` is the denominator).
        """
        denominator = self._alpha * (1.0 - self._beta)
        return math.pow(f, 2.0 * (1.0 - self._beta)) / (denominator * denominator)

    # --- value ----------------------------------------------------------

    def value(self, option_type: OptionType, strike: float, t: float) -> float:
        """Undiscounted CEV value of a European option.

        # C++ parity: ``CEVCalculator::value`` at analyticcevengine.cpp:42-84,
        # transcribed branch for branch.
        """
        k_tilde = self._x(strike)
        delta = self._delta
        f0 = self._f0
        x0 = self._x0

        if option_type == OptionType.Call:
            if delta < 2.0:
                return f0 * (
                    1.0 - _nc_chi2_cdf(4.0 - delta, x0 / t, k_tilde / t)
                ) - strike * _nc_chi2_cdf(2.0 - delta, k_tilde / t, x0 / t)
            # C++ parity: analyticcevengine.cpp:60 — the extra gamma_p factor
            # exists only on the beta > 1 CALL branch.
            g = _gamma_p(0.5 * delta - 1.0, x0 / (2.0 * t))
            return f0 * (
                g - _nc_chi2_cdf(delta - 2.0, k_tilde / t, x0 / t)
            ) - strike * _nc_chi2_cdf(delta, x0 / t, k_tilde / t)

        if option_type == OptionType.Put:
            if delta < 2.0:
                return -f0 * _nc_chi2_cdf(4.0 - delta, x0 / t, k_tilde / t) + strike * (
                    1.0 - _nc_chi2_cdf(2.0 - delta, k_tilde / t, x0 / t)
                )
            return -f0 * _nc_chi2_cdf(delta - 2.0, k_tilde / t, x0 / t) + strike * (
                1.0 - _nc_chi2_cdf(delta, x0 / t, k_tilde / t)
            )

        # C++ parity: analyticcevengine.cpp:83 — QL_FAIL("unknown option type").
        raise LibraryException(f"unknown option type: {option_type}")


class AnalyticCEVEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """European vanilla engine wrapping a :class:`CEVCalculator`.

    # C++ parity: ``class AnalyticCEVEngine : public VanillaOption::engine``
    # in analyticcevengine.hpp:64-74.

    Fills ``results.value`` and nothing else — no Greeks, no additional
    results — exactly as C++ ``AnalyticCEVEngine::calculate`` does.
    """

    def __init__(
        self,
        f0: float,
        alpha: float,
        beta: float,
        discount_curve: YieldTermStructure,
    ) -> None:
        """# C++ parity: analyticcevengine.cpp:86-93."""
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._calculator: CEVCalculator = CEVCalculator(f0, alpha, beta)
        self._discount_curve: YieldTermStructure = discount_curve
        # C++ parity: analyticcevengine.cpp:92 — registerWith(discountCurve_).
        discount_curve.register_with(self)

    def calculator(self) -> CEVCalculator:
        """The wrapped calculator (``calculator_`` in C++, private there)."""
        return self._calculator

    def calculate(self) -> None:
        """# C++ parity: ``AnalyticCEVEngine::calculate`` at analyticcevengine.cpp:95-113."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European option",
        )
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        exercise_date = args.exercise.last_date()

        results.reset()
        results.value = self._calculator.value(
            payoff.option_type(),
            payoff.strike(),
            self._discount_curve.time_from_reference(exercise_date),
        ) * self._discount_curve.discount(exercise_date)


__all__ = ["AnalyticCEVEngine", "CEVCalculator"]
