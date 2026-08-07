"""ChoiBasketEngine — Choi (2018) "Sum of all Black-Scholes-Merton Models".

# C++ parity:
# ql/pricingengines/basket/choibasketengine.{hpp,cpp} (v1.43),
# ``QuantLib::ChoiBasketEngine``.

Jaehyuk Choi, *Sum of all Black-Scholes-Merton Models: An efficient Pricing
Method for Spread, Basket and Asian Options*,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2913048

The method rotates the ``n`` correlated log-normal drivers so that one
direction (``vStar1``) carries the bulk of the basket's variance, prices that
direction exactly with :class:`SingleFactorBsmBasketEngine`, and integrates the
remaining ``n - 1`` directions with a tensor-product Gauss-Hermite rule whose
per-dimension order is chosen from the singular values.

The four knobs all move the answer and are all pinned by the probe:

``lambda``
    Sets the quadrature orders via ``round(1 + lambda * alpha * sv[i])``.
``maxNrIntegrationSteps``
    Caps the *product* of those orders. When the cap is exceeded C++ does not
    truncate — it rescales ``lambda`` by ``0.9`` and retries, failing with
    "can not rescale lambda to fit max integration order" once ``lambda`` has
    shrunk by ``1e-10``. So the two knobs interact.
``calcfwdDelta``
    Populates ``additional_results["forwardDelta k"]`` for each ``k``.
``controlVariate``
    Subtracts ``sum_k fwdDelta_k * fwd_k * (fHat_k - 1)`` from the value —
    **and forces the deltas on**, because C++ stores
    ``calcFwdDelta_ = calcfwdDelta || controlVariate``.

Two more details a port gets wrong easily:

* the std-devs come from ``sqrt(getBlackVariance(...))`` — *unsigned*, unlike
  ``SingleFactorBsmBasketEngine``, which uses the signed ``getBlackStdDev`` —
  and are floored at ``QL_EPSILON**2``;
* when ``sign(g[i]) * vStar1[i] < tol * stdDev[i]`` the component is replaced
  by ``eps * sign(g[i]) * stdDev[i]`` and ``q1`` is then obtained by explicit
  forward substitution against the Cholesky factor rather than by
  ``q1 = C^T g``. The four-asset golden market exercises that branch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import (
    AverageBasketPayoff,
    BasketOption,
    BasketOptionResults,
    SpreadBasketPayoff,
)
from pquantlib.math.array import Array
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.integrals.gaussian_quadrature import (
    GaussHermiteIntegration,
    GaussianQuadrature,
    MultiDimGaussianIntegration,
)
from pquantlib.math.matrix import Matrix
from pquantlib.math.matrixutilities.cholesky import cholesky_decomposition
from pquantlib.math.matrixutilities.get_covariance import get_covariance
from pquantlib.math.matrixutilities.householder import (
    HouseholderReflection,
    HouseholderTransformation,
)
from pquantlib.math.matrixutilities.svd import SVD
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.single_factor_bsm_basket_engine import (
    SingleFactorBsmBasketEngine,
)
from pquantlib.pricingengines.basket.vector_bsm_process_extractor import (
    VectorBsmProcessExtractor,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.black_process import BlackProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)

_QL_EPSILON: float = float(np.finfo(np.float64).eps)
_M_SQRT2: float = math.sqrt(2.0)

# C++ default for maxNrIntegrationSteps is std::numeric_limits<Size>::max().
MAX_NR_INTEGRATION_STEPS_UNBOUNDED: int = (1 << 64) - 1


def _sign(x: float) -> float:
    """``boost::math::sign`` — -1, 0 or +1."""
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


class ChoiBasketEngine(GenericEngine[OptionArguments, BasketOptionResults]):
    """Basket engine on ``n`` correlated Black-Scholes underlyings.

    # C++ parity: ``ChoiBasketEngine`` (choibasketengine.hpp:46-66,
    # choibasketengine.cpp:40-266).

    Args:
        processes: one process per basket leg.
        rho: ``n x n`` correlation matrix.
        lambda_: integration-order scale. C++ default ``10.0``; must be > 0.
        max_nr_integration_steps: cap on the product of the per-dimension
            quadrature orders. C++ default is unbounded.
        calc_fwd_delta: populate ``additional_results["forwardDelta k"]``.
        control_variate: apply the forward-delta control variate (implies
            ``calc_fwd_delta``).
    """

    def __init__(
        self,
        processes: Sequence[GeneralizedBlackScholesProcess],
        rho: Matrix,
        lambda_: float = 10.0,
        max_nr_integration_steps: int = MAX_NR_INTEGRATION_STEPS_UNBOUNDED,
        calc_fwd_delta: bool = False,
        control_variate: bool = False,
    ) -> None:
        # C++ parity: choibasketengine.cpp:40-61.
        super().__init__(OptionArguments(), BasketOptionResults())
        self._processes: tuple[GeneralizedBlackScholesProcess, ...] = tuple(processes)
        self._n: int = len(self._processes)
        self._rho: Matrix = np.asarray(rho, dtype=np.float64)
        self._lambda: float = lambda_
        self._max_nr_integration_steps: int = max_nr_integration_steps
        # NB: C++ stores calcFwdDelta_ = (calcfwdDelta || controlVariate).
        self._calc_fwd_delta: bool = calc_fwd_delta or control_variate
        self._control_variate: bool = control_variate

        qassert.require(self._n > 0, "No Black-Scholes process is given.")
        qassert.require(
            self._rho.ndim == 2
            and self._n == self._rho.shape[0]
            and self._rho.shape[0] == self._rho.shape[1],
            "process and correlation matrix must have the same size.",
        )
        qassert.require(self._lambda > 0.0, "lambda must be positive")

        for p in self._processes:
            p.register_with(self)

    # --- helpers ------------------------------------------------------------

    def _average_payoff(self) -> AverageBasketPayoff:
        """The payoff, with a ``SpreadBasketPayoff`` rewritten as ``{1, -1}``.

        # C++ parity: choibasketengine.cpp:82-93.
        """
        payoff = self._arguments.payoff
        if isinstance(payoff, AverageBasketPayoff):
            return payoff
        if isinstance(payoff, SpreadBasketPayoff):
            return AverageBasketPayoff(payoff.base_payoff(), [1.0, -1.0])
        qassert.fail("average or spread basket payoff expected")
        raise AssertionError  # pragma: no cover - qassert.fail always raises

    def calculate(self) -> None:  # noqa: PLR0915  (one long C++ function)
        """Price the basket.

        # C++ parity: ``ChoiBasketEngine::calculate`` (choibasketengine.cpp:63-266).
        """
        args = self._arguments
        results = self._results
        results.reset()

        exercise = args.exercise
        qassert.require(exercise is not None, "not an European exercise")
        assert exercise is not None
        qassert.require(
            exercise.type() == Exercise.Type.European, "not an European exercise"
        )
        assert isinstance(exercise, EuropeanExercise)
        maturity_date = exercise.last_date()

        extractor = VectorBsmProcessExtractor(self._processes)
        s = extractor.get_spot()
        dq = extractor.get_dividend_yield_df(maturity_date)
        dr0 = extractor.get_interest_rate_df(maturity_date)

        std_dev: Array = np.sqrt(extractor.get_black_variance(maturity_date))
        std_dev = np.maximum(_QL_EPSILON * _QL_EPSILON, std_dev)

        fwd: Array = s * dq / dr0

        avg_payoff = self._average_payoff()
        weights: Array = avg_payoff.weights()
        qassert.require(
            self._n == int(weights.shape[0]) and self._n > 1,
            "wrong number of weights arguments in payoff",
        )

        n = self._n
        wf: Array = weights * fwd
        g: Array = wf / math.sqrt(float(np.dot(wf, wf)))

        sigma: Matrix = get_covariance([float(x) for x in std_dev], self._rho)
        v_star1: Array = sigma @ g
        v_star1 = v_star1 / math.sqrt(float(np.dot(g, v_star1)))

        chol: Matrix = cholesky_decomposition(sigma)

        eps = 100 * math.sqrt(_QL_EPSILON)
        # publication sets tol=0, pyfeng implementation sets tol=0.01
        tol = 100 * math.sqrt(_QL_EPSILON)

        flip = False
        for i in range(n):
            if _sign(float(g[i])) * float(v_star1[i]) < tol * float(std_dev[i]):
                flip = True
                v_star1[i] = eps * _sign(float(g[i])) * float(std_dev[i])

        q1: Array = np.zeros(n, dtype=np.float64)
        if flip:
            # q1 = inverse(C) * vStar1 by forward substitution.
            for i in range(n):
                acc = 0.0
                for jj in range(i):
                    acc += float(chol[i, jj]) * float(q1[jj])
                q1[i] = (float(v_star1[i]) - acc) / float(chol[i, i])
            v_star1 = v_star1 / math.sqrt(float(np.dot(q1, q1)))
        else:
            q1 = chol.T @ g
        q1 = q1 / math.sqrt(float(np.dot(q1, q1)))

        e1: Array = np.zeros(n, dtype=np.float64)
        e1[0] = 1.0

        r_mat: Matrix = HouseholderTransformation(
            HouseholderReflection(e1).reflection_vector(q1)
        ).get_matrix()
        r_2_n: Matrix = r_mat[:, 1:]

        svd = SVD(chol @ r_2_n)
        u_mat = svd.u()
        sv = svd.singular_values()

        v_mat: Matrix = np.zeros((n, n - 1), dtype=np.float64)
        for i in range(n - 1):
            v_mat[:, i] = float(sv[i]) * u_mat[:, i]

        # lambda rescaling loop. C++ mutates lambda inside the do/while and
        # requires lambda/lambda_ > 1e-10 *before* the loop condition is tested.
        n_int_order: list[int] = [0] * (n - 1)
        lambda_ = self._lambda
        alpha = 1.0 / abs(float(np.dot(g, v_star1)))
        while True:
            int_scale = lambda_ * alpha
            for i in range(n - 1):
                n_int_order[i] = _lround(1.0 + int_scale * float(sv[i]))
            lambda_ *= 0.9
            qassert.require(
                lambda_ / self._lambda > 1e-10,
                "can not rescale lambda to fit max integration order",
            )
            product = 1.0
            for e in n_int_order:
                product *= e
            if product <= float(self._max_nr_integration_steps):
                break

        quotes: list[SimpleQuote] = []
        inner_processes: list[GeneralizedBlackScholesProcess] = []
        for i in range(n):
            quotes.append(SimpleQuote(float(fwd[i])))
            bv = self._processes[i].black_volatility()
            vol = float(v_star1[i]) / math.sqrt(
                bv.day_counter().year_fraction(bv.reference_date(), maturity_date)
            )
            inner_processes.append(
                BlackProcess(
                    x0=quotes[i],
                    risk_free_ts=self._processes[i].risk_free_rate(),
                    black_vol_ts=BlackConstantVol(
                        reference_date=bv.reference_date(),
                        calendar=bv.calendar(),
                        day_counter=bv.day_counter(),
                        volatility=SimpleQuote(vol),
                    ),
                )
            )

        option = BasketOption(avg_payoff, exercise)
        option.set_pricing_engine(SingleFactorBsmBasketEngine(inner_processes))

        vq: Array = np.zeros(n, dtype=np.float64)
        for i in range(n):
            vq[i] = 0.5 * float(np.dot(v_mat[i, :], v_mat[i, :]))

        ghq = MultiDimGaussianIntegration(n_int_order, _gauss_hermite)
        norm_factor = math.pow(math.pi, -0.5 * len(n_int_order))

        d_store: list[float] = []

        def bsm1d_pricer(z: Array) -> float:
            # C++ parity: choibasketengine.cpp:211-219.
            f = np.exp(-_M_SQRT2 * (v_mat @ z) - vq) * fwd
            for i in range(int(f.shape[0])):
                quotes[i].set_value(float(f[i]))
            npv = option.npv()
            d_store.append(float(option.additional_results()["d"]))
            return math.exp(-float(np.dot(z, z))) * npv

        results.value = ghq(bsm1d_pricer) * norm_factor

        if not self._calc_fwd_delta:
            return

        payoff = avg_payoff.base_payoff()
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain vanilla payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)
        put_indicator = 0.0 if payoff.option_type() == OptionType.Call else -1.0

        n_dist = CumulativeNormalDistribution()
        fwd_delta: Array = np.zeros(n, dtype=np.float64)
        f_hat: Array = np.zeros(n, dtype=np.float64)

        for k in range(n):
            counter = [0]

            def delta_pricer(z: Array, k: int = k, counter: list[int] = counter) -> float:
                # C++ parity: choibasketengine.cpp:236-243.
                d = d_store[counter[0]]
                counter[0] += 1
                vz = float(np.dot(v_mat[k, :], z))
                f = math.exp(-_M_SQRT2 * vz - float(vq[k]))
                return (
                    math.exp(-float(np.dot(z, z)))
                    * f
                    * n_dist(d + float(v_star1[k]))
                )

            fwd_delta[k] = (
                dr0
                * float(weights[k])
                * (ghq(delta_pricer) * norm_factor + put_indicator)
            )
            results.additional_results[f"forwardDelta {k}"] = float(fwd_delta[k])

        if self._control_variate:
            for k in range(n):

                def f_hat_pricer(z: Array, k: int = k) -> float:
                    # C++ parity: choibasketengine.cpp:253-259.
                    vz = float(np.dot(v_mat[k, :], z))
                    f = math.exp(-_M_SQRT2 * vz - float(vq[k]))
                    return math.exp(-float(np.dot(z, z))) * f

                f_hat[k] = ghq(f_hat_pricer) * norm_factor

            cv = fwd_delta * fwd * (f_hat - 1.0)
            assert results.value is not None
            results.value -= float(np.sum(cv))

    def update(self) -> None:
        self.notify_observers()


def _lround(x: float) -> int:
    """``std::lround`` — half away from zero, unlike Python's banker's rounding."""
    return math.floor(x + 0.5) if x >= 0.0 else -math.floor(-x + 0.5)


def _gauss_hermite(m: int) -> GaussianQuadrature:
    # C++ parity: the lambda passed to MultiDimGaussianIntegration
    # (choibasketengine.cpp:203-206).
    return GaussHermiteIntegration(m)


__all__ = ["MAX_NR_INTEGRATION_STEPS_UNBOUNDED", "ChoiBasketEngine"]
