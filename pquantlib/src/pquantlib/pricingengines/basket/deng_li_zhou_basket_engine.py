"""DengLiZhouBasketEngine — Deng-Li-Zhou (2008) multi-asset spread option.

# C++ parity:
# ql/pricingengines/basket/denglizhoubasketengine.{hpp,cpp} (v1.43),
# ``QuantLib::DengLiZhouBasketEngine``.

S. Deng, M. Li, J. Zhou, *Multi-asset Spread Option Pricing and Hedging* (2008),
https://mpra.ub.uni-muenchen.de/8259/1/MPRA_paper_8259.pdf, with the typo in
formula (37) for ``J^2`` corrected — the correction is in the C++ source and is
reproduced here.

The closed form only works when **exactly one** asset weight is positive. When
more than one is, C.F. Lo's *WKB Approximation for the Sum of Two Correlated
Lognormal Random Variables* (2013) is used to collapse the positive legs onto a
single log-normal proxy. Both branches are exercised by the probe (``dlz_m2_*``
and ``dlz_m3_*`` take the ``M > 1`` branch, ``dlz_upstream_*`` the ``M == 1``
one).

Details a port gets wrong easily, all pinned:

* the ``(weight, index, spot, dividend_df, variance)`` tuples are sorted with
  C++'s ``std::greater<>``, i.e. **lexicographically descending on the whole
  tuple**, not on the weight alone. Ties on the weight break on the index, then
  on the spot;
* ``M`` counts *strictly* positive weights (``lower_bound`` with ``w > 0``), so
  a weight of exactly ``0`` counts as negative for the "at least one negative"
  guard and lands in the ``N`` block;
* a **negative strike** appends a synthetic asset ``(1.0, n, -K, dr0, 0.0)``
  with zero correlation to everything, grows ``rho`` to ``(n+1)x(n+1)`` and
  prices with ``K = 0``;
* the put is ``max(0, call - fwd)`` with ``fwd`` computed from the *reduced*
  ``_s``/``_dq`` arrays, not from the original basket.

``pseudoSqrt(..., SalvagingAlgorithm::Principal)`` has no PQuantLib counterpart
yet (``models/marketmodels/models/pseudo_sqrt.py`` carries only
``rank_reduced_sqrt`` and a two-value ``SalvagingAlgorithm``), so the Principal
branch is transcribed here as :func:`_pseudo_sqrt_principal`. It is the
*symmetric* square root ``V diag(sqrt(max(lambda, 0))) V^T``, which is unique
and therefore independent of the eigen-solver's column ordering and signs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import (
    AverageBasketPayoff,
    BasketOptionResults,
    SpreadBasketPayoff,
)
from pquantlib.math.array import Array
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.matrix import Matrix
from pquantlib.math.matrixutilities.cholesky import (
    cholesky_decomposition,
    cholesky_solve_for,
)
from pquantlib.math.matrixutilities.symmetric_schur_decomposition import (
    SymmetricSchurDecomposition,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.vector_bsm_process_extractor import (
    VectorBsmProcessExtractor,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)

_QL_EPSILON: float = float(np.finfo(np.float64).eps)


def _pseudo_sqrt_principal(m: Matrix) -> Matrix:
    """``pseudoSqrt(m, SalvagingAlgorithm::Principal)``.

    # C++ parity: ``pseudoSqrt`` Principal branch
    # (ql/math/matrixutilities/pseudosqrt.cpp:422-448).

    ``V diag(sqrt(max(lambda, 0))) V^T``, symmetrised. C++ requires the
    smallest eigenvalue to be at least ``-10 * QL_EPSILON``.
    """
    size = int(m.shape[0])
    jd = SymmetricSchurDecomposition(m)
    eigenvalues = jd.eigenvalues()
    qassert.require(
        float(eigenvalues[size - 1]) >= -10 * _QL_EPSILON,
        f"negative eigenvalue(s) ({float(eigenvalues[size - 1]):e})",
    )
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0.0))
    vectors = jd.eigenvectors()
    # C++ builds `diagonal[k][i] = sqrtEigenvalues[k] * V[i][k]` then multiplies
    # V * diagonal, i.e. V diag(sqrt lambda) V^T.
    diagonal: Matrix = (vectors * sqrt_eigenvalues[np.newaxis, :]).T
    result: Matrix = vectors @ diagonal
    return 0.5 * (result + result.T)


class DengLiZhouBasketEngine(GenericEngine[OptionArguments, BasketOptionResults]):
    """Second-order expansion for spread options on a mixed-sign basket.

    # C++ parity: ``DengLiZhouBasketEngine`` (denglizhoubasketengine.hpp:54-72,
    # denglizhoubasketengine.cpp:31-291).

    Args:
        processes: one process per basket leg.
        rho: ``n x n`` correlation matrix.
    """

    def __init__(
        self,
        processes: Sequence[GeneralizedBlackScholesProcess],
        rho: Matrix,
    ) -> None:
        # C++ parity: denglizhoubasketengine.cpp:31-44.
        super().__init__(OptionArguments(), BasketOptionResults())
        self._processes: tuple[GeneralizedBlackScholesProcess, ...] = tuple(processes)
        self._n: int = len(self._processes)
        self._rho: Matrix = np.asarray(rho, dtype=np.float64)

        qassert.require(self._n > 0, "No Black-Scholes process is given.")
        qassert.require(
            self._rho.ndim == 2
            and self._n == self._rho.shape[0]
            and self._rho.shape[0] == self._rho.shape[1],
            "process and correlation matrix must have the same size.",
        )
        for p in self._processes:
            p.register_with(self)

    # --- helpers ------------------------------------------------------------

    def _average_payoff(self) -> AverageBasketPayoff:
        """# C++ parity: denglizhoubasketengine.cpp:52-63."""
        payoff = self._arguments.payoff
        if isinstance(payoff, AverageBasketPayoff):
            return payoff
        if isinstance(payoff, SpreadBasketPayoff):
            return AverageBasketPayoff(payoff.base_payoff(), [1.0, -1.0])
        qassert.fail("average or spread basket payoff expected")
        raise AssertionError  # pragma: no cover - qassert.fail always raises

    def calculate(self) -> None:  # noqa: PLR0915  (one long C++ function)
        """Price the basket.

        # C++ parity: ``DengLiZhouBasketEngine::calculate``
        # (denglizhoubasketengine.cpp:46-180).
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

        avg_payoff = self._average_payoff()
        weights: Array = avg_payoff.weights()
        qassert.require(
            self._n == int(weights.shape[0]) and self._n > 1,
            "wrong number of weights arguments in payoff",
        )

        extractor = VectorBsmProcessExtractor(self._processes)
        s = extractor.get_spot()
        dq = extractor.get_dividend_yield_df(maturity_date)
        v = extractor.get_black_variance(maturity_date)
        dr0 = extractor.get_interest_rate_df(maturity_date)

        # (weight, index, spot, dividend_df, variance)
        p: list[tuple[float, int, float, float, float]] = [
            (float(weights[i]), i, float(s[i]), float(dq[i]), float(v[i]))
            for i in range(self._n)
        ]

        payoff = avg_payoff.base_payoff()
        qassert.require(
            isinstance(payoff, PlainVanillaPayoff), "non-plain vanilla payoff given"
        )
        assert isinstance(payoff, PlainVanillaPayoff)

        if payoff.strike() < 0.0:
            # C++ parity: denglizhoubasketengine.cpp:86-94.
            p.append((1.0, self._n, -payoff.strike(), dr0, 0.0))
            rho: Matrix = np.zeros((self._n + 1, self._n + 1), dtype=np.float64)
            rho[: self._n, : self._n] = self._rho
            rho[self._n, self._n] = 1.0
        else:
            rho = self._rho

        strike = max(0.0, payoff.strike())

        # C++ std::sort with std::greater<> on the WHOLE tuple.
        p.sort(reverse=True)

        m_count = 0
        for entry in p:
            if entry[0] > 0.0:
                m_count += 1
            else:
                break

        qassert.require(m_count > 0, "at least one positive asset weight must be given")
        qassert.require(
            m_count < len(p), "at least one negative asset weight must be given"
        )

        n_count = len(p) - m_count

        n_rho: Matrix = np.zeros((n_count + 1, n_count + 1), dtype=np.float64)
        _s: Array = np.zeros(n_count + 1, dtype=np.float64)
        _dq: Array = np.zeros(n_count + 1, dtype=np.float64)
        _v: Array = np.zeros(n_count + 1, dtype=np.float64)

        if m_count > 1:
            # Lo (2013) WKB collapse of the positive legs onto one lognormal.
            # C++ parity: denglizhoubasketengine.cpp:117-151.
            f_arr: Array = np.zeros(m_count, dtype=np.float64)
            vol: Array = np.zeros(m_count, dtype=np.float64)
            for i in range(m_count):
                vol[i] = math.sqrt(p[i][4])
                f_arr[i] = p[i][0] * p[i][2] * p[i][3] / dr0

            s0 = 0.0
            for i in range(m_count):
                s0 += p[i][0] * p[i][2]
            f0 = float(np.sum(f_arr))
            dq_s0 = f0 / s0 * dr0

            v_s = 0.0
            for i in range(m_count):
                for j in range(m_count):
                    v_s += (
                        float(vol[i])
                        * float(vol[j])
                        * float(f_arr[i])
                        * float(f_arr[j])
                        * float(rho[p[i][1], p[j][1]])
                    )
            v_s /= f0 * f0
            _s[0], _dq[0], _v[0] = s0, dq_s0, v_s

            n_rho[0, 0] = 1.0

            for i in range(n_count):
                rho_hat = 0.0
                for j in range(m_count):
                    rho_hat += (
                        float(rho[p[m_count + i][1], p[j][1]])
                        * float(vol[j])
                        * float(f_arr[j])
                    )
                n_rho[i + 1, 0] = n_rho[0, i + 1] = min(
                    1.0, max(-1.0, rho_hat / (math.sqrt(v_s) * f0))
                )
        else:
            # C++ parity: denglizhoubasketengine.cpp:152-158.
            _s[0] = abs(p[0][0] * p[0][2])
            _dq[0] = p[0][3]
            _v[0] = p[0][4]
            for i in range(n_count + 1):
                n_rho[0, i] = n_rho[i, 0] = float(rho[p[i][1], p[0][1]])

        for i in range(n_count):
            _s[i + 1] = abs(p[m_count + i][0] * p[m_count + i][2])
            _dq[i + 1] = p[m_count + i][3]
            _v[i + 1] = p[m_count + i][4]

            idx = p[m_count + i][1]
            for j in range(n_count):
                n_rho[i + 1, j + 1] = float(rho[idx, p[m_count + j][1]])

        # A weight of exactly 0 gives ``_s[i] == 0`` and C++'s ``Log(_s)``
        # silently yields -inf there (``exp(mu)`` then vanishes, which is the
        # right answer). numpy would raise under the suite's
        # ``filterwarnings = ["error"]``, so the divide is muted explicitly
        # rather than guarded away.
        with np.errstate(divide="ignore"):
            log_s = np.log(_s)

        call_value = self._calculate_vanilla_call(log_s, dr0, _dq, _v, n_rho, strike)

        if payoff.option_type() == OptionType.Call:
            results.value = max(0.0, call_value)
        else:
            fwd = (
                float(_s[0]) * float(_dq[0])
                - dr0 * strike
                - float(np.dot(_s[1:], _dq[1:]))
            )
            results.value = max(0.0, call_value - fwd)

    # --- the closed form ----------------------------------------------------

    @staticmethod
    def _i_term(u: float, t_f2: float, d_mat: Matrix, df_mat: Matrix, i: int) -> float:
        """``I(u, trF2, D, DF, i)``.

        # C++ parity: ``DengLiZhouBasketEngine::I``
        # (denglizhoubasketengine.cpp:182-205). The corrected formula (37).
        """
        d_row = d_mat[i, :]
        df_row = df_mat[i, :]

        psi = 1.0 / (1.0 + float(np.dot(d_row, d_row)))
        sqrt_psi = math.sqrt(psi)

        n_u_sqrt_psi = NormalDistribution()(u * sqrt_psi)
        j_0 = CumulativeNormalDistribution()(u * sqrt_psi)

        v_f_v = float(np.dot(df_row, d_row))
        j_1 = psi * sqrt_psi * (psi * u * u - 1.0) * v_f_v * n_u_sqrt_psi

        v_ff_v = float(np.dot(df_row, df_row))
        j_2 = (
            u
            * psi
            * sqrt_psi
            * n_u_sqrt_psi
            * (
                2 * t_f2
                + v_f_v
                * v_f_v
                * (
                    (psi * u) ** 4
                    - 10.0 * psi * psi * psi * u * u
                    + 15 * psi * psi
                )
                + v_ff_v * (4 * psi * psi * u * u - 12 * psi)
            )
        )

        return j_0 + j_1 - 0.5 * j_2

    @staticmethod
    def _calculate_vanilla_call(
        x: Array, dr: float, dq: Array, v: Array, rho: Matrix, k: float
    ) -> float:
        """# C++ parity: ``calculate_vanilla_call`` (…cpp:207-291)."""
        mu: Array = x + np.log(dq / dr) - 0.5 * v
        nu: Array = np.sqrt(v)

        r_sum = 0.0
        for i in range(1, int(mu.shape[0])):
            r_sum += math.exp(float(mu[i]))

        n = int(x.shape[0]) - 1

        sig11: Matrix = np.array(rho[1:, 1:], dtype=np.float64)
        sig10: Array = np.array(rho[0, 1:], dtype=np.float64)

        sq_sig11 = _pseudo_sqrt_principal(sig11)
        sig11_inv10 = cholesky_solve_for(cholesky_decomposition(sig11), sig10)

        sig_xy = 1.0 - float(np.dot(sig10, sig11_inv10))
        qassert.require(sig_xy > 0.0, "approximation loses validity")
        sq_sig_xy = math.sqrt(sig_xy)

        a = -0.5 / sq_sig_xy
        e_mat: Matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                diag = (
                    float(nu[j]) ** 2 * math.exp(float(mu[j])) / (float(nu[0]) * (r_sum + k))
                    if i == j
                    else 0.0
                )
                val = a * (
                    diag
                    - float(nu[i])
                    * float(nu[j])
                    * math.exp(float(mu[i]) + float(mu[j]))
                    / (float(nu[0]) * (r_sum + k) ** 2)
                )
                e_mat[i - 1, j - 1] = val
                e_mat[j - 1, i - 1] = val

        f_mat: Matrix = sq_sig11 @ e_mat @ sq_sig11

        tr_f = 0.0
        tr_f2 = 0.0
        for i in range(n):
            tr_f += float(f_mat[i, i])
            acc = 0.0
            for j in range(i + 1, n):
                acc += float(f_mat[i, j]) ** 2
            tr_f2 += float(f_mat[i, i]) ** 2 + 2.0 * acc

        c = -(math.log(r_sum + k) - float(mu[0])) / (float(nu[0]) * sq_sig_xy)

        d_vec: Array = (
            sig11_inv10 - np.exp(mu[1:]) * nu[1:] / (float(nu[0]) * (r_sum + k))
        ) / sq_sig_xy

        e_sig10: Array = e_mat @ sig10
        e_sig11: Matrix = e_mat @ sig11
        sig11d: Array = sig11 @ d_vec

        c_vec: Array = np.zeros(n + 2, dtype=np.float64)
        c_vec[0] = (
            c
            + tr_f
            + float(nu[0]) * sq_sig_xy
            + float(nu[0]) * float(np.dot(sig10, d_vec))
            + float(nu[0]) ** 2 * float(np.dot(sig10, e_sig10))
        )
        c_vec[n + 1] = c + tr_f

        for kk in range(1, n + 1):
            c_vec[kk] = (
                c
                + tr_f
                + float(nu[kk]) * float(sig11d[kk - 1])
                + float(nu[kk]) ** 2
                * float(np.dot(sig11[kk - 1, :], e_sig11[:, kk - 1]))
            )

        d_list: list[Array] = [np.zeros(n, dtype=np.float64) for _ in range(n + 2)]
        d_list[0] = sq_sig11 @ (d_vec + 2 * float(nu[0]) * e_sig10)
        d_list[n + 1] = sq_sig11 @ d_vec
        for kk in range(1, n + 1):
            d_list[kk] = sq_sig11 @ (d_vec + 2 * float(nu[kk]) * e_sig11[:, kk - 1])

        dm: Matrix = np.zeros((n + 2, n), dtype=np.float64)
        for kk in range(n + 2):
            dm[kk, :] = d_list[kk]

        df_mat: Matrix = dm @ f_mat

        npv = dr * math.exp(
            float(mu[0]) + 0.5 * float(nu[0]) ** 2
        ) * DengLiZhouBasketEngine._i_term(
            float(c_vec[0]), tr_f2, dm, df_mat, 0
        ) - k * dr * DengLiZhouBasketEngine._i_term(
            float(c_vec[n + 1]), tr_f2, dm, df_mat, n + 1
        )

        for kk in range(1, n + 1):
            npv -= (
                dr
                * math.exp(float(mu[kk]) + 0.5 * float(nu[kk]) ** 2)
                * DengLiZhouBasketEngine._i_term(
                    float(c_vec[kk]), tr_f2, dm, df_mat, kk
                )
            )

        return npv

    def update(self) -> None:
        self.notify_observers()


__all__ = ["DengLiZhouBasketEngine"]
