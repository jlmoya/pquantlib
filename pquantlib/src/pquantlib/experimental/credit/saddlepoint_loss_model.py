"""SaddlepointLossModel — Lugannani-Rice saddlepoint expansion.

# C++ parity: ql/experimental/credit/saddlepointlossmodel.hpp @ v1.42.1.

Approximates the portfolio loss distribution via the Lugannani-Rice
saddlepoint expansion of the cumulant-generating function:

    K(t) = sum_i log(1 - p_i_hat + p_i_hat exp(t * l_i))

where p_i_hat is the per-name conditional default probability and l_i
is the per-name LGD. The expansion is conditional on the market factor
M and integrated over M to recover the unconditional distribution.

The Python port keeps the high-level Martin-Thompson-Browne (2001)
formulation. For full Antonov-Mechkov-Misirpashaev high-order
corrections see the C++ source (they are not required for W3-B's
LHP cross-validation).

# C++ parity divergence: the C++ template lives behind a
# Basket+ConstantLossLatentModel coupling; the Python port takes
# (probs, notionals) directly. The saddlepoint equation is solved via
# scipy.optimize.brentq instead of QL's hand-rolled Newton-Brent
# hybrid; that change does not affect the answer because the function
# is monotone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

from pquantlib import qassert
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentModel,
)
from pquantlib.experimental.credit.default_loss_model import DefaultLossModelBase
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)


class SaddlepointLossModel(DefaultLossModelBase):
    """Lugannani-Rice saddlepoint loss-distribution model.

    Constructor takes a ``ConstantLossLatentModel`` (carries the copula
    and recovery vector).
    """

    __slots__ = ("_latent_model", "_phi")

    def __init__(self, latent_model: ConstantLossLatentModel) -> None:
        super().__init__()
        self._latent_model = latent_model
        self._phi = CumulativeNormalDistribution()

    def latent_model(self) -> ConstantLossLatentModel:
        return self._latent_model

    @staticmethod
    def _cgf(t: float, lgds: Sequence[float], probs: Sequence[float]) -> float:
        """K(t) = sum_i log(1 - p_i + p_i exp(t l_i))."""
        acc = 0.0
        for i, lgd in enumerate(lgds):
            p = probs[i]
            # exp(t * lgd) can overflow; cap it.
            arg = t * lgd
            if arg > 50:
                # log(p exp(arg)) = log(p) + arg
                acc += float(np.log(p) + arg) if p > 0 else 0.0
            else:
                acc += float(np.log(1.0 - p + p * np.exp(arg)))
        return acc

    @staticmethod
    def _cgf_prime(
        t: float, lgds: Sequence[float], probs: Sequence[float]
    ) -> float:
        """K'(t) = sum_i (p_i l_i exp(t l_i)) / (1 - p_i + p_i exp(t l_i))."""
        acc = 0.0
        for i, lgd in enumerate(lgds):
            p = probs[i]
            arg = t * lgd
            if arg > 50:
                acc += lgd
            else:
                e = float(np.exp(arg))
                num = p * lgd * e
                denom = 1.0 - p + p * e
                acc += num / denom
        return acc

    @staticmethod
    def _cgf_double_prime(
        t: float, lgds: Sequence[float], probs: Sequence[float]
    ) -> float:
        """K''(t) = sum_i p_i (1 - p_i) l_i^2 exp(t l_i) / (1 - p_i + p_i exp(t l_i))^2."""
        acc = 0.0
        for i, lgd in enumerate(lgds):
            p = probs[i]
            arg = t * lgd
            if arg > 50:
                # Term vanishes since the denominator dominates.
                continue
            e = float(np.exp(arg))
            denom = 1.0 - p + p * e
            num = p * (1.0 - p) * lgd**2 * e
            acc += num / (denom**2)
        return acc

    def _find_saddle(
        self,
        target_loss: float,
        lgds: Sequence[float],
        probs: Sequence[float],
    ) -> float:
        """Find t such that K'(t) = target_loss."""

        def f(t: float) -> float:
            return self._cgf_prime(t, lgds, probs) - target_loss

        # The CGF first derivative is monotone in t. Search over a wide
        # bracket; the answer is finite for target_loss strictly
        # between 0 and sum(lgds).
        # Try a few bracket widths until a sign change is found.
        for hi in (1.0, 5.0, 20.0, 50.0):
            for lo in (-50.0, -20.0, -5.0, -1.0):
                try:
                    f_lo = f(lo)
                    f_hi = f(hi)
                    if f_lo * f_hi < 0:
                        root = brentq(f, lo, hi, xtol=1e-10, maxiter=500)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]
                        return float(root)  # pyright: ignore[reportArgumentType]
                except (ValueError, OverflowError):
                    continue
        # Last resort: return 0 (loss at the mean).
        return 0.0

    def _prob_over_loss_conditional(
        self,
        loss_level: float,
        lgds: Sequence[float],
        probs: Sequence[float],
    ) -> float:
        """Lugannani-Rice approximation of P(L >= loss_level | M=m)."""
        total = sum(lgds)
        if loss_level <= 0:
            return 1.0
        if loss_level >= total:
            return 0.0
        # Conditional mean
        mean = sum(probs[i] * lgds[i] for i in range(len(probs)))
        if abs(loss_level - mean) < 1e-10:
            # Mean point: simple normal approximation
            return 0.5
        # Find saddle point
        t_hat = self._find_saddle(loss_level, lgds, probs)
        if abs(t_hat) < 1e-10:
            return 0.5
        # CGF + second derivative at saddle
        k = self._cgf(t_hat, lgds, probs)
        k2 = self._cgf_double_prime(t_hat, lgds, probs)
        if k2 <= 0:
            return 0.5
        # Lugannani-Rice formula
        w = float(np.sign(t_hat) * np.sqrt(2.0 * (t_hat * loss_level - k)))
        u = t_hat * float(np.sqrt(k2))
        # Avoid division by zero
        if abs(w) < 1e-10 or abs(u) < 1e-10:
            return 0.5
        # P(L >= x) approximation
        phi_w = float(self._phi(w))
        # Standard normal PDF at w
        pdf_w = float(np.exp(-w**2 / 2.0) / np.sqrt(2.0 * np.pi))
        return 1.0 - phi_w + pdf_w * (1.0 / u - 1.0 / w)

    def prob_over_loss_unconditional(
        self,
        loss_level: float,
        probs: Sequence[float],
        notionals: Sequence[float],
    ) -> float:
        """Unconditional P(L >= loss_level) — integrates over M.

        # C++ parity: saddlepointlossmodel.hpp probOverLossPortfNoPrioM.
        """
        rrs = self._latent_model.recoveries()
        lgds = [(1.0 - rrs[i]) * notionals[i] for i in range(len(notionals))]
        cop = self._latent_model.copula()
        cop.calculate()
        acc = 0.0
        for k in range(cop.steps()):
            mk = cop.m(k)
            cond_probs = self._latent_model.conditional_default_probability_vec(
                probs, mk
            )
            p = self._prob_over_loss_conditional(loss_level, lgds, cond_probs)
            acc += p * cop.density_dm(k)
        return min(max(acc, 0.0), 1.0)

    def expected_tranche_loss(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
    ) -> float:
        """Expected tranche loss via integration of the survival function.

        Uses ETL = integral over loss of P(L >= max(attach, l)) - P(L >= detach).

        # C++ parity: saddlepointlossmodel.hpp expectedTrancheLoss —
        # we use a simpler quadrature-of-survival-function formulation.
        """
        n = self._latent_model.pool_size()
        return self.expected_tranche_loss_heterogeneous(
            remaining_notional=remaining_notional,
            probs=[prob] * n,
            notionals=[remaining_notional / n] * n,
            attach=attach,
            detach=detach,
        )

    def expected_tranche_loss_heterogeneous(
        self,
        remaining_notional: float,
        probs: Sequence[float],
        notionals: Sequence[float],
        attach: float,
        detach: float,
    ) -> float:
        """ETL via integration of survival function P(L > l) over (attach, detach).

        ETL = integral_attach^detach P(L > l) dl.
        """
        _ = remaining_notional
        qassert.require(
            attach < detach, f"attach {attach} >= detach {detach}"
        )
        # Use 64-point Simpson rule over (attach, detach). n_quad is
        # always even so no odd-n branch needed.
        n_quad = 64
        x = np.linspace(attach, detach, n_quad + 1)
        y = np.array(
            [
                self.prob_over_loss_unconditional(float(xi), probs, notionals)
                for xi in x
            ]
        )
        # Simpson's rule
        h = (detach - attach) / n_quad
        integral = (
            h / 3.0 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-2:2].sum() + y[-1])
        )
        return float(integral)
