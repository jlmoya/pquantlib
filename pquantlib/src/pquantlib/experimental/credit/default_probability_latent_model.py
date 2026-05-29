"""DefaultProbabilityLatentModel — joint default events via a copula.

# C++ parity: ql/experimental/credit/defaultprobabilitylatentmodel.hpp
# @ v1.42.1 (099987f0).

The C++ class is a template parameterised on a copula policy with a full
multi-factor (factor_weights matrix) latent-model machinery; it depends
heavily on ``Basket`` for the per-name unconditional default-prob lookup.
The Python port follows the simpler one-factor-copula formulation that
matches all of W3-B's downstream uses:

    p_hat_i(m) = F_Z((F_Y^-1(p_i) - sqrt(rho_i) m) / sqrt(1 - rho_i))

The model takes a ``OneFactorCopula`` at construction (which fixes ``F_Y``,
``F_Z``, and the single common-factor loading ``rho``) and operates on
pool of size ``n``. ``conditional_default_probability_vec(probs, m)`` and
``prob_at_least_n_events(n, probs)`` are the primary entry points.

# C++ parity divergence: the C++ template ``DefaultLatentModel<CP>``
# requires a per-name factor-weight matrix; the Python port assumes a
# uniform per-name loading on the single common factor (the standard
# one-factor Gaussian / Student copula setup). Multi-factor extensions
# are documented as a permanent carve-out — the C++ template was largely
# used for the basket-coupled bookkeeping the Python port replaces with
# direct argument lists.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula


class DefaultProbabilityLatentModel:
    """Joint-default model built on a one-factor copula.

    The model takes:

      - ``copula`` — the OneFactorCopula instance (Gaussian or Student).
      - ``pool_size`` — number of names in the basket.

    All methods take a per-name unconditional-default-probability vector
    of length ``pool_size`` and return either scalar conditional
    probabilities at a factor draw ``m``, or the integrated
    unconditional joint probability of N or more defaults.
    """

    __slots__ = ("_copula", "_n")

    def __init__(self, copula: OneFactorCopula, pool_size: int) -> None:
        qassert.require(pool_size > 0, f"pool_size must be > 0, got {pool_size}")
        self._copula = copula
        self._n = pool_size

    def copula(self) -> OneFactorCopula:
        return self._copula

    def pool_size(self) -> int:
        return self._n

    # ---- conditional default probability vectors ----

    def conditional_default_probability(
        self, prob: float, m: float
    ) -> float:
        """Per-name conditional default probability at factor ``m``.

        # C++ parity: defaultprobabilitylatentmodel.hpp:114 +
        # conditionalDefaultProbabilityInvP at line 151.
        """
        return self._copula.conditional_probability(prob, m)

    def conditional_default_probability_vec(
        self, probs: Sequence[float], m: float
    ) -> list[float]:
        """Vector of per-name conditional default probabilities at factor ``m``."""
        qassert.require(
            len(probs) == self._n,
            f"probs size {len(probs)} != pool_size {self._n}",
        )
        return self._copula.conditional_probability_vec(list(probs), m)

    # ---- prob at least n events ----

    def conditional_prob_at_least_n_events(
        self, n: int, probs: Sequence[float], m: float
    ) -> float:
        """Conditional probability of at least ``n`` defaults given M=m.

        Iterates all 2^size subsets of names — O(2^size). Use only for
        small pools.

        # C++ parity: defaultprobabilitylatentmodel.hpp:202 — same
        # combinatorial walk over bitmasks.
        """
        qassert.require(
            len(probs) == self._n,
            f"probs size {len(probs)} != pool_size {self._n}",
        )
        cond_probs = self.conditional_default_probability_vec(probs, m)
        limit = 1 << self._n
        mask = (1 << n) - 1
        prob = 0.0
        for k in range(mask, limit):
            bits = bin(k).count("1")
            if bits >= n:
                p_config = 1.0
                for j in range(self._n):
                    if (k >> j) & 1:
                        p_config *= cond_probs[j]
                    else:
                        p_config *= 1.0 - cond_probs[j]
                prob += p_config
        return prob

    def prob_at_least_n_events(self, n: int, probs: Sequence[float]) -> float:
        """Unconditional probability of at least ``n`` defaults — integrates
        the conditional probability over the M factor.

        # C++ parity: defaultprobabilitylatentmodel.hpp:237 — integration
        # over rho_M(m).
        """
        qassert.require(
            len(probs) == self._n,
            f"probs size {len(probs)} != pool_size {self._n}",
        )
        self._copula.calculate()
        avg = 0.0
        for k in range(self._copula.steps()):
            mk = self._copula.m(k)
            avg += self.conditional_prob_at_least_n_events(
                n, probs, mk
            ) * self._copula.density_dm(k)
        return avg

    # ---- prob of default of name ----

    def prob_of_default(self, i_name: int, prob: float) -> float:
        """Unconditional probability of default of name i — sanity check.

        Mirrors C++ ``probOfDefault`` (defaultprobabilitylatentmodel.hpp:211)
        but takes the per-name unconditional default probability directly.
        Reproduces ``prob`` up to integration noise.
        """
        qassert.require(0 <= i_name < self._n, f"i_name {i_name} out of range")
        if prob < 1e-10:
            return 0.0
        self._copula.calculate()
        avg = 0.0
        for k in range(self._copula.steps()):
            mk = self._copula.m(k)
            avg += self.conditional_default_probability(
                prob, mk
            ) * self._copula.density_dm(k)
        return avg

    # ---- default correlation ----

    def default_correlation(
        self, prob_i: float, prob_j: float
    ) -> float:
        """Pearson default-event correlation between names i and j.

        # C++ parity: defaultprobabilitylatentmodel.hpp:249.
        Assumes a uniform single-factor loading; names i != j only.
        """
        if prob_i < 1e-10 or prob_j < 1e-10:
            return 0.0
        self._copula.calculate()
        # Compute E[1_i 1_j] = integral over m of p_hat_i(m) p_hat_j(m).
        e1i1j = 0.0
        for k in range(self._copula.steps()):
            mk = self._copula.m(k)
            pi_m = self._copula.conditional_probability(prob_i, mk)
            pj_m = self._copula.conditional_probability(prob_j, mk)
            e1i1j += pi_m * pj_m * self._copula.density_dm(k)
        pipj = prob_i * prob_j
        denom = (pipj * (1.0 - prob_i) * (1.0 - prob_j)) ** 0.5
        return (e1i1j - pipj) / denom if denom > 0 else 0.0
