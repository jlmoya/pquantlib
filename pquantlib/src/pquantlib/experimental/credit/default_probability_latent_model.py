"""Default-event latent models.

This module hosts two classes:

* :class:`DefaultLatentModel` — the faithful port of the C++ class template
  ``DefaultLatentModel<copulaPolicy>``
  (# C++ parity: ql/experimental/credit/defaultprobabilitylatentmodel.hpp:43-330
  @ v1.43). Multi-factor, copula-policy parametric, ``Basket``-coupled, and
  cross-validated against the C++ in
  ``tests/experimental/credit/test_credit_latent_models.py``.
* :class:`DefaultProbabilityLatentModel` — a pre-existing one-factor
  convenience adaptation that the W3-B loss models
  (Binomial / Recursive / Saddlepoint) are written against. It is *not* a
  port of a C++ class; see its own docstring.

# C++ parity: ql/experimental/credit/defaultprobabilitylatentmodel.hpp (v1.43).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.experimental.math.latent_model import (
    CopulaPolicy,
    LatentModel,
    LMIntegration,
)
from pquantlib.experimental.math.latent_model import (
    LatentModelIntegrationType as CanonicalIntegrationType,
)

if TYPE_CHECKING:
    from pquantlib.experimental.credit.basket import Basket
    from pquantlib.time.date import Date


class LatentModelIntegrationType(IntEnum):
    """Integration algorithm a latent model integrates its density with.

    # C++ parity: ``namespace LatentModelIntegrationType`` in
    # ql/experimental/math/latentmodel.hpp:98-107 (v1.43).

    # Port note: this is the credit package's spelling of the enum. The
    # canonical port lives next to ``LatentModel`` in
    # ``pquantlib.experimental.math.latent_model``; the values are the same,
    # and :func:`create_lm_integration` forwards to that module's factory.
    """

    GaussianQuadrature = 0
    Trapezoid = 1


def create_lm_integration(
    dimension: int,
    integral_type: LatentModelIntegrationType = (
        LatentModelIntegrationType.GaussianQuadrature
    ),
) -> LMIntegration:
    """Build the integration facility for a latent model of ``dimension``.

    # C++ parity: ``LatentModel<CP>::IntegrationFactory::createLMIntegration``
    # (latentmodel.hpp:446-493). Thin forwarder onto the port of that factory;
    # the magic numbers (quadrature order 25; per-axis
    # ``TrapezoidIntegral<Default>(1e-4, 20)`` over ``[-35, 35]``) live there.
    """
    return LatentModel.IntegrationFactory.create_lm_integration(
        dimension, CanonicalIntegrationType(int(integral_type))
    )


class DefaultLatentModel(LatentModel):
    """Joint default-event model built on a generic latent model.

    # C++ parity: ``template<class copulaPolicy> class DefaultLatentModel``
    # (defaultprobabilitylatentmodel.hpp:43-243, definitions 248-330).

    Models solely the default events in a portfolio — no severities, no
    exposures. The correspondence between the modelled latent variables and
    the basket names is positional: variable ``i`` is basket name ``i``.

    :param factor_weights: ``factor_weights[i_variable][i_factor]``.
    :param copula: the copula policy instance, built over the same
        ``factor_weights``. In C++ the model constructs the policy itself from
        ``factorWeights`` plus the policy's ``initTraits``; Python copula
        policies are ordinary classes with ordinary constructors, so the
        instance is passed in.
    :param integral_type: which integration facility to build.
    """

    __slots__ = ("_basket",)

    def __init__(
        self,
        factor_weights: Sequence[Sequence[float]],
        copula: CopulaPolicy,
        integral_type: LatentModelIntegrationType = (
            LatentModelIntegrationType.GaussianQuadrature
        ),
    ) -> None:
        super().__init__(
            factor_weights,
            copula,
            create_lm_integration(len(factor_weights[0]), integral_type),
        )
        self._basket: Basket | None = None

    # ---- basket wiring ----

    def reset_basket(self, basket: Basket) -> None:
        """Attach (or swap) the portfolio basket.

        # C++ parity: defaultprobabilitylatentmodel.hpp:97-102. The model
        # caches nothing, so the basket may be replaced freely.
        """
        qassert.require(
            basket.size() == len(self.factor_weights()),
            "Incompatible new basket and model sizes.",
        )
        self._basket = basket

    def basket(self) -> Basket:
        """The attached basket, or raise if none was set."""
        qassert.require(self._basket is not None, "No portfolio basket set.")
        assert self._basket is not None
        return self._basket

    def integration(self) -> LMIntegration:
        """The integration facility.

        # C++ parity: defaultprobabilitylatentmodel.hpp:205.
        """
        integration = self._integration
        qassert.require(integration is not None, "No integration facility.")
        assert integration is not None
        return integration

    # ---- conditional default probabilities ----

    def conditional_default_probability(
        self, prob_or_date: float | Date, i_name: int, mkt_factors: Sequence[float]
    ) -> float:
        """Probability of default of ``i_name`` given the market factors.

        # C++ parity: the two overloads at
        # defaultprobabilitylatentmodel.hpp:114-129 (unconditional probability)
        # and :179-188 (date; looks the probability up in the basket's pool).
        # Python dispatches on the argument type.
        """
        if not isinstance(prob_or_date, (int, float)):
            basket = self.basket()
            pool = basket.pool()
            p_def_uncond = (
                pool.get(pool.names()[i_name])
                .default_probability(basket.default_keys()[i_name])
                .default_probability(prob_or_date)
            )
            return self.conditional_default_probability(
                p_def_uncond, i_name, mkt_factors
            )
        # C++ parity: avoids a redundant (possibly infinite) inversion.
        if prob_or_date < 1.0e-10:
            return 0.0
        return self.conditional_default_probability_inv_p(
            self.inverse_cumulative_y(prob_or_date, i_name), i_name, mkt_factors
        )

    def conditional_default_probability_inv_p(
        self, inv_cum_y_prob: float, i_name: int, m: Sequence[float]
    ) -> float:
        """Same as above, taking the *already inverted* probability.

        # C++ parity: defaultprobabilitylatentmodel.hpp:151-165. Saves the
        # cumulative inversion when integrating over the market factors.
        """
        row = self._factor_weights[i_name]
        sum_ms = 0.0
        for k in range(len(row)):
            sum_ms += row[k] * m[k]
        return self.cumulative_z(
            (inv_cum_y_prob - sum_ms) / self._idiosync_fctrs[i_name]
        )

    def cond_prob_product(
        self,
        inv_cum_y_prob1: float,
        inv_cum_y_prob2: float,
        i_name1: int,
        i_name2: int,
        mkt_factors: Sequence[float],
    ) -> float:
        """Product of two conditional default probabilities.

        # C++ parity: defaultprobabilitylatentmodel.hpp:191-199 — the
        # intermediate step of the default-correlation integral.
        """
        return self.conditional_default_probability_inv_p(
            inv_cum_y_prob1, i_name1, mkt_factors
        ) * self.conditional_default_probability_inv_p(
            inv_cum_y_prob2, i_name2, mkt_factors
        )

    def conditional_prob_at_least_n_events(
        self, n: int, date: Date, mkt_factors: Sequence[float]
    ) -> float:
        """Conditional probability of ``n`` or more defaults by ``date``.

        # C++ parity: defaultprobabilitylatentmodel.hpp:278-330. The C++
        # walks every bit pattern from the lowest one with ``n`` bits set up
        # to ``2^poolSize`` and sums the ones with at least ``n`` bits; the
        # port keeps that traversal (and its cost) verbatim.
        """
        basket = self.basket()
        pool_size = basket.size()
        pool = basket.pool()

        limit = int(math.pow(2.0, pool_size))

        p_def_cond: list[float] = []
        for i in range(pool_size):
            p_def_cond.append(
                self.conditional_default_probability(
                    pool.get(pool.names()[i])
                    .default_probability(basket.default_keys()[i])
                    .default_probability(date),
                    i,
                    mkt_factors,
                )
            )

        prob_n_events_or_more = 0.0
        mask = (1 << n) - 1
        for bits in range(mask, limit):
            if bin(bits).count("1") >= n:
                p_config = 1.0
                for i in range(pool_size):
                    p_config *= (
                        p_def_cond[i] if (bits >> i) & 1 else (1.0 - p_def_cond[i])
                    )
                prob_n_events_or_more += p_config
        return prob_n_events_or_more

    # ---- integrated quantities ----

    def prob_of_default(self, i_name: int, d: Date) -> float:
        """Unconditional probability of default of ``i_name`` by ``d``.

        # C++ parity: defaultprobabilitylatentmodel.hpp:211-225. Reproduces
        # the unconditional probability by integration — a self-consistency
        # check on the model, as the C++ comment says.
        """
        basket = self.basket()
        pool = basket.pool()
        p_uncond = (
            pool.get(pool.names()[i_name])
            .default_probability(basket.default_keys()[i_name])
            .default_probability(d)
        )
        if p_uncond < 1.0e-10:
            return 0.0
        inv_p = self.inverse_cumulative_y(p_uncond, i_name)
        return self.integrated_expected_value(
            lambda v1: self.conditional_default_probability_inv_p(inv_p, i_name, v1)
        )

    def default_correlation(self, d: Date, i_name_i: int, i_name_j: int) -> float:
        """Pearson correlation of the two default indicators at ``d``.

        # C++ parity: defaultprobabilitylatentmodel.hpp:248-275.
        """
        basket = self.basket()
        pool = basket.pool()
        p_i = (
            pool.get(pool.names()[i_name_i])
            .default_probability(basket.default_keys()[i_name_i])
            .default_probability(d)
        )
        p_j = (
            pool.get(pool.names()[i_name_j])
            .default_probability(basket.default_keys()[i_name_j])
            .default_probability(d)
        )
        pipj = p_i * p_j
        inv_pi = self.inverse_cumulative_y(p_i, i_name_i)
        inv_pj = self.inverse_cumulative_y(p_j, i_name_j)
        if i_name_i != i_name_j:
            e1i1j = self.integrated_expected_value(
                lambda v1: self.cond_prob_product(
                    inv_pi, inv_pj, i_name_i, i_name_j, v1
                )
            )
        else:
            e1i1j = p_i
        return (e1i1j - pipj) / math.sqrt(pipj * (1.0 - p_i) * (1.0 - p_j))

    def prob_at_least_n_events(self, n: int, date: Date) -> float:
        """Unconditional probability of ``n`` or more defaults by ``date``.

        # C++ parity: defaultprobabilitylatentmodel.hpp:237-242.
        """
        return self.integrated_expected_value(
            lambda v1: self.conditional_prob_at_least_n_events(n, date, v1)
        )


class DefaultProbabilityLatentModel:
    """Joint-default model built on a one-factor copula.

    .. warning::

       This class is **not** a port of a C++ class. It is a one-factor
       convenience adaptation, keyed on
       :class:`~pquantlib.experimental.credit.one_factor_copula.OneFactorCopula`
       and taking per-name unconditional default probabilities as explicit
       arguments instead of reading them out of a ``Basket``. The W3-B loss
       models (Binomial / Recursive / Saddlepoint) are written against it.

       The faithful port of C++ ``DefaultLatentModel<copulaPolicy>`` is
       :class:`DefaultLatentModel`, above. Prefer it for anything that has to
       agree with QuantLib: it is multi-factor, copula-policy parametric, and
       integrates with the same quadrature the C++ uses, whereas this class
       integrates on the ``OneFactorCopula`` 50-step Euler grid and therefore
       carries ~1e-6 integration error.

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
