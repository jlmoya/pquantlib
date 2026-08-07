"""Spot-recovery latent models.

This module hosts two classes:

* :class:`SpotRecoveryLatentModel` — the faithful port of the C++ class
  template ``SpotRecoveryLatentModel<copulaPolicy>``.
  # C++ parity: ql/experimental/credit/spotlosslatentmodel.hpp:41-374 (v1.43).
* :class:`SpotLossLatentModel` — a pre-existing single-factor adaptation.
  Not a port of a C++ class; see its own docstring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.credit.default_probability_latent_model import (
    LatentModelIntegrationType,
    create_lm_integration,
)
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.experimental.math.latent_model import CopulaPolicy, LatentModel, LMIntegration
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)

if TYPE_CHECKING:
    from pquantlib.experimental.credit.basket import Basket
    from pquantlib.time.date import Date


class SpotRecoveryLatentModel(LatentModel):
    """Random spot-recovery-rate latent variable portfolio model.

    # C++ parity: ``template <class copulaPolicy> class SpotRecoveryLatentModel``
    # (spotlosslatentmodel.hpp:41-127, definitions 136-374).

    Bennani-Maetz (2009) / Li (2009) spot recovery, generalised to a
    multifactor set-up and a generic copula. The latent model carries ``2N``
    variables: the first ``N`` drive defaults, the last ``N`` drive recovery
    rates, and ``crossIdiosyncFctrs_[i] = sum_k a_ik^2 a_(N+i)k^2`` couples
    them.

    :param factor_weights: ``2N`` rows — default betas first, recovery betas
        second.
    :param recoveries: ``N`` unconditional mean recovery rates.
    :param model_a: the recovery-noise multiplier of the reference model.
    :param copula: the copula policy built over the same ``factor_weights``.
    :param integral_type: which integration facility to build.
    """

    __slots__ = ("_basket", "_cross_idiosync_fctrs", "_model_a", "_num_names",
                 "_recoveries")

    def __init__(
        self,
        factor_weights: Sequence[Sequence[float]],
        recoveries: Sequence[float],
        model_a: float,
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
        qassert.require(
            len(factor_weights) % 2 == 0,
            "Number of RR variables must be equal to number of default variables",
        )
        num_names = len(factor_weights) // 2
        qassert.require(
            len(recoveries) == num_names,
            "Number of recoveries does not match number of defaultable entities.",
        )
        self._recoveries: list[float] = list(recoveries)
        self._model_a: float = model_a
        self._num_names: int = num_names
        self._basket: Basket | None = None
        # C++ parity: spotlosslatentmodel.hpp:357-372 — sum_k a_ik^2 a_(N+i)k^2.
        cross: list[float] = []
        for i_name in range(num_names):
            cumul = 0.0
            row_d = factor_weights[i_name]
            row_r = factor_weights[i_name + num_names]
            for i_b in range(len(row_d)):
                cumul += row_d[i_b] * row_d[i_b] * row_r[i_b] * row_r[i_b]
            cross.append(cumul)
        self._cross_idiosync_fctrs: list[float] = cross

    # ---- inspectors ----

    def recoveries(self) -> list[float]:
        """The per-name unconditional mean recovery rates."""
        return self._recoveries

    def model_a(self) -> float:
        """The model's ``A`` parameter."""
        return self._model_a

    def num_names(self) -> int:
        """``N`` — half the number of latent variables."""
        return self._num_names

    def cross_idiosync_fctrs(self) -> list[float]:
        """``sum_k a_ik^2 a_(N+i)k^2`` per name.

        # C++ parity: ``crossIdiosyncFctrs_`` (spotlosslatentmodel.hpp:54).
        """
        return self._cross_idiosync_fctrs

    def integration(self) -> LMIntegration:
        """# C++ parity: spotlosslatentmodel.hpp:60."""
        integration = self._integration
        qassert.require(integration is not None, "No integration facility.")
        assert integration is not None
        return integration

    def basket(self) -> Basket:
        """The attached basket, or raise if none was set."""
        qassert.require(self._basket is not None, "No portfolio basket set.")
        assert self._basket is not None
        return self._basket

    def reset_basket(self, basket: Basket) -> None:
        """# C++ parity: spotlosslatentmodel.hpp:136-143."""
        qassert.require(
            basket.size() == self._num_names,
            "Incompatible new basket and model sizes.",
        )
        self._basket = basket

    # ---- conditional default probability ----

    def conditional_default_probability(
        self, prob_or_date: float | Date, i_name: int, mkt_factors: Sequence[float]
    ) -> float:
        """# C++ parity: spotlosslatentmodel.hpp:145-177 (both overloads)."""
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
        if prob_or_date < 1.0e-10:
            return 0.0
        return self.conditional_default_probability_inv_p(
            self.inverse_cumulative_y(prob_or_date, i_name), i_name, mkt_factors
        )

    def conditional_default_probability_inv_p(
        self, inv_cum_y_prob: float, i_name: int, m: Sequence[float]
    ) -> float:
        """# C++ parity: spotlosslatentmodel.hpp:179-197."""
        row = self._factor_weights[i_name]
        sum_ms = 0.0
        for k in range(len(row)):
            sum_ms += row[k] * m[k]
        return self.cumulative_z(
            (inv_cum_y_prob - sum_ms) / self._idiosync_fctrs[i_name]
        )

    # ---- expected conditional recovery (eq. 44 of Li 2009) ----

    def exp_cond_recovery(
        self, d: Date, i_name: int, mkt_factors: Sequence[float]
    ) -> float:
        """# C++ parity: ``expCondRecovery`` (spotlosslatentmodel.hpp:199-216)."""
        basket = self.basket()
        pool = basket.pool()
        p_def_uncond = (
            pool.get(pool.names()[i_name])
            .default_probability(basket.default_keys()[i_name])
            .default_probability(d)
        )
        return self.exp_cond_recovery_p(p_def_uncond, i_name, mkt_factors)

    def exp_cond_recovery_p(
        self, uncond_def_p: float, i_name: int, mkt_factors: Sequence[float]
    ) -> float:
        """# C++ parity: ``expCondRecoveryP`` (spotlosslatentmodel.hpp:218-226)."""
        return self.exp_cond_recovery_inv_p_inv_rr(
            self.inverse_cumulative_y(uncond_def_p, i_name),
            self.inverse_cumulative_y(
                self._recoveries[i_name], i_name + self._num_names
            ),
            i_name,
            mkt_factors,
        )

    def exp_cond_recovery_inv_p_inv_rr(
        self,
        inv_uncond_def_p: float,
        inv_uncond_rr: float,
        i_name: int,
        mkt_factors: Sequence[float],
    ) -> float:
        """# C++ parity: ``expCondRecoveryInvPinvRR``
        # (spotlosslatentmodel.hpp:228-253) — eq. 44 of Li (2009).
        """
        fctrs = self.factor_weights()
        row_d = fctrs[i_name]
        row_r = fctrs[i_name + self._num_names]
        sum_ms = 0.0
        for k in range(len(row_d)):
            sum_ms += row_d[k] * mkt_factors[k]
        sum_beta_loss = 0.0
        for k in range(len(row_r)):
            sum_beta_loss += row_r[k] * row_r[k]
        cross = self._cross_idiosync_fctrs[i_name]
        model_a = self._model_a
        return self.cumulative_z(
            (
                sum_ms
                + math.sqrt(1.0 - cross)
                * math.sqrt(1.0 + model_a * model_a)
                * inv_uncond_rr
                - math.sqrt(cross) * inv_uncond_def_p
            )
            / math.sqrt(1.0 - sum_beta_loss + model_a * model_a * (1.0 - cross))
        )

    # ---- realised (simulation) recovery ----

    def conditional_recovery(
        self, latent_var_sample: float, i_name: int, d: Date
    ) -> float:
        """Realised spot recovery for a latent-variable sample that defaulted.

        # C++ parity: spotlosslatentmodel.hpp:255-279 — eq. 42 of Li (2009).
        # There is no check that the sample actually led to a default; that is
        # the caller's business, as the C++ comment says.
        """
        basket = self.basket()
        pool = basket.pool()
        dfts = pool.get(basket.names()[i_name]).default_probability(
            basket.default_keys()[i_name]
        )
        pdef = dfts.default_probability(d, True)
        # before asking for -infinity
        if pdef < 1.0e-10:
            return 0.0
        i_recovery = i_name + self._num_names
        cross = self._cross_idiosync_fctrs[i_name]
        model_a = self._model_a
        return self.cumulative_y(
            (
                latent_var_sample
                - math.sqrt(cross) * self.inverse_cumulative_y(pdef, i_name)
            )
            / (model_a * math.sqrt(1.0 - cross))
            + math.sqrt(1.0 + 1.0 / (model_a * model_a))
            * self.inverse_cumulative_y(self._recoveries[i_name], i_recovery),
            i_recovery,
        )

    def latent_rr_var_value(
        self, all_factors: Sequence[float], i_name: int
    ) -> float:
        """The recovery latent variable of ``i_name`` from a full factor sample.

        # C++ parity: spotlosslatentmodel.hpp:281-288.
        """
        return self.latent_var_value(all_factors, i_name + self._num_names)

    # ---- uninstantiable in C++ v1.43 ----

    _UNINSTANTIABLE = (
        "SpotRecoveryLatentModel::conditionalExpLossRRInv cannot be "
        "instantiated in QuantLib v1.43: its body calls "
        "conditionalRecoveryInvPinvRR (spotlosslatentmodel.hpp:315), which is "
        "declared nowhere in the library. conditionalExpLossRR and "
        "expectedLoss both route through it, so all three are dead. There is "
        "no C++ behaviour to reproduce and the port does not invent one."
    )

    def conditional_exp_loss_rr_inv(
        self,
        inv_p: float,
        inv_rr: float,
        i_name: int,
        mkt_factors: Sequence[float],
    ) -> float:
        """Always fails — see :attr:`_UNINSTANTIABLE`.

        # C++ parity: spotlosslatentmodel.hpp:307-316. The body reads::
        #
        #     return conditionalDefaultProbabilityInvP(invP, iName, mktFactors)
        #         * (1.-this->conditionalRecoveryInvPinvRR(invP, invRR, iName,
        #                                                  mktFactors));
        #
        # `conditionalRecoveryInvPinvRR` does not exist. Because these are
        # templates the body is only instantiated on use, so the library
        # builds — but the first caller fails to compile. The member was
        # evidently renamed to `expCondRecoveryInvPinvRR` and this call site
        # was missed.
        """
        del inv_p, inv_rr, i_name, mkt_factors
        qassert.fail(self._UNINSTANTIABLE)

    def conditional_exp_loss_rr(
        self, d: Date, i_name: int, mkt_factors: Sequence[float]
    ) -> float:
        """Always fails — routes through :meth:`conditional_exp_loss_rr_inv`.

        # C++ parity: spotlosslatentmodel.hpp:290-305.
        """
        del d, i_name, mkt_factors
        qassert.fail(self._UNINSTANTIABLE)

    def expected_loss(self, d: Date, i_name: int) -> float:
        """Always fails — routes through :meth:`conditional_exp_loss_rr_inv`.

        # C++ parity: spotlosslatentmodel.hpp:318-335.
        """
        del d, i_name
        qassert.fail(self._UNINSTANTIABLE)


# ---------------------------------------------------------------------------
# Pre-existing single-factor adaptation (not a C++ class).
# ---------------------------------------------------------------------------

class SpotLossLatentModel:
    """Spot-recovery latent model (single-factor Python adaptation).

    .. warning::

       This class is **not** a port of a C++ class. It is a single-factor
       adaptation with a per-name ``(default_loading, rr_loading)`` pair; the
       ``cross_idiosync_factors`` array reduces to ``rho * rr_loading[i]^2``
       rather than C++'s ``sum_k a_ik^2 a_(N+i)k^2``.

       The faithful port of C++ ``SpotRecoveryLatentModel<copulaPolicy>`` is
       :class:`SpotRecoveryLatentModel`, above.

    Pricing inputs per-name:
      - ``rr_mean[i]`` — the unconditional mean recovery rate.
      - ``rr_loading[i]`` — the per-name factor loading on the *recovery*
        latent variable (square root of correlation).

    Per-default latent variable:
      - ``copula`` — the OneFactorCopula instance carrying ``F_Y`` and ``F_Z``
        (Gaussian or Student).

    The cross-idiosyncratic factor is ``rho_d_i * rho_l_i`` (product of
    the per-name default and RR loadings). The model A parameter is the
    multiplier on the RR latent variable noise.
    """

    __slots__ = ("_copula", "_cum_norm", "_inv_norm", "_model_a", "_n", "_rr_loading", "_rr_mean")

    def __init__(
        self,
        copula: OneFactorCopula,
        rr_mean: Sequence[float],
        rr_loading: Sequence[float],
        model_a: float,
    ) -> None:
        qassert.require(
            len(rr_mean) == len(rr_loading),
            f"rr_mean ({len(rr_mean)}) and rr_loading ({len(rr_loading)}) sizes differ",
        )
        qassert.require(
            len(rr_mean) > 0,
            "rr_mean must be non-empty",
        )
        qassert.require(model_a > 0.0, f"model_a must be > 0, got {model_a}")
        for i, r in enumerate(rr_mean):
            qassert.require(
                0.0 <= r <= 1.0, f"rr_mean[{i}] = {r} not in [0, 1]"
            )
        for i, rl in enumerate(rr_loading):
            qassert.require(
                0.0 <= rl <= 1.0, f"rr_loading[{i}] = {rl} not in [0, 1]"
            )
        self._copula = copula
        self._rr_mean = list(rr_mean)
        self._rr_loading = list(rr_loading)
        self._model_a = model_a
        self._n = len(rr_mean)
        self._inv_norm = InverseCumulativeNormal()
        self._cum_norm = CumulativeNormalDistribution()

    def copula(self) -> OneFactorCopula:
        return self._copula

    def pool_size(self) -> int:
        return self._n

    def rr_mean(self) -> list[float]:
        return list(self._rr_mean)

    def rr_loading(self) -> list[float]:
        return list(self._rr_loading)

    def model_a(self) -> float:
        return self._model_a

    def _cross_idiosync(self, i: int) -> float:
        # # C++ parity: spotlosslatentmodel.hpp constructor — uniform case
        # reduces to rho_d * rho_l = (loading_default)^2 * (loading_rr)^2.
        # In the one-factor Gaussian copula the default loading is sqrt(rho)
        # so the cross factor is rho * rr_loading[i]^2.
        return self._copula.correlation() * self._rr_loading[i] ** 2

    def conditional_default_probability(
        self, prob: float, m: float
    ) -> float:
        """Per-name conditional default probability at factor ``m``.

        # C++ parity: spotlosslatentmodel.hpp:159-177 (the same call as
        # DefaultLatentModel for the single-factor case).
        """
        return self._copula.conditional_probability(prob, m)

    def exp_conditional_recovery(
        self,
        i_name: int,
        prob: float,
        m: float,
    ) -> float:
        """Expected recovery conditional on default and factor m.

        # C++ parity: spotlosslatentmodel.hpp:218 expCondRecoveryP +
        # 229 expCondRecoveryInvPinvRR — Eq. 44 of Li (2009) under the
        # one-factor reduction.
        """
        qassert.require(0 <= i_name < self._n, f"i_name {i_name} out of range")
        if prob < 1e-10:
            return self._rr_mean[i_name]
        rho = self._copula.correlation()
        # Inverse Phi of the unconditional default prob and mean RR
        inv_p = self._inv_norm(prob)
        inv_rr = self._inv_norm(self._rr_mean[i_name])
        cross = self._cross_idiosync(i_name)
        sum_betas_loss = self._rr_loading[i_name] ** 2
        # Eq. 44 reduction with sum_ms = sqrt(rho) m
        sum_ms = np.sqrt(rho) * m
        num = (
            sum_ms
            + np.sqrt(1.0 - cross) * np.sqrt(1.0 + self._model_a**2) * inv_rr
            - np.sqrt(cross) * inv_p
        )
        denom = np.sqrt(
            1.0
            - sum_betas_loss
            + self._model_a**2 * (1.0 - cross)
        )
        return float(self._cum_norm(float(num / denom)))

    def expected_loss(self, i_name: int, prob: float) -> float:
        """Single-name expected loss = pd * (1 - exp RR).

        # C++ parity: spotlosslatentmodel.hpp:318. Integrates the
        # conditional product over M.
        """
        qassert.require(0 <= i_name < self._n, f"i_name {i_name} out of range")
        if prob < 1e-10:
            return 0.0
        self._copula.calculate()
        acc = 0.0
        for k in range(self._copula.steps()):
            mk = self._copula.m(k)
            pd_m = self._copula.conditional_probability(prob, mk)
            rr_m = self.exp_conditional_recovery(i_name, prob, mk)
            acc += pd_m * (1.0 - rr_m) * self._copula.density_dm(k)
        return acc
