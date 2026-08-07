"""SaddlePointLossModel — saddle-point portfolio credit default loss model.

# C++ parity: ql/experimental/credit/saddlepointlossmodel.hpp:99-1362 (v1.43).

Default loss model implementing the saddle-point expansion integrations on
several default risk metrics. Codependence is handled through a latent model
which makes the integrals conditional on the latent model factor; the latent
variables are then integrated out.

The pipeline is four stages deep::

    cumulants K, K', K'', K''', K''''   (closed form, per market factor)
        -> Brent search for the saddle s* solving K'(s*) = loss
            -> a high-order expansion in (s*, K(s*), K''(s*), K'''(s*), ...)
                -> integration over the market factor

See the references listed in the C++ header (Martin/Thompson/Browne RISK 2001,
Martin RISK 2006, Antonov/Mechkov/Misirpashaev 2005, Huang/Oosterlee/Mesters
2007, Butler 2007).

Every value in this module is cross-validated against
``migration-harness/references/v143/experimental/creditloss.json``
(``saddle_*`` keys), which the
``migration-harness/cpp/probes/v143_experimental_creditloss`` probe produces by
running C++ v1.43 itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentmodel,
)
from pquantlib.math.constants import M_PI, QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.patterns.observable_settings import ObservableSettings

if TYPE_CHECKING:
    from pquantlib.experimental.credit.basket import Basket
    from pquantlib.time.date import Date


def remaining_probabilities(basket: Basket, d: Date) -> list[float]:
    """Per-name unconditional default probability by ``d`` for the live names.

    # C++ parity: ``Basket::remainingProbabilities`` (basket.cpp:209-221) —
    # walks ``liveList()`` and asks each issuer's default-probability curve
    # for ``defaultProbability(d, true)`` (extrapolation ON).

    # C++ parity divergence: this belongs on ``Basket``, but the PQuantLib
    # ``Basket`` carries no defaulted-name accounting yet (see the
    # ``remaining_notional`` note in ``basket.py``), so ``liveList()`` is the
    # whole pool and the "remaining" probabilities are simply
    # ``Basket::probabilities(d)``. Hosted here until ``basket.py`` grows the
    # remaining-* family.
    """
    pool = basket.pool()
    names = pool.names()
    keys = pool.default_keys()
    return [
        pool.get(names[i]).default_probability(keys[i]).default_probability(d)
        for i in range(basket.size())
    ]


class SaddlePointLossModel:
    """Saddle-point expansion loss model over a constant-loss latent model.

    # C++ parity: ``template<class CP> class SaddlePointLossModel``
    # (saddlepointlossmodel.hpp:99-380).

    The C++ template parameter is the copula policy; following the PQuantLib
    convention (see ``experimental/math/latent_model.py``) the policy arrives
    inside the :class:`ConstantLossLatentmodel` passed to the constructor.

    Args:
        m: the constant-recovery default latent model this expansion runs on.
    """

    __slots__ = (
        "_attach_ratio",
        "_basket",
        "_copula",
        "_detach_ratio",
        "_remaining_notional",
        "_remaining_notionals",
        "_remaining_size",
    )

    def __init__(self, m: ConstantLossLatentmodel) -> None:
        # C++ parity: saddlepointlossmodel.hpp:102-104.
        self._copula = m
        self._basket: Basket | None = None
        self._remaining_notionals: list[float] = []
        self._remaining_notional: float = 0.0
        self._remaining_size: int = 0
        self._attach_ratio: float = 0.0
        self._detach_ratio: float = 0.0

    # ---- basket wiring ---------------------------------------------------

    def set_basket(self, basket: object) -> None:
        """Receive the owning basket and refresh the cached arguments.

        # C++ parity: ``DefaultLossModel::setBasket`` (defaultlossmodel.hpp:141-154)
        # followed by ``resetModel``.

        # C++ parity divergence: C++ reaches this lazily, from
        # ``Basket::performCalculations``; the PQuantLib ``Basket`` calls it
        # eagerly from ``set_loss_model``. Same end state.
        """
        from pquantlib.experimental.credit.basket import Basket as _Basket  # noqa: PLC0415

        qassert.require(isinstance(basket, _Basket), "not a Basket")
        assert isinstance(basket, _Basket)
        self._basket = basket
        self.reset_model()

    def basket(self) -> Basket:
        """The attached basket, or raise if none was set."""
        qassert.require(self._basket is not None, "No portfolio basket set.")
        assert self._basket is not None
        return self._basket

    def reset_model(self) -> None:
        """Re-read every basket-derived cache.

        # C++ parity: saddlepointlossmodel.hpp:338-346.
        """
        basket = self.basket()
        # # C++ parity divergence: ``remainingNotionals()`` /
        # ``remainingAttachmentAmount()`` / ``remainingDetachmentAmount()``
        # are the eval-date caches of a basket with no settled defaults, which
        # is the only state the PQuantLib ``Basket`` models.
        self._remaining_notionals = basket.notionals()
        self._remaining_notional = basket.remaining_notional()
        self._remaining_size = basket.remaining_size()
        self._attach_ratio = min(
            basket.attachment_amount() / basket.remaining_notional(), 1.0
        )
        self._detach_ratio = min(
            basket.detachment_amount() / basket.remaining_notional(), 1.0
        )
        self._copula.reset_basket(basket)

    def _inv_uncond_probs(self, d: Date) -> list[float]:
        """Copula-inverted unconditional default probabilities at ``d``.

        Every integrated statistic starts here.

        # C++ parity: the identical three-line preamble repeated at
        # saddlepointlossmodel.hpp:390-394, 406-410, 422-426, 438-442,
        # 454-458, 475-479, 491-495, 507-511, 523-527, 538-542, 1342-1346.
        """
        probs = remaining_probabilities(self.basket(), d)
        return [
            self._copula.inverse_cumulative_y(probs[i], i) for i in range(len(probs))
        ]

    # ---- conditional cumulants and derivatives ---------------------------

    def cumulant_generating_cond(
        self,
        inv_uncond_probs: Sequence[float],
        loss_fraction: float,
        mkt_factor: Sequence[float],
    ) -> float:
        r"""Cumulant generating function conditional on the market factor.

        :math:`K = \sum_j \ln(1 - p_j + p_j e^{N_j \, lgd_j \, s})`

        # C++ parity: saddlepointlossmodel.hpp:565-584.
        """
        n_names = len(self._remaining_notionals)
        total = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            total += math.log(
                1.0
                - p_buffer
                + p_buffer
                * math.exp(
                    self._remaining_notionals[i_name]
                    * (
                        1.0
                        - self._copula.conditional_recovery_inv_p(
                            inv_uncond_probs[i_name], i_name, mkt_factor
                        )
                    )
                    * loss_fraction
                    / self._remaining_notional
                )
            )
        return total

    def _loss_in_def(
        self,
        inv_uncond_probs: Sequence[float],
        i_name: int,
        mkt_factor: Sequence[float],
    ) -> float:
        """Per-name loss given default, in fractional portfolio units.

        # C++ parity: the ``lossInDef`` local recomputed identically at
        # saddlepointlossmodel.hpp:600-602, 623-625, 647-649, 678-680,
        # 714-716, 754-756.
        """
        return (
            self._remaining_notionals[i_name]
            * (
                1.0
                - self._copula.conditional_recovery_inv_p(
                    inv_uncond_probs[i_name], i_name, mkt_factor
                )
            )
            / self._remaining_notional
        )

    def cum_gen_1st_derivative_cond(
        self,
        inv_uncond_probs: Sequence[float],
        saddle: float,
        mkt_factor: Sequence[float],
    ) -> float:
        r"""First derivative of the conditional CGF.

        :math:`K_1 = \sum_j \frac{p_j N_j LGD_j e^{N_j LGD_j s}}
        {1 - p_j + p_j e^{N_j LGD_j s}}`

        At ``saddle == 0`` this is the conditional portfolio expected loss in
        fractional units; at infinity it is the maximum attainable loss.

        # C++ parity: saddlepointlossmodel.hpp:586-607.
        """
        n_names = len(self._remaining_notionals)
        total = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            loss_in_def = self._loss_in_def(inv_uncond_probs, i_name, mkt_factor)
            mid_factor = p_buffer * math.exp(loss_in_def * saddle)
            total += loss_in_def * mid_factor / (1.0 - p_buffer + mid_factor)
        return total

    def cum_gen_2nd_derivative_cond(
        self,
        inv_uncond_probs: Sequence[float],
        saddle: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Second derivative of the conditional CGF.

        # C++ parity: saddlepointlossmodel.hpp:609-632.
        """
        n_names = len(self._remaining_notionals)
        total = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            loss_in_def = self._loss_in_def(inv_uncond_probs, i_name, mkt_factor)
            mid_factor = p_buffer * math.exp(loss_in_def * saddle)
            denominator = 1.0 - p_buffer + mid_factor
            total += loss_in_def * loss_in_def * mid_factor / denominator - math.pow(
                loss_in_def * mid_factor / denominator, 2.0
            )
        return total

    def cum_gen_3rd_derivative_cond(
        self,
        inv_uncond_probs: Sequence[float],
        saddle: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Third derivative of the conditional CGF.

        # C++ parity: saddlepointlossmodel.hpp:634-663.
        """
        n_names = len(self._remaining_notionals)
        total = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            loss_in_def = self._loss_in_def(inv_uncond_probs, i_name, mkt_factor)
            mid_factor = p_buffer * math.exp(loss_in_def * saddle)
            suma0 = 1.0 - p_buffer + mid_factor
            suma1 = loss_in_def * mid_factor
            suma2 = loss_in_def * suma1
            suma3 = loss_in_def * suma2
            total += (
                suma3 + (2.0 * math.pow(suma1, 3.0) / suma0 - 3.0 * suma1 * suma2) / suma0
            ) / suma0
        return total

    def cum_gen_4th_derivative_cond(
        self,
        inv_uncond_probs: Sequence[float],
        saddle: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Fourth derivative of the conditional CGF.

        # C++ parity: saddlepointlossmodel.hpp:665-696.
        """
        n_names = len(self._remaining_notionals)
        total = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            loss_in_def = self._loss_in_def(inv_uncond_probs, i_name, mkt_factor)
            mid_factor = p_buffer * math.exp(loss_in_def * saddle)
            suma0 = 1.0 - p_buffer + mid_factor
            suma1 = loss_in_def * mid_factor
            suma2 = loss_in_def * suma1
            suma3 = loss_in_def * suma2
            suma4 = loss_in_def * suma3
            total += (
                suma4
                + (
                    -4.0 * suma1 * suma3
                    - 3.0 * suma2 * suma2
                    + (
                        12.0 * suma1 * suma1 * suma2
                        - 6.0 * math.pow(suma1, 4.0) / suma0
                    )
                    / suma0
                )
                / suma0
            ) / suma0
        return total

    def cum_gen_0234_deriv_cond(
        self,
        inv_uncond_probs: Sequence[float],
        saddle: float,
        mkt_factor: Sequence[float],
    ) -> tuple[float, float, float, float]:
        """``(K, K'', K''', K'''')`` in one pass.

        # C++ parity: saddlepointlossmodel.hpp:698-738. Note the C++ builds
        # ``deriv0`` as ``sum log(suma0)`` — i.e. it re-derives
        # ``CumulantGeneratingCond`` from the same ``suma0`` denominator
        # rather than calling it.
        """
        n_names = len(self._remaining_notionals)
        deriv0 = 0.0
        deriv2 = 0.0
        deriv3 = 0.0
        deriv4 = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            loss_in_def = self._loss_in_def(inv_uncond_probs, i_name, mkt_factor)
            mid_factor = p_buffer * math.exp(loss_in_def * saddle)
            suma0 = 1.0 - p_buffer + mid_factor
            suma1 = loss_in_def * mid_factor
            suma2 = loss_in_def * suma1
            suma3 = loss_in_def * suma2
            suma4 = loss_in_def * suma3

            deriv0 += math.log(suma0)
            deriv2 += suma2 / suma0 - math.pow(suma1 / suma0, 2.0)
            deriv3 += (
                suma3 + (2.0 * math.pow(suma1, 3.0) / suma0 - 3.0 * suma1 * suma2) / suma0
            ) / suma0
            deriv4 += (
                suma4
                + (
                    -4.0 * suma1 * suma3
                    - 3.0 * suma2 * suma2
                    + (
                        12.0 * suma1 * suma1 * suma2
                        - 6.0 * math.pow(suma1, 4.0) / suma0
                    )
                    / suma0
                )
                / suma0
            ) / suma0
        return deriv0, deriv2, deriv3, deriv4

    def cum_gen_02_deriv_cond(
        self,
        inv_uncond_probs: Sequence[float],
        saddle: float,
        mkt_factor: Sequence[float],
    ) -> tuple[float, float]:
        """``(K, K'')`` in one pass.

        # C++ parity: saddlepointlossmodel.hpp:740-771.
        """
        n_names = len(self._remaining_notionals)
        deriv0 = 0.0
        deriv2 = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            loss_in_def = self._loss_in_def(inv_uncond_probs, i_name, mkt_factor)
            mid_factor = p_buffer * math.exp(loss_in_def * saddle)
            suma0 = 1.0 - p_buffer + mid_factor
            suma1 = loss_in_def * mid_factor
            suma2 = loss_in_def * suma1

            deriv0 += math.log(suma0)
            deriv2 += suma2 / suma0 - math.pow(suma1 / suma0, 2.0)
        return deriv0, deriv2

    # ---- unconditional cumulants -----------------------------------------
    #
    # # C++ parity: saddlepointlossmodel.hpp:164-174 — "Because this class
    # integrates the various statistics it provides in indirect mode they are
    # never used. Provided for completeness/extendability".

    def cumulant_generating(self, date: Date, s: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:386-400."""
        inv = self._inv_uncond_probs(date)
        return self._copula.integrated_expected_value(
            lambda v1: self.cumulant_generating_cond(inv, s, v1)
        )

    def cum_gen_1st_derivative(self, date: Date, s: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:402-416."""
        inv = self._inv_uncond_probs(date)
        return self._copula.integrated_expected_value(
            lambda v1: self.cum_gen_1st_derivative_cond(inv, s, v1)
        )

    def cum_gen_2nd_derivative(self, date: Date, s: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:418-432."""
        inv = self._inv_uncond_probs(date)
        return self._copula.integrated_expected_value(
            lambda v1: self.cum_gen_2nd_derivative_cond(inv, s, v1)
        )

    def cum_gen_3rd_derivative(self, date: Date, s: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:434-448."""
        inv = self._inv_uncond_probs(date)
        return self._copula.integrated_expected_value(
            lambda v1: self.cum_gen_3rd_derivative_cond(inv, s, v1)
        )

    def cum_gen_4th_derivative(self, date: Date, s: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:450-464."""
        inv = self._inv_uncond_probs(date)
        return self._copula.integrated_expected_value(
            lambda v1: self.cum_gen_4th_derivative_cond(inv, s, v1)
        )

    # ---- saddle point search ---------------------------------------------

    class SaddleObjectiveFunction:
        """``K'(x) - target``, the function the saddle search brackets.

        # C++ parity: the nested ``class SaddleObjectiveFunction``
        # (saddlepointlossmodel.hpp:177-202). The passed target is in
        # fractional loss units.
        """

        __slots__ = ("_inv_uncond_probs", "_me", "_mkt_factor", "_target_value")

        def __init__(
            self,
            me: SaddlePointLossModel,
            target: float,
            inv_uncond_probs: Sequence[float],
            mkt_factor: Sequence[float],
        ) -> None:
            # C++ parity: saddlepointlossmodel.hpp:184-193.
            self._me = me
            self._target_value = target
            self._mkt_factor = mkt_factor
            self._inv_uncond_probs = inv_uncond_probs

        def __call__(self, x: float) -> float:
            """# C++ parity: saddlepointlossmodel.hpp:194-197."""
            return (
                self._me.cum_gen_1st_derivative_cond(
                    self._inv_uncond_probs, x, self._mkt_factor
                )
                - self._target_value
            )

        def derivative(self, x: float) -> float:
            """# C++ parity: saddlepointlossmodel.hpp:198-201."""
            return self._me.cum_gen_2nd_derivative_cond(
                self._inv_uncond_probs, x, self._mkt_factor
            )

    def find_saddle(
        self,
        inv_uncond_ps: Sequence[float],
        loss_level: float,
        mkt_factor: Sequence[float],
        accuracy: float = 1.0e-3,
        max_evaluations: int = 50,
    ) -> float:
        """Market-factor-conditional saddle point for the given loss level.

        ``loss_level`` is in total-portfolio fractional loss units. Below the
        resolvable minimum loss (or above the maximum) the bracket endpoint is
        returned instead of a solved root — "typically the functionals to
        integrate will have a low dependency on this point".

        # C++ parity: saddlepointlossmodel.hpp:775-840.
        """
        f = SaddlePointLossModel.SaddleObjectiveFunction(
            self, loss_level, inv_uncond_ps, mkt_factor
        )

        n_names = len(self._remaining_notionals)
        lgds = [
            self._remaining_notionals[i]
            * (
                1.0
                - self._copula.conditional_recovery_inv_p(
                    inv_uncond_ps[i], i, mkt_factor
                )
            )
            for i in range(n_names)
        ]

        # Position of the name with the largest relative exposure loss. C++
        # uses std::max_element, which returns the FIRST maximum on ties.
        i_nam_max = lgds.index(max(lgds))
        # Gap to be considered zero at the negative side of the logistic
        # inversion. # C++ parity: saddlepointlossmodel.hpp:803.
        delta_min = 1.0e-5

        p_max_name = self._copula.conditional_default_probability_inv_p(
            inv_uncond_ps[i_nam_max], i_nam_max, mkt_factor
        )
        rel_lgd_max = lgds[i_nam_max] / self._remaining_notional
        # Approximates the saddle point corresponding to this minimum, using
        # only the smallest logistic term, so it is below the true value.
        saddle_min = (
            1.0
            / rel_lgd_max
            * math.log(
                delta_min
                * (1.0 - p_max_name)
                / (p_max_name * rel_lgd_max - p_max_name * delta_min)
            )
        )
        min_loss = self.cum_gen_1st_derivative_cond(
            inv_uncond_ps, saddle_min, mkt_factor
        )
        if loss_level < min_loss:
            return saddle_min

        saddle_max = (
            1.0
            / rel_lgd_max
            * math.log(
                (rel_lgd_max - delta_min) * (1.0 - p_max_name) / (p_max_name * delta_min)
            )
        )
        max_loss = self.cum_gen_1st_derivative_cond(
            inv_uncond_ps, saddle_max, mkt_factor
        )
        if loss_level > max_loss:
            return saddle_max

        solver_brent = Brent()
        guess = (saddle_min + saddle_max) / 2.0
        solver_brent.set_max_evaluations(max_evaluations)
        return solver_brent.solve(f, accuracy, guess, saddle_min, saddle_max)

    class SaddlePercObjFunction:
        """``P(L >= x) - (1 - target)`` over the *tranche* loss fraction.

        # C++ parity: the nested ``class SaddlePercObjFunction``
        # (saddlepointlossmodel.hpp:226-240). Note the constructor stores
        # ``1 - target``, not ``target``.
        """

        __slots__ = ("_date", "_me", "_target_value")

        def __init__(
            self, me: SaddlePointLossModel, target: float, date: Date
        ) -> None:
            # C++ parity: saddlepointlossmodel.hpp:231-235.
            self._me = me
            self._target_value = 1.0 - target
            self._date = date

        def __call__(self, x: float) -> float:
            """``x`` is the tranche loss fraction.

            # C++ parity: saddlepointlossmodel.hpp:236-239.
            """
            return self._me.prob_over_loss(self._date, x) - self._target_value

    # ---- portfolio statistics --------------------------------------------

    def percentile(self, d: Date, percentile: float) -> float:
        """Loss amount whose probability of not being exceeded is ``percentile``.

        # C++ parity: saddlepointlossmodel.hpp:846-874.
        """
        qassert.require(
            percentile >= 0.0 and percentile <= 1.0, "Incorrect percentile value."
        )

        # Still does not tackle the situation where we have cumulated losses
        # from previous defaults.
        if d <= ObservableSettings().evaluation_date_or_today():
            return 0.0

        # Trivial cases when the percentile is outside the prob range
        # associated to the tranche limits.
        if percentile <= 1.0 - self.prob_over_loss(d, 0.0):
            return 0.0
        if percentile >= 1.0 - self.prob_over_loss(d, 1.0):
            return self._remaining_tranche_notional()

        f = SaddlePointLossModel.SaddlePercObjFunction(self, percentile, d)
        solver = Brent()
        solver.set_max_evaluations(100)
        min_val = QL_EPSILON
        max_val = 1.0 - QL_EPSILON
        guess = 0.5

        solut = solver.solve(f, 1.0e-4, guess, min_val, max_val)
        return self._remaining_tranche_notional() * solut

    def _remaining_tranche_notional(self) -> float:
        """# C++ parity: ``Basket::remainingTrancheNotional`` (basket.hpp:206-209).

        # C++ parity divergence: hosted here for the same reason as
        # :func:`remaining_probabilities`.
        """
        basket = self.basket()
        return basket.detachment_amount() - basket.attachment_amount()

    def prob_over_loss_cond(
        self,
        inv_uncond_ps: Sequence[float],
        tranche_loss_fract: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Conditional probability of a tranche loss fraction being exceeded.

        ``tranche_loss_fract`` is a fraction of the tranche notional, in [0, 1].

        # C++ parity: saddlepointlossmodel.hpp:876-893.
        """
        portf_fract = self._attach_ratio + tranche_loss_fract * (
            self._detach_ratio - self._attach_ratio
        )  # these are remaining ratios
        return self.prob_over_loss_portf_cond(
            inv_uncond_ps,
            # below; should subtract realized losses. Use remaining amounts??
            portf_fract * self.basket().basket_notional(),
            mkt_factor,
        )

    def prob_over_loss(self, d: Date, tranche_loss_fract: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:466-485.

        # C++ parity note (DEFECT, reproduced verbatim): the early return
        # compares a *fraction* against an *amount* ::

            if (trancheLossFract >=
                // time dependent soon:
                basket_->detachmentAmount()) return 0.;

        # (saddlepointlossmodel.hpp:470-473). ``trancheLossFract`` lives in
        # [0, 1] while ``detachmentAmount()`` is a currency amount, so the
        # shortcut only fires for a basket whose detachment amount is below 1.
        """
        # avoid computation:
        if tranche_loss_fract >= self.basket().detachment_amount():
            return 0.0

        inv = self._inv_uncond_probs(d)
        return self._copula.integrated_expected_value(
            lambda v1: self.prob_over_loss_cond(inv, tranche_loss_fract, v1)
        )

    def loss_distribution(self, d: Date) -> dict[float, float]:
        """Loss amount -> probability of *not* exceeding it.

        # C++ parity: saddlepointlossmodel.hpp:895-905. The grid is C++'s:
        # 500 points, ``lossFraction`` from 1/500 in steps of 1/500 while
        # strictly below 0.45.
        """
        distrib: dict[float, float] = {}
        num_pts = 500.0
        loss_fraction = 1.0 / num_pts
        while loss_fraction < 0.45:
            distrib[loss_fraction * self._remaining_notional] = (
                1.0
                - self.prob_over_portf_loss(
                    d, loss_fraction * self._remaining_notional
                )
            )
            loss_fraction += 1.0 / num_pts
        return distrib

    def prob_over_loss_portf_cond(
        self,
        inv_uncond_probs: Sequence[float],
        loss: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Conditional probability of the *untranched* portfolio loss ``loss``
        being met or exceeded, via the high-order saddle-point expansion.

        ``loss`` is an absolute amount.

        # C++ parity: saddlepointlossmodel.hpp:918-1009. The Antonov et al.
        # correction terms that C++ leaves commented out (hpp:975-983 and
        # :1000-1006, "FIX ME: this term introduces at times numerical
        # instabilty") are left out here too.
        """
        if loss <= QL_EPSILON:
            return 1.0

        relative_loss = loss / self._remaining_notional
        if relative_loss >= 1.0 - QL_EPSILON:
            return 0.0

        n_names = len(self._remaining_notionals)

        average_recovery = 0.0
        for i_name in range(n_names):
            average_recovery += self._copula.conditional_recovery_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
        average_recovery = average_recovery / n_names

        max_att_loss_fract = 1.0 - average_recovery
        if relative_loss > max_att_loss_fract:
            return 0.0

        saddle_pt = self.find_saddle(inv_uncond_probs, relative_loss, mkt_factor)

        base_val, second_val, k3_saddle, k4_saddle = self.cum_gen_0234_deriv_cond(
            inv_uncond_probs, saddle_pt, mkt_factor
        )

        saddle_to2 = saddle_pt * saddle_pt
        saddle_to3 = saddle_to2 * saddle_pt
        saddle_to4 = saddle_to3 * saddle_pt
        saddle_to6 = saddle_to4 * saddle_to2
        k3_saddle_to2 = k3_saddle * k3_saddle

        exponent = base_val - relative_loss * saddle_pt + 0.5 * saddle_to2 * second_val
        if saddle_pt > 0.0:  # <-> (loss > condEL)
            if abs(exponent) > 700.0:
                return 0.0
            return (
                math.exp(exponent)
                * CumulativeNormalDistribution()(
                    -abs(saddle_pt) * math.sqrt(second_val)
                )
                # high order corrections:
                * (
                    1.0
                    - saddle_to3 * k3_saddle / 6.0
                    + saddle_to4 * k4_saddle / 24.0
                    + saddle_to6 * k3_saddle_to2 / 72.0
                )
            )
        if saddle_pt == 0.0:  # <-> (loss == condEL)
            return 0.5
        # <-> (loss < condEL)
        if abs(exponent) > 700.0:
            return 0.0
        return 1.0 - math.exp(exponent) * CumulativeNormalDistribution()(
            -abs(saddle_pt) * math.sqrt(second_val)
        ) * (
            1.0
            - saddle_to3 * k3_saddle / 6.0
            + saddle_to4 * k4_saddle / 24.0
            + saddle_to6 * k3_saddle_to2 / 72.0
        )

    def prob_over_loss_portf_cond_1st_order(
        self,
        inv_uncond_ps: Sequence[float],
        loss: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Cheaper first-order variant of :meth:`prob_over_loss_portf_cond`.

        Fewer terms retained; the cost still lies in the saddle-point search.

        # C++ parity: saddlepointlossmodel.hpp:1011-1072.
        """
        if loss <= QL_EPSILON:
            return 1.0
        n_names = len(self._remaining_notionals)

        relative_loss = loss / self._remaining_notional
        if relative_loss >= 1.0 - QL_EPSILON:
            return 0.0

        # only true for constant recovery models......?
        average_recovery = 0.0
        for i_name in range(n_names):
            average_recovery += self._copula.conditional_recovery_inv_p(
                inv_uncond_ps[i_name], i_name, mkt_factor
            )
        average_recovery = average_recovery / n_names

        max_att_loss_fract = 1.0 - average_recovery
        if relative_loss > max_att_loss_fract:
            return 0.0

        saddle_pt = self.find_saddle(inv_uncond_ps, relative_loss, mkt_factor)

        base_val, second_val = self.cum_gen_02_deriv_cond(
            inv_uncond_ps, saddle_pt, mkt_factor
        )

        saddle_to2 = saddle_pt * saddle_pt
        exponent = base_val - relative_loss * saddle_pt + 0.5 * saddle_to2 * second_val

        if saddle_pt > 0.0:  # <-> (loss > condEL)
            if abs(exponent) > 700.0:
                return 0.0
            # dangerous exponential; fix me
            return math.exp(exponent) * CumulativeNormalDistribution()(
                -abs(saddle_pt) * math.sqrt(second_val)
            )
        if saddle_pt == 0.0:  # <-> (loss == condEL)
            return 0.5
        # <-> (loss < condEL)
        if abs(exponent) > 700.0:
            return 0.0
        return 1.0 - math.exp(exponent) * CumulativeNormalDistribution()(
            -abs(saddle_pt) * math.sqrt(second_val)
        )

    def prob_over_portf_loss(self, d: Date, loss: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:487-501."""
        inv = self._inv_uncond_probs(d)
        return self._copula.integrated_expected_value(
            lambda v1: self.prob_over_loss_portf_cond(inv, loss, v1)
        )

    def expected_tranche_loss(self, d: Date) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:503-517."""
        inv = self._inv_uncond_probs(d)
        return self._copula.integrated_expected_value(
            lambda v1: self.conditional_expected_tranche_loss(inv, v1)
        )

    def prob_density_cond(
        self,
        inv_uncond_ps: Sequence[float],
        loss: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Conditional probability *density* of the untranched portfolio loss.

        # C++ parity: saddlepointlossmodel.hpp:1081-1113. See R. Martin, "The
        # saddle point method and portfolio optionalities", Risk Dec 2006 p.93.
        """
        if loss <= QL_EPSILON:
            return 0.0

        relative_loss = loss / self._remaining_notional
        saddle_pt = self.find_saddle(inv_uncond_ps, relative_loss, mkt_factor)

        k0_saddle, k2_saddle, k3_saddle, k4_saddle = self.cum_gen_0234_deriv_cond(
            inv_uncond_ps, saddle_pt, mkt_factor
        )
        return (
            (
                1.0
                + k4_saddle / (8.0 * math.pow(k2_saddle, 2.0))
                - 5.0 * math.pow(k3_saddle, 2.0) / (24.0 * math.pow(k2_saddle, 3.0))
            )
            * math.exp(k0_saddle - saddle_pt * relative_loss)
            / math.sqrt(2.0 * M_PI * k2_saddle)
        )

    def prob_density(self, d: Date, loss: float) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:519-533."""
        inv = self._inv_uncond_probs(d)
        return self._copula.integrated_expected_value(
            lambda v1: self.prob_density_cond(inv, loss, v1)
        )

    def split_loss_cond(
        self,
        inv_uncond_probs: Sequence[float],
        loss: float,
        mkt_factor: Sequence[float],
    ) -> list[float]:
        """Per-name sensitivities to a given portfolio loss value.

        See equation 8 in "VAR: who contributes and how much?" by R. Martin,
        K. Thompson and C. Browne, Risk Magazine, August 2001.

        # C++ parity: saddlepointlossmodel.hpp:1126-1153. Note the per-name
        # ``lossInDef`` here is the ABSOLUTE loss, unlike everywhere else.
        """
        n_names = len(self._remaining_notionals)
        cond_contrib = [0.0] * n_names
        if loss <= QL_EPSILON:
            return cond_contrib

        saddle_pt = self.find_saddle(
            inv_uncond_probs, loss / self._remaining_notional, mkt_factor
        )

        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            loss_in_def = self._remaining_notionals[i_name] * (
                1.0
                - self._copula.conditional_recovery_inv_p(
                    inv_uncond_probs[i_name], i_name, mkt_factor
                )
            )
            mid_factor = p_buffer * math.exp(
                loss_in_def * saddle_pt / self._remaining_notional
            )
            denominator = 1.0 - p_buffer + mid_factor

            cond_contrib[i_name] = loss_in_def * mid_factor / denominator
        return cond_contrib

    def split_var_level(self, date: Date, s: float) -> list[float]:
        """# C++ parity: saddlepointlossmodel.hpp:535-548."""
        inv = self._inv_uncond_probs(date)
        return self._copula.integrated_expected_value_v(
            lambda v1: self.split_loss_cond(inv, s, v1)
        )

    def conditional_expected_loss(
        self, inv_uncond_probs: Sequence[float], mkt_factor: Sequence[float]
    ) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:1155-1171."""
        n_names = len(self._remaining_notionals)
        eloss = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            eloss += (
                p_buffer
                * self._remaining_notionals[i_name]
                * (
                    1.0
                    - self._copula.conditional_recovery_inv_p(
                        inv_uncond_probs[i_name], i_name, mkt_factor
                    )
                )
            )
        return eloss

    def conditional_expected_tranche_loss(
        self, inv_uncond_probs: Sequence[float], mkt_factor: Sequence[float]
    ) -> float:
        """# C++ parity: saddlepointlossmodel.hpp:1173-1192."""
        eloss = self.conditional_expected_loss(inv_uncond_probs, mkt_factor)
        return min(
            max(eloss - self._attach_ratio * self._remaining_notional, 0.0),
            (self._detach_ratio - self._attach_ratio) * self._remaining_notional,
        )

    def expected_shortfall_split_cond(
        self,
        inv_uncond_probs: Sequence[float],
        loss_perc: float,
        mkt_factor: Sequence[float],
    ) -> list[float]:
        """Per-name expected-shortfall split, conditional on the market factor.

        # C++ parity: saddlepointlossmodel.hpp:1194-1227.
        """
        n_names = len(self._remaining_notionals)
        lgds = [
            self._remaining_notionals[i]
            * (
                1.0
                - self._copula.conditional_recovery_inv_p(
                    inv_uncond_probs[i], i, mkt_factor
                )
            )
            for i in range(n_names)
        ]
        vola = [0.0] * n_names
        mu = [0.0] * n_names
        vola_tot = 0.0
        mu_tot = 0.0
        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            mu[i_name] = (
                lgds[i_name] * p_buffer / self._remaining_notionals[i_name]
            )
            mu_tot += lgds[i_name] * p_buffer
            vola[i_name] = (
                lgds[i_name]
                * lgds[i_name]
                * p_buffer
                * (1.0 - p_buffer)
                / self._remaining_notionals[i_name]
            )
            vola_tot += lgds[i_name] * lgds[i_name] * p_buffer * (1.0 - p_buffer)

        for i_name in range(n_names):
            vola[i_name] = vola[i_name] / vola_tot

        esf_partition = [0.0] * n_names
        for i_name in range(n_names):
            u_edisp = (loss_perc - mu_tot) / math.sqrt(vola_tot)
            esf_partition[i_name] = mu[i_name] * CumulativeNormalDistribution()(
                u_edisp
            ) + vola[i_name] * NormalDistribution()(u_edisp)
        return esf_partition

    def expected_shortfall_tranche_cond(
        self,
        inv_uncond_probs: Sequence[float],
        loss_perc: float,
        percentile: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Conditional tranche expected shortfall.

        # C++ parity: saddlepointlossmodel.hpp:1229-1263. The C++ TODO stands:
        # "this is too crude, a general expression valid for all situations is
        # possible".
        """
        basket = self.basket()
        # tranche correction term:
        correction_term = 0.0
        prob_l_over = self.prob_over_loss_portf_cond(
            inv_uncond_probs, basket.detachment_amount(), mkt_factor
        )
        if basket.attachment_amount() > QL_EPSILON:
            if loss_perc < basket.attachment_amount():
                correction_term = (
                    (basket.detachment_amount() - 2.0 * basket.attachment_amount())
                    * self.prob_over_loss_portf_cond(
                        inv_uncond_probs, loss_perc, mkt_factor
                    )
                    + basket.attachment_amount() * prob_l_over
                ) / (1.0 - percentile)
            else:
                correction_term = (
                    (percentile - 1) * basket.attachment_amount()
                    + basket.detachment_amount() * prob_l_over
                ) / (1.0 - percentile)

        return (
            self.expected_shortfall_full_portfolio_cond(
                inv_uncond_probs,
                max(loss_perc, basket.attachment_amount()),
                mkt_factor,
            )
            + self.expected_shortfall_full_portfolio_cond(
                inv_uncond_probs, basket.detachment_amount(), mkt_factor
            )
            - correction_term
        )

    def expected_shortfall_full_portfolio_cond(
        self,
        inv_uncond_probs: Sequence[float],
        loss_perc: float,
        mkt_factor: Sequence[float],
    ) -> float:
        """Conditional expected shortfall of the whole (untranched) portfolio.

        Based on Martin (2006) and on the expression in "SaddlePoint
        approximation of expected shortfall for transformed means",
        S. A. Broda and M. S. Paolella.

        # C++ parity: saddlepointlossmodel.hpp:1265-1326.
        """
        loss_perc_ratio = loss_perc / self._remaining_notional
        el_cond = 0.0
        n_names = len(self._remaining_notionals)

        for i_name in range(n_names):
            p_buffer = self._copula.conditional_default_probability_inv_p(
                inv_uncond_probs[i_name], i_name, mkt_factor
            )
            el_cond += (
                p_buffer
                * self._remaining_notionals[i_name]
                * (
                    1.0
                    - self._copula.conditional_recovery_inv_p(
                        inv_uncond_probs[i_name], i_name, mkt_factor
                    )
                )
            )
        saddle_pt = self.find_saddle(inv_uncond_probs, loss_perc_ratio, mkt_factor)

        # Martin 2006:
        return (
            el_cond
            * self.prob_over_loss_portf_cond(inv_uncond_probs, loss_perc, mkt_factor)
            + (loss_perc - el_cond)
            * self.prob_density_cond(inv_uncond_probs, loss_perc, mkt_factor)
            / saddle_pt
        )

    def expected_shortfall(self, d: Date, perc_prob: float) -> float:
        """Expected shortfall of the tranche at the ``perc_prob`` percentile.

        # C++ parity: saddlepointlossmodel.hpp:1328-1359.
        """
        # assuming I have the tranched one.
        loss_perc = self.percentile(d, perc_prob)

        # check the trivial case when the loss is over the detachment limit
        # to avoid computation:
        tranche_amount = self.basket().tranche_notional() * (
            self._detach_ratio - self._attach_ratio
        )
        # assumed the amount includes the realized losses
        if loss_perc >= tranche_amount:
            return tranche_amount
        # SHOULD CHECK NOW THE OPPOSITE LIMIT ("zero" losses)....
        inv = self._inv_uncond_probs(d)

        # Integrate with the tranche or the portfolio according to the limits.
        return self._copula.integrated_expected_value(
            lambda v1: self.expected_shortfall_full_portfolio_cond(
                inv, loss_perc, v1
            )
        ) / (1.0 - perc_prob)


# C++ parity note: ``SaddlePointLossModel`` also nominally carries a nested
# ``ESFIntegrator`` (saddlepointlossmodel.hpp:355-379). It is NOT ported
# because in v1.43 the entire class body sits inside a ``/* ... */`` block
# comment, introduced by "Just for testing the ESF direct integration, not for
# release, this is very inneficient:". It compiles to nothing, has no callers,
# and cannot be instantiated from C++ either. Porting it would add a class
# QuantLib does not actually have.


__all__ = ["SaddlePointLossModel", "remaining_probabilities"]
