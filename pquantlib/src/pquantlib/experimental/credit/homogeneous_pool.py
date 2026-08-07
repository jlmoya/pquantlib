"""HomogeneousPool — pool with uniform notional + uniform recovery.

# C++ parity: ql/experimental/credit/homogeneouspooldef.hpp:42-161 (v1.43).

The C++ header defines :class:`HomogeneousPoolLossModel` — the exact loss
distribution of a finite pool of equal-LGD names, obtained by convolving the
per-name conditional default distributions on a bucket grid and integrating
the result over a one-factor market variable.

The module also carries ``HomogeneousPool``, a PQuantLib-only ``Pool``
subclass recording the uniform notional / recovery every issuer shares. It is
NOT a port of a C++ class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentmodel,
)
from pquantlib.experimental.credit.distribution import Distribution
from pquantlib.experimental.credit.loss_distribution import LossDistHomogeneous
from pquantlib.experimental.credit.pool import Pool
from pquantlib.experimental.credit.saddlepoint_loss_model import (
    remaining_probabilities,
)

if TYPE_CHECKING:
    from pquantlib.experimental.credit.basket import Basket
    from pquantlib.time.date import Date


class HomogeneousPool(Pool):
    """Pool where every issuer shares one notional and one recovery rate.

    # C++ parity divergence: the C++ HomogeneousPoolLossModel is a
    # full loss-distribution computation primed with these scalars +
    # a copula. The Python foundation strips that out — the loss-model
    # computation lives in W3-B/C as ``HomogeneousPoolLossModel``
    # under ``pquantlib.experimental.credit.loss_models``.

    Adds two scalar attributes:

      - ``uniform_notional``: the notional each issuer contributes
        (e.g. 1.0 for a normalised basket).
      - ``uniform_recovery_rate``: the recovery rate applied uniformly
        across issuers on default.
    """

    __slots__ = ("_uniform_notional", "_uniform_recovery_rate")

    def __init__(
        self,
        uniform_notional: float = 1.0,
        uniform_recovery_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self._uniform_notional = uniform_notional
        self._uniform_recovery_rate = uniform_recovery_rate

    def uniform_notional(self) -> float:
        """Common notional per issuer."""
        return self._uniform_notional

    def uniform_recovery_rate(self) -> float:
        """Common recovery rate applied across issuers."""
        return self._uniform_recovery_rate

    def notionals(self) -> list[float]:
        """Return per-name notionals (uniform for homogeneous pool)."""
        return [self._uniform_notional] * self.size()

    def recovery_rates(self) -> list[float]:
        """Return per-name recovery rates (uniform for homogeneous pool)."""
        return [self._uniform_recovery_rate] * self.size()


class HomogeneousPoolLossModel:
    """Default loss distribution convolution for a finite homogeneous pool.

    # C++ parity: ``template<class copulaPolicy> class HomogeneousPoolLossModel``
    # (homogeneouspooldef.hpp:42-97, definitions :105-161).

    The C++ template parameter is the copula policy; following the PQuantLib
    convention the policy arrives inside the :class:`ConstantLossLatentmodel`
    passed to the constructor.

    A note on the number of buckets, from the C++ header: "As it is now the
    code goes splitting losses into buckets from loses equal to zero to losses
    up to the value of the underlying basket. This is in view of a stochastic
    loss given default but in a constant LGD situation this is a waste and it
    is more efficient to go up to the attainable losses."

    Args:
        copula: constant-recovery default latent model; must be one-factor.
        n_buckets: number of loss buckets in the convolution grid.
        max: upper end of the market-factor integration range.
        min: lower end of the market-factor integration range.
        n_steps: number of midpoint steps across ``[min, max]``.
    """

    __slots__ = (
        "_attach",
        "_attach_amount",
        "_basket",
        "_copula",
        "_delta",
        "_detach",
        "_detach_amount",
        "_max",
        "_min",
        "_n_buckets",
        "_n_steps",
        "_notional",
        "_notionals",
    )

    def __init__(
        self,
        copula: ConstantLossLatentmodel,
        n_buckets: int,
        max: float = 5.0,  # C++ parameter name (shadows the builtin)
        min: float = -5.0,  # C++ parameter name (shadows the builtin)
        n_steps: int = 50,
    ) -> None:
        # C++ parity: homogeneouspooldef.hpp:48-61.
        self._copula = copula
        self._n_buckets = n_buckets
        self._max = max
        self._min = min
        self._n_steps = n_steps
        self._delta = (max - min) / n_steps
        qassert.require(
            copula.num_factors() == 1,
            "Inhomogeneous model not implemented for multifactor",
        )
        self._basket: Basket | None = None
        self._attach: float = 0.0
        self._detach: float = 0.0
        self._notional: float = 0.0
        self._attach_amount: float = 0.0
        self._detach_amount: float = 0.0
        self._notionals: list[float] = []

    # ---- basket wiring ---------------------------------------------------

    def set_basket(self, basket: object) -> None:
        """# C++ parity: ``DefaultLossModel::setBasket`` + ``resetModel``."""
        from pquantlib.experimental.credit.basket import Basket as _Basket  # noqa: PLC0415

        qassert.require(isinstance(basket, _Basket), "not a Basket")
        assert isinstance(basket, _Basket)
        self._basket = basket
        self.reset_model()

    def basket(self) -> Basket:
        qassert.require(self._basket is not None, "No portfolio basket set.")
        assert self._basket is not None
        return self._basket

    def reset_model(self) -> None:
        """Re-read every basket-derived cache.

        # C++ parity: homogeneouspooldef.hpp:105-120. The attach/detach ratios
        # "need to be capped now since the limit amounts might be over the
        # remaining notional (think amortizing)".

        # C++ parity divergence: the ``remaining*`` accessors collapse to the
        # inception values on the PQuantLib ``Basket``, which carries no
        # defaulted-name accounting.
        """
        basket = self.basket()
        self._attach = min(
            basket.attachment_amount() / basket.remaining_notional(), 1.0
        )
        self._detach = min(
            basket.detachment_amount() / basket.remaining_notional(), 1.0
        )
        self._notional = basket.remaining_notional()
        self._notionals = basket.notionals()
        self._attach_amount = basket.attachment_amount()
        self._detach_amount = basket.detachment_amount()

        self._copula.reset_basket(basket)

    # ---- accessors -------------------------------------------------------

    def n_buckets(self) -> int:
        return self._n_buckets

    def copula(self) -> ConstantLossLatentmodel:
        return self._copula

    # ---- the model -------------------------------------------------------

    def loss_distrib(self, d: Date) -> Distribution:
        """Portfolio loss distribution at ``d``, on ``[0, detach_amount]``.

        # C++ parity: homogeneouspooldef.hpp:122-161.
        """
        buckt_l_dist_buff = LossDistHomogeneous(self._n_buckets, self._detach_amount)

        recoveries = self._copula.recoveries()
        lgd = [1.0 - x for x in recoveries]
        lgd = [lgd[i] * self._notionals[i] for i in range(len(lgd))]
        prob = remaining_probabilities(self.basket(), d)
        prob = [
            self._copula.inverse_cumulative_y(prob[i], i) for i in range(len(prob))
        ]

        # integrate locally (1 factor).
        dist = Distribution(self._n_buckets, 0.0, self._detach_amount)
        mkft = [self._min + self._delta / 2.0]
        for _ in range(self._n_steps):
            conditional_probs = [
                self._copula.conditional_default_probability_inv_p(
                    prob[i_name], i_name, mkft
                )
                for i_name in range(len(self._notionals))
            ]
            bld = buckt_l_dist_buff(lgd, conditional_probs)
            # # C++ parity: ``copula_->density(mkft)`` is
            # ``LatentModel::density`` (latentmodel.hpp:296-303), a straight
            # passthrough to the copula policy. The PQuantLib ``LatentModel``
            # has no such passthrough, so the policy is asked directly.
            densitydm = self._delta * self._copula.copula().density(mkft)
            for j in range(self._n_buckets):
                dist.add_density(j, bld.density(j) * densitydm)
            mkft[0] += self._delta
        return dist

    def expected_tranche_loss(self, d: Date) -> float:
        """# C++ parity: homogeneouspooldef.hpp:65-73."""
        return self.loss_distrib(d).cumulative_excess_probability(
            self._attach_amount, self._detach_amount
        )

    def percentile(self, d: Date, percentile: float) -> float:
        """# C++ parity: homogeneouspooldef.hpp:74-77."""
        portf_loss = self.loss_distrib(d).confidence_level(percentile)
        return min(
            max(portf_loss - self._attach_amount, 0.0),
            self._detach_amount - self._attach_amount,
        )

    def expected_shortfall(self, d: Date, percentile: float) -> float:
        """# C++ parity: homogeneouspooldef.hpp:78-82."""
        dist = self.loss_distrib(d)
        dist.tranche(self._attach_amount, self._detach_amount)
        return dist.expected_shortfall(percentile)


# C++ parity: the two typedefs at homogeneouspooldef.hpp:99-101. Python has no
# template arguments to bind, so they are aliases of the one class; the copula
# policy is chosen when the ``ConstantLossLatentmodel`` is built.
HomogGaussPoolLossModel = HomogeneousPoolLossModel
HomogTPoolLossModel = HomogeneousPoolLossModel


__all__ = [
    "HomogGaussPoolLossModel",
    "HomogTPoolLossModel",
    "HomogeneousPool",
    "HomogeneousPoolLossModel",
]
