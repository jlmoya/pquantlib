"""InhomogeneousPool — pool with per-issuer notionals + recoveries.

# C++ parity: ql/experimental/credit/inhomogeneouspooldef.hpp:45-170 (v1.43).

The C++ header defines :class:`InhomogeneousPoolLossModel` — the loss
distribution of a finite pool of heterogeneous LGDs, obtained by Hull-White
bucketing of the per-name conditional default distributions and integrating
the result over a one-factor market variable.

The module also carries ``InhomogeneousPool``, a PQuantLib-only ``Pool``
subclass recording per-issuer notional + recovery. It is NOT a port of a C++
class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentmodel,
)
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.distribution import Distribution
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.experimental.credit.loss_distribution import LossDistBucketing
from pquantlib.experimental.credit.pool import Pool
from pquantlib.experimental.credit.saddlepoint_loss_model import (
    remaining_probabilities,
)

if TYPE_CHECKING:
    from pquantlib.experimental.credit.basket import Basket
    from pquantlib.time.date import Date


class InhomogeneousPool(Pool):
    """Pool where each issuer carries its own notional + recovery rate.

    The two arrays (``notionals`` and ``recovery_rates``) must be kept
    in sync with ``names()``. They grow when ``add_with_attributes`` is
    used; ``add`` (inherited from ``Pool``) requires the caller to push
    matching entries explicitly.

    # C++ parity divergence: the C++ InhomogeneousPoolLossModel is a
    # full loss-distribution computation. The Python foundation strips
    # that out — the loss-model computation lives in W3-B/C as
    # ``InhomogeneousPoolLossModel``.
    """

    __slots__ = ("_notionals", "_recovery_rates")

    def __init__(self) -> None:
        super().__init__()
        self._notionals: list[float] = []
        self._recovery_rates: list[float] = []

    def notionals(self) -> list[float]:
        return list(self._notionals)

    def recovery_rates(self) -> list[float]:
        return list(self._recovery_rates)

    def add_with_attributes(
        self,
        name: str,
        issuer: Issuer,
        notional: float,
        recovery_rate: float,
        contract_trigger: DefaultProbKey | None = None,
    ) -> None:
        """Add an issuer with explicit notional + recovery.

        If the name is already present, this is a no-op (mirrors
        ``Pool.add``). Otherwise the issuer is registered AND the
        parallel attribute arrays are extended.
        """
        already_present = self.has(name)
        super().add(name, issuer, contract_trigger)
        if not already_present:
            self._notionals.append(notional)
            self._recovery_rates.append(recovery_rate)

    def clear(self) -> None:
        super().clear()
        self._notionals.clear()
        self._recovery_rates.clear()


class InhomogeneousPoolLossModel:
    """Default loss distribution convolution for a finite non-homogeneous pool.

    # C++ parity: ``template<class copulaPolicy> class InhomogeneousPoolLossModel``
    # (inhomogeneouspooldef.hpp:45-105, definitions :113-170).

    The C++ template parameter is the copula policy; following the PQuantLib
    convention the policy arrives inside the :class:`ConstantLossLatentmodel`
    passed to the constructor.

    A note on the number of buckets, from the C++ header: "As it is now the
    code goes splitting losses into buckets from loses equal to zero to losses
    up to the value of the underlying basket. This is in view of a stochastic
    loss given default but in a constant LGD situation this is a waste and it
    is more efficient to go up to the attainable losses." The header also
    records: "Many common code with the homogeneous version, both classes
    perform the same work on different loss distribution types, merge and
    send the distribution object?" — the duplication below is C++'s.

    # C++ parity: the class also declares ``typedef copulaPolicy copulaType``
    # (inhomogeneouspooldef.hpp:52) "allow base correlations"; the Python
    # equivalent is :meth:`copula`.

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
        # C++ parity: inhomogeneouspooldef.hpp:54-68.
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

        # C++ parity: inhomogeneouspooldef.hpp:113-128. The attach/detach ratios
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

        # C++ parity: inhomogeneouspooldef.hpp:130-170.
        """
        buckt_l_dist_buff = LossDistBucketing(self._n_buckets, self._detach_amount)

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
        """# C++ parity: inhomogeneouspooldef.hpp:73-81."""
        return self.loss_distrib(d).cumulative_excess_probability(
            self._attach_amount, self._detach_amount
        )

    def percentile(self, d: Date, percentile: float) -> float:
        """# C++ parity: inhomogeneouspooldef.hpp:82-85."""
        portf_loss = self.loss_distrib(d).confidence_level(percentile)
        return min(
            max(portf_loss - self._attach_amount, 0.0),
            self._detach_amount - self._attach_amount,
        )

    def expected_shortfall(self, d: Date, percentile: float) -> float:
        """# C++ parity: inhomogeneouspooldef.hpp:86-90."""
        dist = self.loss_distrib(d)
        dist.tranche(self._attach_amount, self._detach_amount)
        return dist.expected_shortfall(percentile)


# C++ parity: the two typedefs at inhomogeneouspooldef.hpp:106-109. Python has
# no template arguments to bind, so they are aliases of the one class; the
# copula policy is chosen when the ``ConstantLossLatentmodel`` is built.
IHGaussPoolLossModel = InhomogeneousPoolLossModel
IHStudentPoolLossModel = InhomogeneousPoolLossModel


__all__ = [
    "IHGaussPoolLossModel",
    "IHStudentPoolLossModel",
    "InhomogeneousPool",
    "InhomogeneousPoolLossModel",
]
