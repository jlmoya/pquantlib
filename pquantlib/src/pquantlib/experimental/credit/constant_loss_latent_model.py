"""Constant-loss (deterministic recovery) default latent models.

This module hosts three classes:

* :class:`ConstantLossLatentmodel` — the faithful port of the C++ class
  template ``ConstantLossLatentmodel<copulaPolicy>``. The lower-case ``m`` in
  ``Latentmodel`` is C++'s, reproduced verbatim.
  # C++ parity: ql/experimental/credit/constantlosslatentmodel.hpp:37-103 (v1.43).
* :class:`ConstantLossModel` — the ``DefaultLossModel`` face of the same
  model, for pricing digital products such as NTDs.
  # C++ parity: constantlosslatentmodel.hpp:117-164 (v1.43).
* :class:`ConstantLossLatentModel` — a pre-existing one-factor convenience
  adaptation on top of
  :class:`~pquantlib.experimental.credit.default_probability_latent_model.DefaultProbabilityLatentModel`,
  which the W3-B loss models consume. Not a port of a C++ class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.credit.default_probability_latent_model import (
    DefaultLatentModel,
    DefaultProbabilityLatentModel,
    LatentModelIntegrationType,
)
from pquantlib.experimental.credit.one_factor_copula import OneFactorCopula
from pquantlib.experimental.math.latent_model import CopulaPolicy

if TYPE_CHECKING:
    from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
    from pquantlib.time.date import Date


class ConstantLossLatentmodel(DefaultLatentModel):
    """Default latent model with a constant deterministic recovery per name.

    # C++ parity: ``template <class copulaPolicy> class ConstantLossLatentmodel``
    # (constantlosslatentmodel.hpp:37-103).

    # C++ parity note: the class name really is spelled ``Latentmodel`` in
    # C++ — lower-case ``m`` — while every neighbouring class uses
    # ``LatentModel``. Reproduced exactly.

    :param factor_weights: ``factor_weights[i_variable][i_factor]``.
    :param recoveries: one recovery rate per name.
    :param copula: the copula policy built over the same ``factor_weights``.
    :param integral_type: which integration facility to build.
    """

    __slots__ = ("_recoveries_c",)

    def __init__(
        self,
        factor_weights: Sequence[Sequence[float]],
        recoveries: Sequence[float],
        copula: CopulaPolicy,
        integral_type: LatentModelIntegrationType = (
            LatentModelIntegrationType.GaussianQuadrature
        ),
    ) -> None:
        super().__init__(factor_weights, copula, integral_type)
        qassert.require(
            len(recoveries) == len(factor_weights),
            "Incompatible factors and recovery sizes.",
        )
        self._recoveries_c: list[float] = list(recoveries)

    def recoveries(self) -> list[float]:
        """The per-name recovery vector.

        # C++ parity: constantlosslatentmodel.hpp:92-94.
        """
        return self._recoveries_c

    def conditional_recovery(
        self,
        _prob_or_date_or_sample: float | Date,
        i_name: int,
        _mkt_factors_or_date: Sequence[float] | Date,
    ) -> float:
        """Recovery of ``i_name`` — constant, so every argument is ignored.

        # C++ parity: the four overloads at constantlosslatentmodel.hpp:72-90
        # (by date, by unconditional probability, by inverted probability, by
        # latent-variable sample) all collapse to ``recoveries_[iName]``. The
        # port keeps a single method; the leading and trailing arguments exist
        # for signature parity only.
        """
        return self._recoveries_c[i_name]

    def conditional_recovery_inv_p(
        self,
        _inv_uncond_def_p: float,
        i_name: int,
        _mkt_factors: Sequence[float],
    ) -> float:
        """# C++ parity: constantlosslatentmodel.hpp:82-85."""
        return self._recoveries_c[i_name]

    def expected_recovery(
        self, _d: Date, i_name: int, _def_keys: DefaultProbKey
    ) -> float:
        """Expected recovery of ``i_name`` at ``_d``.

        # C++ parity: constantlosslatentmodel.hpp:99-102 — "really an
        # interface to rr models even if not imposed ... enforced only through
        # duck typing".
        """
        return self._recoveries_c[i_name]


class ConstantLossModel(ConstantLossLatentmodel):
    """``DefaultLossModel`` face of :class:`ConstantLossLatentmodel`.

    # C++ parity: ``template <class copulaPolicy> class ConstantLossModel``
    # (constantlosslatentmodel.hpp:117-164).

    Lacking an integration algorithm over the loss distribution it provides no
    tranche-loss statistics; it exists so digital products (NTDs) can be
    priced off the default-event machinery.

    # C++ parity note: the C++ default arguments are written
    # ``copulaPolicy::initTraits()`` without ``typename``
    # (constantlosslatentmodel.hpp:127-128 and 137-138). Clang rejects that as
    # soon as the default is used, so in C++ the traits must be passed
    # explicitly. Python has no such problem; the defect is recorded here
    # because it is the reason the C++ probe passes ``int()`` by hand.
    """

    __slots__ = ()

    def set_basket(self, basket: object) -> None:
        """Receive the owning basket.

        # C++ parity: ``DefaultLossModel::setBasket`` (defaultlossmodel.hpp:141)
        # followed by ``ConstantLossModel::resetModel``
        # (constantlosslatentmodel.hpp:159-163), which forwards the basket to
        # the ``DefaultLatentModel`` it derives from.

        # C++ parity divergence: C++ calls ``setBasket`` lazily, from
        # ``Basket::performCalculations``; the PQuantLib ``Basket`` calls it
        # eagerly from ``set_loss_model``. Same end state.
        """
        # local import: Basket imports this module, so a module-level import
        # would close a cycle. C++ has no such constraint (basket.hpp is a
        # forward declaration there).
        from pquantlib.experimental.credit.basket import Basket as _Basket  # noqa: PLC0415

        qassert.require(isinstance(basket, _Basket), "not a Basket")
        assert isinstance(basket, _Basket)
        self.reset_basket(basket)

    def expected_tranche_loss(self, d: Date) -> float:
        """Always fails.

        # C++ parity: ``ConstantLossModel`` does not override
        # ``DefaultLossModel::expectedTrancheLoss``, whose body is
        # ``QL_FAIL("expectedTrancheLoss Not implemented for this model.")``
        # (defaultlossmodel.hpp:67-69).
        """
        del d
        qassert.fail("expectedTrancheLoss Not implemented for this model.")


class ConstantLossLatentModel(DefaultProbabilityLatentModel):
    """Default-prob latent model plus a per-name constant recovery vector.

    .. warning::

       This class is **not** a port of a C++ class — note the capital ``M``,
       which the C++ name does not have. It is the one-factor convenience
       adaptation the W3-B loss models (Binomial / Recursive / Saddlepoint)
       are written against, built on
       :class:`~pquantlib.experimental.credit.default_probability_latent_model.DefaultProbabilityLatentModel`.

       The faithful port of C++ ``ConstantLossLatentmodel<copulaPolicy>`` is
       :class:`ConstantLossLatentmodel`, above.
    """

    __slots__ = ("_recoveries",)

    def __init__(
        self,
        copula: OneFactorCopula,
        recoveries: Sequence[float],
    ) -> None:
        super().__init__(copula, pool_size=len(recoveries))
        qassert.require(
            len(recoveries) > 0,
            "recoveries must be non-empty",
        )
        for i, r in enumerate(recoveries):
            qassert.require(
                0.0 <= r <= 1.0,
                f"recoveries[{i}] = {r} not in [0, 1]",
            )
        self._recoveries = list(recoveries)

    def recoveries(self) -> list[float]:
        """Return a copy of the per-name recovery vector."""
        return list(self._recoveries)

    def recovery(self, i_name: int) -> float:
        qassert.require(
            0 <= i_name < len(self._recoveries),
            f"i_name {i_name} out of range",
        )
        return self._recoveries[i_name]

    def conditional_recovery(
        self,
        i_name: int,
        m: float | None = None,  # kept for signature parity with C++
    ) -> float:
        """Constant LGD model — recovery is the stored value regardless of ``m``.

        # C++ parity: constantlosslatentmodel.hpp:72-90.
        """
        _ = m  # the constant model ignores the factor draw
        return self.recovery(i_name)

    def expected_loss(self, i_name: int, prob: float) -> float:
        """Single-name expected loss at unconditional default probability ``prob``.

        Reproduces ``prob * (1 - recoveries_[i])`` up to integration noise
        and provides a sanity check on the latent-model implementation.

        # C++ parity: defaultprobabilitylatentmodel.hpp:probOfDefault times
        # the constant LGD = 1 - recoveries_[i].
        """
        return self.prob_of_default(i_name, prob) * (
            1.0 - self.recovery(i_name)
        )
