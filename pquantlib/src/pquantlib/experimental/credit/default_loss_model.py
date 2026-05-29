"""Tranche-loss-model interfaces.

# C++ parity: ql/experimental/credit/defaultlossmodel.hpp @ v1.42.1.

This module hosts BOTH the W3-B abstract base (basket-agnostic, explicit-args
interface used by the concrete loss models) AND the W3-C structural Protocol
(basket-coupled `(d: Date) -> float` interface used by the CDO engines /
Basket).

W3-B concretes inherit from :class:`DefaultLossModelBase`. W3-C consumers
parametrise against :class:`DefaultLossModel` (Protocol). A small adapter
in W3-C `Basket.expected_tranche_loss` bridges the two by reading the live
basket state and forwarding it to W3-B's explicit-args concretes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pquantlib.patterns.observer import Observable

if TYPE_CHECKING:
    from pquantlib.time.date import Date


@runtime_checkable
class DefaultLossModel(Protocol):
    """Structural Protocol for basket-coupled loss models (W3-C interface).

    Implementers expose ``set_basket(basket)`` + ``expected_tranche_loss(d)``.
    Used by ``Basket.set_loss_model`` and the CDO engines.

    The W3-B concrete models (`GaussianLHPLossModel`, `BinomialLossModel`,
    etc.) inherit from :class:`DefaultLossModelBase` and expose the richer
    basket-agnostic interface; an adapter on :class:`Basket` bridges between
    them.
    """

    def set_basket(self, basket: object) -> None:
        """Receive a back-reference to the owning ``Basket``.

        # C++ parity: ``DefaultLossModel::setBasket`` (friend access from
        # Basket::performCalculations).
        """
        ...

    def expected_tranche_loss(self, d: Date) -> float:
        """Expected tranche loss at date ``d``.

        # C++ parity: ``DefaultLossModel::expectedTrancheLoss``.
        """
        ...


class DefaultLossModelBase(Observable, ABC):
    """Abstract base for tranche-loss models.

    All five W3-B concrete models inherit from this class and override
    ``expected_tranche_loss`` + ``percentile`` (the C++ ``Basket``
    interface contract). Some subclasses also override ``prob_over_loss``
    and ``expected_shortfall`` (LHP and binomial); others raise via
    the default ``NotImplementedError`` path (matching the C++
    ``QL_FAIL`` default).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def expected_tranche_loss(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
    ) -> float:
        """Expected loss in the (attach, detach) tranche given live state.

        # C++ parity: defaultlossmodel.hpp:67 expectedTrancheLoss(Date).
        # The Python signature takes the basket state explicitly.
        """

    def prob_over_loss(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
        remaining_loss_fraction: float,
    ) -> float:
        """Probability of the tranche losing >= remaining_loss_fraction.

        Default: raises NotImplementedError. Override if a closed form
        is available (LHP), or by sampling the loss distribution.

        # C++ parity: defaultlossmodel.hpp:77 probOverLoss(Date, Real).
        """
        _ = remaining_notional, prob, average_rr, attach, detach, remaining_loss_fraction
        raise NotImplementedError(
            "prob_over_loss not implemented for this model"
        )

    def percentile(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
        perctl: float,
    ) -> float:
        """Tranche-loss percentile at ``perctl`` ∈ [0, 1].

        Default: raises NotImplementedError. Each concrete loss model
        overrides this with the model-specific quantile computation.

        # C++ parity: defaultlossmodel.hpp:81 percentile(Date, Real).
        """
        _ = remaining_notional, prob, average_rr, attach, detach, perctl
        raise NotImplementedError(
            "percentile not implemented for this model"
        )

    def expected_shortfall(
        self,
        remaining_notional: float,
        prob: float,
        average_rr: float,
        attach: float,
        detach: float,
        perctl: float,
    ) -> float:
        """Expected shortfall at the ``perctl`` percentile.

        Default: raises NotImplementedError. Override in concrete subclasses
        that can compute it (e.g. LHP via the closed-form integral).

        # C++ parity: defaultlossmodel.hpp:85 expectedShortfall(Date, Real).
        """
        _ = remaining_notional, prob, average_rr, attach, detach, perctl
        raise NotImplementedError(
            "expected_shortfall not implemented for this model"
        )
