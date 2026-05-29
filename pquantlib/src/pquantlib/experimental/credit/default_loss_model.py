"""DefaultLossModel — abstract base for tranche-loss models.

# C++ parity: ql/experimental/credit/defaultlossmodel.hpp @ v1.42.1.

The C++ class is an Observable abstract base coupled to ``Basket`` via
a friend relationship. Each subclass exposes the same protected
interface (``expectedTrancheLoss(d)``, ``probOverLoss(d, loss_fraction)``,
``percentile(d, perctl)``, ``expectedShortfall(d, perctl)``, etc.) and
the Basket invokes them after setting itself as the "current basket".

The Python port replaces the basket coupling with a basket-agnostic
interface: each call takes the live (remaining) basket state directly
as arguments. The downstream W3-C basket layer will wrap the loss
models and forward live state to them as needed.

# C++ parity divergence: the C++ method takes only ``const Date& d`` and
# pulls every other input from ``basket_->...``; the Python method takes
# all inputs as explicit args. The basket integration lives in W3-C.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.patterns.observer import Observable


class DefaultLossModel(Observable, ABC):
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
