"""DefaultLossModel — Protocol for basket loss models.

# C++ parity: ql/experimental/credit/defaultlossmodel.hpp (v1.42.1).

C++ defines an abstract ``DefaultLossModel`` as a friend of ``Basket``
that exposes a uniform interface for expected-tranche-loss queries,
probability-of-loss, percentile, expected shortfall, value-at-risk
splits, and full loss distributions. The class is the strategy
plug-in between a basket and a pricing engine (e.g.
``GaussianLHPLossModel``, ``BinomialLossModel``, ``BaseCorrelationLossModel``).

Python port: a :class:`typing.Protocol` exposing the same surface.
Concrete loss models live in cross-cluster ``W3-B`` (Gaussian LHP +
binomial + base-correlation + recursive). Structural typing means a
``W3-B`` implementer satisfies this Protocol automatically — no
inheritance or registration glue.

# C++ parity divergence: the C++ class is an Observable AND holds a
# ``RelinkableHandle<Basket>`` member that the Basket itself populates
# via ``setBasket`` (the friend trick). The Python port keeps the
# basket-reference handling on the loss model concrete (it is set in
# ``Basket.set_loss_model``) but does NOT require the Protocol to
# expose ``set_basket`` — the concrete in W3-B carries it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pquantlib.time.date import Date


@runtime_checkable
class DefaultLossModel(Protocol):
    """Structural Protocol for credit-basket loss models.

    All methods take a basket-injected reference internally (set via
    :meth:`set_basket`) and return tranche-level statistics on the
    distribution of cumulative losses at the requested date.

    Implementers MUST provide ``set_basket`` and ``expected_tranche_loss``
    at minimum. The remaining methods MAY raise NotImplementedError
    (mirroring C++ ``QL_FAIL("Not implemented for this model.")``).
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


__all__ = ["DefaultLossModel"]
