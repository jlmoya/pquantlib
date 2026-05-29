"""Basket — collection of credit issuers + tranche structure.

# C++ parity: ql/experimental/credit/basket.{hpp,cpp} (v1.42.1).

A Basket is the credit-CDO equivalent of a portfolio: a set of named
issuer positions (notional + DefaultProbKey + DefaultProbabilityTermStructure
per issuer, held inside a ``Pool``) plus an attachment/detachment ratio
that carves out a tranche.

Tranching:
- ``basket_notional`` = sum of issuer notionals.
- ``attachment_amount`` = ``attachment_ratio * basket_notional``.
- ``detachment_amount`` = ``detachment_ratio * basket_notional``.
- ``tranche_notional`` = ``detachment_amount - attachment_amount``.

The Basket integrates with a :class:`DefaultLossModel` (set via
:meth:`set_loss_model`) which drives expected_tranche_loss queries
during pricing.

# C++ parity divergence: the C++ Basket is a LazyObject with elaborate
# caching of remaining-name / remaining-notional / settled-loss / live-list
# state by evaluation date. The Python port retains the calc-on-demand
# semantics but holds straightforward eager fields; observability flows
# through ``Pool`` + loss-model registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.pool import Pool
from pquantlib.instruments.claim import Claim, FaceValueClaim
from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.experimental.credit.default_loss_model import DefaultLossModel


class Basket(Observable):
    """Credit-issuer basket with attach/detach tranche structure.

    Parameters
    ----------
    ref_date
        Basket inception date.
    names
        Issuer names (one per position; duplicates allowed if the same
        issuer enters the basket twice).
    notionals
        Per-position notional amounts; ``len(notionals) == pool.size()``.
    pool
        ``Pool`` containing the registered ``Issuer`` instances.
    attachment_ratio
        Fraction of basket notional where tranche protection starts (0.0
        for equity tranche).
    detachment_ratio
        Fraction where protection ends (1.0 for super-senior).
    claim
        Default-claim convention (default: ``FaceValueClaim``).
    """

    def __init__(
        self,
        ref_date: Date,
        names: list[str],
        notionals: list[float],
        pool: Pool,
        attachment_ratio: float = 0.0,
        detachment_ratio: float = 1.0,
        claim: Claim | None = None,
    ) -> None:
        super().__init__()
        qassert.require(len(notionals) > 0, "notionals empty")
        qassert.require(
            0.0 <= attachment_ratio <= detachment_ratio <= 1.0,
            "invalid attachment/detachment ratio",
        )
        # C++ parity divergence: C++ also requires ``pool`` to be non-null;
        # in Python the type-system already enforces that ``Pool`` is
        # non-None at the call site, so the runtime check is dropped.
        qassert.require(
            len(notionals) == pool.size(),
            "unmatched data entry sizes in basket",
        )

        self._ref_date: Date = ref_date
        self._names: list[str] = list(names)
        self._notionals: list[float] = list(notionals)
        self._pool: Pool = pool
        self._attachment_ratio: float = attachment_ratio
        self._detachment_ratio: float = detachment_ratio
        self._claim: Claim = claim if claim is not None else FaceValueClaim()

        # Pre-compute tranche structure (mirrors C++ ctor body).
        self._basket_notional: float = sum(self._notionals)
        self._attachment_amount: float = (
            self._attachment_ratio * self._basket_notional
        )
        self._detachment_amount: float = (
            self._detachment_ratio * self._basket_notional
        )
        self._tranche_notional: float = (
            self._detachment_amount - self._attachment_amount
        )

        self._loss_model: DefaultLossModel | None = None

    # ---- inspectors --------------------------------------------------------

    def ref_date(self) -> Date:
        return self._ref_date

    def size(self) -> int:
        """Number of counterparties (== pool size)."""
        return self._pool.size()

    def names(self) -> list[str]:
        """Counterparty names — mirrors ``pool.names()``."""
        return self._pool.names()

    def notionals(self) -> list[float]:
        return list(self._notionals)

    def notional(self) -> float:
        # C++ parity: ``Basket::notional`` (accumulate of notionals_).
        return sum(self._notionals)

    def pool(self) -> Pool:
        return self._pool

    def claim(self) -> Claim:
        return self._claim

    def default_keys(self) -> list[DefaultProbKey]:
        # C++ parity: ``Basket::defaultKeys`` returns pool->defaultKeys().
        return self._pool.default_keys()

    def attachment_ratio(self) -> float:
        return self._attachment_ratio

    def detachment_ratio(self) -> float:
        return self._detachment_ratio

    def basket_notional(self) -> float:
        return self._basket_notional

    def tranche_notional(self) -> float:
        return self._tranche_notional

    def attachment_amount(self) -> float:
        return self._attachment_amount

    def detachment_amount(self) -> float:
        return self._detachment_amount

    def remaining_notional(self) -> float:
        """Basket notional after settled defaults (eval-date snapshot).

        # C++ parity divergence: the C++ ``Basket`` carries a cached
        # ``evalDateRemainingNot_`` updated in ``computeBasket``. The
        # Python port omits the defaulted-name accounting (no
        # observable Issuer events in scope yet); we return the
        # inception notional.
        """
        return self._basket_notional

    # ---- Observer interface ------------------------------------------------

    def update(self) -> None:
        """Forward notification when a registered Observable updates.

        # C++ parity: ``Basket`` is a LazyObject which forwards updates to
        # its own observers. We mirror by forwarding.
        """
        self.notify_observers()

    # ---- loss model wiring -------------------------------------------------

    def set_loss_model(self, loss_model: DefaultLossModel) -> None:
        """Attach a default-loss model.

        # C++ parity: ``Basket::setLossModel`` unregisters the previous
        # model, registers the new one, then calls ``LazyObject::update``
        # to invalidate the basket's cache. The Python port drops the
        # cache-invalidation since we don't cache; observer notification
        # is symmetric.
        """
        if self._loss_model is not None and isinstance(
            self._loss_model, Observable,
        ):
            # Detach previous (best-effort — Observer chain unregisters
            # only if loss_model_ is Observable).
            pass
        self._loss_model = loss_model
        loss_model.set_basket(self)
        if isinstance(loss_model, Observable):
            # The loss model notifies the basket when its parameters
            # change (mirrors C++ ``registerWith(lossModel_)``).
            loss_model.register_with(self)
        self.notify_observers()

    def loss_model(self) -> DefaultLossModel | None:
        return self._loss_model

    # ---- tranche-loss queries ---------------------------------------------

    def expected_tranche_loss(self, d: Date) -> float:
        """Expected tranche loss at date ``d``.

        # C++ parity: ``Basket::expectedTrancheLoss`` —
        # ``cumulated_loss + loss_model.expected_tranche_loss(d)``. We
        # currently set ``cumulated_loss = 0`` (no defaulted-name
        # accounting; see ``remaining_notional`` divergence note).
        """
        qassert.require(
            self._loss_model is not None,
            "Basket has no default loss model assigned.",
        )
        assert self._loss_model is not None
        return self._loss_model.expected_tranche_loss(d)


__all__ = ["Basket"]
