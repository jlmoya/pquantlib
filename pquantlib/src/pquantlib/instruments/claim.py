"""Claim — default-event claim hierarchy.

# C++ parity: ql/instruments/claim.{hpp,cpp} (v1.42.1).

A Claim computes the payoff at default time given the notional and
recovery rate. The default convention (``FaceValueClaim``) is
``notional * (1 - recovery)``; other claim conventions (e.g.
``FaceValueAccrualClaim`` for bonds) follow below.

C++ ``Claim`` is both Observable and Observer; we mirror Observable
behaviour but the Observer registration is implicit (Python observables
auto-notify their observers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.instruments.bond import Bond


class Claim(Observable, ABC):
    """Abstract base for default-event claim conventions.

    Subclasses implement :meth:`amount` to compute the payoff at a
    default date given the notional and recovery rate.

    # C++ parity: ql/instruments/claim.hpp — base class is Observable +
    # Observer. We inherit only Observable; Observer ``update`` is a
    # no-op for the standalone Claim hierarchy and concrete
    # ``FaceValueAccrualClaim`` (deferred) would re-add it.
    """

    @abstractmethod
    def amount(
        self,
        default_date: Date,
        notional: float,
        recovery_rate: float,
    ) -> float:
        """Payoff at default time.

        # C++ parity: ``Claim::amount(default_date, notional, recovery_rate)``.
        """

    def update(self) -> None:
        """C++ parity: ``Claim::update`` forwards notification."""
        self.notify_observers()


class FaceValueClaim(Claim):
    """Standard claim: ``notional * (1 - recovery)``.

    # C++ parity: ql/instruments/claim.cpp:24-28.
    """

    def amount(
        self,
        default_date: Date,
        notional: float,
        recovery_rate: float,
    ) -> float:
        del default_date  # face-value claim does not depend on default date.
        return notional * (1.0 - recovery_rate)


class FaceValueAccrualClaim(Claim):
    """Claim on the notional of a reference security, including accrual.

    # C++ parity: ql/instruments/claim.hpp:51-59, claim.cpp:32-46.

    ``notional * (1 - recovery - accrual)`` where ``accrual`` is the
    reference bond's accrued amount **per unit of its notional at that
    date** — so for an amortising reference security the divisor moves over
    the bond's life, and using the initial face amount instead would be
    wrong by the amortisation factor.
    """

    def __init__(self, reference_security: Bond) -> None:
        super().__init__()
        self._reference_security: Bond = reference_security
        # C++ ``registerWith(referenceSecurity)``.
        reference_security.register_with(self)

    def reference_security(self) -> Bond:
        return self._reference_security

    def amount(
        self,
        default_date: Date,
        notional: float,
        recovery_rate: float,
    ) -> float:
        accrual = self._reference_security.accrued_amount(
            default_date
        ) / self._reference_security.notional(default_date)
        return notional * (1.0 - recovery_rate - accrual)


__all__ = ["Claim", "FaceValueAccrualClaim", "FaceValueClaim"]
