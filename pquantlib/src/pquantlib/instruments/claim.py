"""Claim — default-event claim hierarchy.

# C++ parity: ql/instruments/claim.{hpp,cpp} (v1.42.1).

A Claim computes the payoff at default time given the notional and
recovery rate. The default convention (``FaceValueClaim``) is
``notional * (1 - recovery)``; other claim conventions (e.g.
``FaceValueAccrualClaim`` for bonds) are deferred.

C++ ``Claim`` is both Observable and Observer; we mirror Observable
behaviour but the Observer registration is implicit (Python observables
auto-notify their observers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.patterns.observer import Observable
from pquantlib.time.date import Date


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


__all__ = ["Claim", "FaceValueClaim"]
