"""Replication — Sub / Central / Super digital-option replication strategy.

# C++ parity: ql/cashflows/replication.{hpp,cpp} (v1.42.1, 099987f0).

Specifies how the embedded digital option in a :class:`DigitalCoupon
<pquantlib.cashflows.digital_coupon.DigitalCoupon>` is replicated by a
call/put spread: ``Sub`` (under-replication), ``Central`` (centred), or
``Super`` (over-replication). ``DigitalReplication`` bundles the strategy with
the spread ``gap``.

# C++ parity divergence: the C++ ``struct Replication { enum Type {...} }`` +
# ``operator<<`` becomes a Python ``IntEnum`` (matching the project's
# enum idioms, e.g. ``PositionType`` / ``OptionType``).
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert


class Replication(IntEnum):
    """Digital-option replication strategy.

    # C++ parity: ``Replication::Type`` — integer values match the C++ enum
    # order: Sub=0, Central=1, Super=2.
    """

    Sub = 0
    Central = 1
    Super = 2

    def __str__(self) -> str:
        return {Replication.Sub: "Sub", Replication.Central: "Central", Replication.Super: "Super"}[
            self
        ]


class DigitalReplication:
    """Replication-strategy + spread-gap specification.

    # C++ parity: ql/cashflows/replication.hpp:44-53.
    """

    def __init__(
        self,
        replication_type: Replication = Replication.Central,
        gap: float = 1e-4,
    ) -> None:
        qassert.require(gap > 0.0, "Non positive epsilon not allowed")
        self._replication_type: Replication = replication_type
        self._gap: float = gap

    def replication_type(self) -> Replication:
        """# C++ parity: ``DigitalReplication::replicationType``."""
        return self._replication_type

    def gap(self) -> float:
        """# C++ parity: ``DigitalReplication::gap``."""
        return self._gap


__all__ = ["DigitalReplication", "Replication"]
