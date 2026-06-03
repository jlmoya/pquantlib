"""SoftCallability — callability that may trigger conversion.

# C++ parity: ql/instruments/bonds/convertiblebonds.hpp:42-50 (v1.42.1).

A :class:`SoftCallability` is a :class:`~pquantlib.instruments.callability.Callability`
of type ``Call`` carrying an extra ``trigger`` — the multiple of the
conversion value above which the (soft) call may be exercised. Used by the
convertible-bond Tsiveriotis-Fernandes lattice (``DiscretizedConvertible``)
to gate the call on the underlying breaching ``trigger * conversionValue``.

The C++ class lives in ``convertiblebonds.hpp`` rather than
``callabilityschedule.hpp``; the Python port keeps it in its own module so
the convertible cluster can import it without dragging in the whole bond
instrument, but it subclasses the existing core ``Callability`` (no
duplicate Callability type).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.instruments.callability import Callability, CallabilityType

if TYPE_CHECKING:
    from pquantlib.instruments.bond import BondPrice
    from pquantlib.time.date import Date


class SoftCallability(Callability):
    """A soft-call right (always ``Call``) with a conversion trigger.

    # C++ parity: ``class SoftCallability : public Callability``
    # (convertiblebonds.hpp:42-50). The C++ ctor fixes the type to
    # ``Callability::Call``; we mirror that.
    """

    def __init__(self, price: BondPrice, date: Date, trigger: float) -> None:
        super().__init__(price, CallabilityType.Call, date)
        self._trigger: float = trigger

    def trigger(self) -> float:
        """C++ parity: ``SoftCallability::trigger`` (convertiblebonds.hpp:46)."""
        return self._trigger


__all__ = ["SoftCallability"]
