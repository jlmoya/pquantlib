"""CompositeInstrument — a weighted sum of other instruments.

# C++ parity: ql/instruments/compositeinstrument.hpp + .cpp (v1.43).

Its NPV is ``sum(multiplier_i * component_i.npv())``. ``subtract`` is
defined as ``add`` with the negated multiplier, so the sign convention
lives in exactly one place, as in C++.

``isExpired()`` is an **AND** over the components — a composite is expired
only once every component is. An empty composite is therefore expired
(vacuous AND) and worth zero, which C++ relies on and this port reproduces.

C++ ``CompositeInstrument::add`` additionally calls
``instrument->alwaysForwardNotifications()``. That call exists because a C++
``LazyObject`` which has never been recalculated stops forwarding
notifications, so an expired component would go silent and never wake up
again. **This port's** ``LazyObject.update`` calls ``notify_observers()``
unconditionally (patterns/lazy_object.py) — i.e. it already behaves as C++
does *after* ``alwaysForwardNotifications()``. There is nothing to turn on,
so the call has no counterpart here; the behaviour it buys is the default.
"""

from __future__ import annotations

from pquantlib.instruments.instrument import Instrument


class CompositeInstrument(Instrument):
    """Aggregate of other instruments, each with a multiplier.

    Warning (from C++): methods that drive the calculation directly, such as
    ``recalculate()`` / ``freeze()``, might not work correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._components: list[tuple[Instrument, float]] = []

    def add(self, instrument: Instrument, multiplier: float = 1.0) -> None:
        """Add an instrument to the composite.

        C++ parity: compositeinstrument.cpp:23-38.
        """
        # C++ parity: compositeinstrument.cpp:24 guards against a null
        # shared_ptr. ``instrument`` is a non-optional ``Instrument`` here,
        # so that branch is unreachable and is deliberately not reproduced.
        self._components.append((instrument, multiplier))
        instrument.register_with(self)
        self.update()

    def subtract(self, instrument: Instrument, multiplier: float = 1.0) -> None:
        """Short an instrument from the composite.

        C++ parity: compositeinstrument.cpp:40-43 — ``add(instrument,
        -multiplier)``. The negation lives here and nowhere else.
        """
        self.add(instrument, -multiplier)

    def components(self) -> list[tuple[Instrument, float]]:
        """The (instrument, multiplier) pairs, in insertion order.

        Not in C++, whose ``components_`` is private; exposed here so tests
        can assert that a multiplier reached the slot it configures.
        """
        return list(self._components)

    def is_expired(self) -> bool:
        # C++ parity: compositeinstrument.cpp:45-51 — AND, not OR.
        return all(instrument.is_expired() for instrument, _ in self._components)

    def deep_update(self) -> None:
        # C++ parity: compositeinstrument.cpp:60-65.
        for instrument, _ in self._components:
            instrument.deep_update()
        self.update()

    def _perform_calculations(self) -> None:
        # C++ parity: compositeinstrument.cpp:53-58.
        self._npv = sum(
            (multiplier * instrument.npv() for instrument, multiplier in self._components),
            0.0,
        )
        self._error_estimate = 0.0
        self._additional_results = {}


__all__ = ["CompositeInstrument"]
