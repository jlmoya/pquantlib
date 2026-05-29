"""Loss — pair of (time, amount) sortable by time.

# C++ parity: ql/experimental/credit/loss.hpp (v1.42.1).

Used by random-default-model code in downstream W3-B/C to record
realised losses on a Monte-Carlo simulation.

# C++ parity divergence: the C++ ``Loss`` overloads ``==``/``!=`` to
# compare only on time (intentional — losses at the same time collapse
# in sorted containers). The Python port keeps the same semantics,
# explicitly documented, and adds ``__hash__`` based on time for
# set/dict membership.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Loss:
    """A realised loss occurring at ``time`` of magnitude ``amount``.

    Optional ``name`` carries the issuer identifier (extension over C++
    which only stores time + amount — used by Python callers that want
    to map losses back to their source issuer in a basket).
    """

    amount: float = 0.0
    time: float = 0.0
    name: str | None = None

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Loss):
            return NotImplemented
        return self.time < other.time

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Loss):
            return NotImplemented
        return self.time > other.time

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Loss):
            return NotImplemented
        # # C++ parity: loss.hpp:43-45 — equality on time only.
        return self.time == other.time

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.time)
