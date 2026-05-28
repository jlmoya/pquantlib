"""Digital (binary) option — convenience wrapper over CashOrNothing /
AssetOrNothing payoffs from L3-A.

# C++ parity: there is no separate ``DigitalOption`` class in C++
# QuantLib v1.42.1 — binary options are constructed by combining a
# ``CashOrNothingPayoff`` / ``AssetOrNothingPayoff`` with a vanilla
# ``EuropeanOption``. The Python port follows the same idiom; this
# module exports a small ``DigitalOption`` alias for documentation
# purposes only.

The L5-E carve-out documents that the engine side (analytic binary
European) is already covered by ``AnalyticEuropeanEngine`` from L3-D
because the payoff dispatch in ``BlackCalculator`` routes Cash/Asset-
OrNothing correctly. For American-style binary touches see the
``AnalyticBinaryBarrierEngine`` (one-touch / no-touch).
"""

from __future__ import annotations

from pquantlib.exercise import Exercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
)


class DigitalOption(VanillaOption):
    """Digital option = VanillaOption with a Cash/Asset-OrNothing payoff.

    # C++ parity: ``DigitalOption`` is not a separate C++ class. This
    # alias exists so user code that wants to ``isinstance``-discriminate
    # a binary option from a plain vanilla can do so cleanly.

    Use ``CashOrNothingPayoff`` or ``AssetOrNothingPayoff`` to construct
    a digital. The engine choice is the same as for a vanilla
    European (``AnalyticEuropeanEngine``).
    """

    def __init__(
        self,
        payoff: CashOrNothingPayoff | AssetOrNothingPayoff,
        exercise: Exercise,
    ) -> None:
        super().__init__(payoff, exercise)


__all__ = ["DigitalOption"]
