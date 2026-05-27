"""G2SwaptionEngine structural-typing tests.

The engine delegates pricing to ``G2.swaption(arguments, fixed_rate,
range, intervals)``. The concrete G2 class lands in L4-D; until then,
these tests verify only that:

1. The ``G2ModelLike`` Protocol's structural contract is well-formed.
2. ``G2SwaptionEngine.__init__`` succeeds with any model exposing
   the required surface.

End-to-end cross-validation vs the C++ probe's
``g2_swaption_5y10y`` block lands once L4-D's ``G2`` ports — the
probe's reference value (105.46110912692544) is already in the JSON.
"""

from __future__ import annotations

from typing import cast

from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.instruments.swaption import SwaptionArguments
from pquantlib.pricingengines.swaption.g2_swaption_engine import (
    G2ModelLike,
    G2SwaptionEngine,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


class _StubG2Model:
    """Minimal stub satisfying G2ModelLike.

    Returns a pre-set sentinel value from ``swaption()`` — just enough
    to confirm the engine's delegation wiring is correct.
    """

    def __init__(self, ts: YieldTermStructureProtocol, sentinel: float) -> None:
        self._ts: YieldTermStructureProtocol = ts
        self._sentinel: float = sentinel

    @property
    def term_structure(self) -> YieldTermStructureProtocol:
        return self._ts

    def swaption(
        self,
        arguments: SwaptionArguments,
        fixed_rate: float,
        range_: float,
        intervals: int,
    ) -> float:
        del arguments, fixed_rate, range_, intervals
        return self._sentinel


def _curve() -> YieldTermStructureProtocol:
    eval_date = Date.from_ymd(17, Month.January, 2024)
    return cast(
        YieldTermStructureProtocol,
        FlatForward.from_rate(
            eval_date, 0.05, Actual360(), Compounding.Continuous, Frequency.Annual
        ),
    )


def test_stub_g2_satisfies_protocol() -> None:
    curve = _curve()
    stub = _StubG2Model(curve, sentinel=42.0)
    assert isinstance(stub, G2ModelLike)


def test_g2_swaption_engine_constructs() -> None:
    curve = _curve()
    stub = _StubG2Model(curve, sentinel=42.0)
    engine = G2SwaptionEngine(stub, range_=6.0, intervals=32)
    # Engine has the required arguments/results carriers.
    assert isinstance(engine.get_arguments(), SwaptionArguments)
