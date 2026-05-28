"""TreeCapFloorEngine — numerical lattice engine for caps/floors/collars.

# C++ parity: ql/pricingengines/capfloor/treecapfloorengine.{hpp,cpp} (v1.42.1).

Same scaffolding as :class:`TreeSwaptionEngine` but driving a
:class:`DiscretizedCapFloor` underlying:

1. Resolve reference date + day-counter via the model or explicit TS.
2. Build a ``DiscretizedCapFloor`` from the cap/floor arguments.
3. Build the time grid from the cap/floor's mandatory times.
4. Build the model's tree.
5. Initialise at the *last* end date, roll back to the *first* start
   date, and read NPV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.instruments.cap_floor import CapFloorArguments, CapFloorResults
from pquantlib.methods.lattices.discretized_cap_floor import DiscretizedCapFloor
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swaption.jamshidian_swaption_engine import (
    TermStructureConsistentModelLike,
)
from pquantlib.pricingengines.swaption.tree_swaption_engine import (
    TreeBuildingModelLike,
)
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class TreeCapFloorEngine(GenericEngine[CapFloorArguments, CapFloorResults]):
    """Numerical-lattice cap/floor engine.

    # C++ parity: ``class TreeCapFloorEngine``
    # (treecapfloorengine.{hpp,cpp}, v1.42.1).
    """

    def __init__(
        self,
        model: TreeBuildingModelLike,
        time_steps: int,
        term_structure: YieldTermStructureProtocol | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(CapFloorArguments(), CapFloorResults())
        self._model: TreeBuildingModelLike = model
        self._time_steps: int = int(time_steps)
        self._term_structure: YieldTermStructureProtocol | None = term_structure
        self._day_counter: DayCounter | None = day_counter

    def calculate(self) -> None:
        # # C++ parity: treecapfloorengine.cpp:43-79.
        args = self._arguments
        results = self._results
        results.reset()

        # Resolve reference date + day-counter — model takes precedence.
        ref_date = None
        dc: DayCounter | None = None
        if isinstance(self._model, TermStructureConsistentModelLike):
            ts = self._model.term_structure
            ref_date = ts.reference_date()
            dc = ts.day_counter()
        if ref_date is None:
            qassert.require(
                self._term_structure is not None,
                "no term_structure available; pass one explicitly or use a TermStructureConsistentModel",
            )
            assert self._term_structure is not None
            ref_date = self._term_structure.reference_date()
            dc = self._term_structure.day_counter()
        if self._day_counter is not None:
            dc = self._day_counter
        assert dc is not None
        assert ref_date is not None

        # Build the discretized cap/floor.
        capfloor = DiscretizedCapFloor(args, ref_date, dc)
        times = capfloor.mandatory_times()
        time_grid = TimeGrid.with_mandatory_and_steps(times, self._time_steps)

        # Build the lattice.
        lattice = self._model.tree(time_grid)

        first_time = dc.year_fraction(ref_date, args.start_dates[0])
        last_time = dc.year_fraction(ref_date, args.end_dates[-1])

        capfloor.initialize(lattice, last_time)
        capfloor.rollback(first_time)

        results.value = capfloor.present_value()


__all__ = ["TreeCapFloorEngine"]
