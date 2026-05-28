"""TreeSwaptionEngine — numerical lattice engine for swaptions.

# C++ parity: ql/pricingengines/swaption/treeswaptionengine.{hpp,cpp} (v1.42.1).

Generic across any short-rate model that exposes ``tree(grid) ->
ShortRateTree`` and a held term structure (or accepts one explicitly).
The C++ class is ``LatticeShortRateModelEngine`` parameterised by
``Swaption::arguments`` / ``Swaption::results``; we mirror the same
pattern with a concrete subclass of :class:`GenericEngine`.

Algorithm:

1. Resolve the reference date + day-counter from the model
   (TermStructureConsistentModel takes precedence) or the engine's
   explicit term structure.
2. Build a ``DiscretizedSwaption`` from the swaption arguments.
3. Build a ``TimeGrid`` from the swaption's mandatory times + a
   ``time_steps`` budget (extra interior points).
4. Build the model's tree on that grid.
5. Initialise the swaption at the *last* stopping time, roll back
   to the *first non-negative* exercise time, and read NPV via
   ``present_value()`` (= Arrow-Debreu weighted sum at slice 0).

Constraints:

  * ``ParYieldCurve`` cash settlement is rejected (matches C++).
  * No model = error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.methods.lattices.discretized_swaption import DiscretizedSwaption
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swaption.jamshidian_swaption_engine import (
    TermStructureConsistentModelLike,
)
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


@runtime_checkable
class TreeBuildingModelLike(Protocol):
    """Structural surface needed by the tree engines.

    The model must expose ``tree(grid) -> TreeLattice1D``.  All
    1-factor short-rate models in L4-B/L5-B (Vasicek / HullWhite /
    CIR / ExtendedCIR / BlackKarasinski) satisfy this Protocol.
    """

    def tree(self, grid: TimeGrid) -> TreeLattice1D: ...


class TreeSwaptionEngine(GenericEngine[SwaptionArguments, SwaptionResults]):
    """Numerical-lattice swaption engine.

    # C++ parity: ``class TreeSwaptionEngine``
    # (treeswaptionengine.hpp:44-67, treeswaptionengine.cpp:25-96).
    """

    def __init__(
        self,
        model: TreeBuildingModelLike,
        time_steps: int,
        term_structure: YieldTermStructureProtocol | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._model: TreeBuildingModelLike = model
        self._time_steps: int = int(time_steps)
        self._term_structure: YieldTermStructureProtocol | None = term_structure
        self._day_counter: DayCounter | None = day_counter

    def calculate(self) -> None:
        # # C++ parity: treeswaptionengine.cpp:51-95.
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.settlement_method != SettlementMethod.ParYieldCurve,
            "cash settled (ParYieldCurve) swaptions not priced with TreeSwaptionEngine",
        )

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

        # Build the discretized swaption.
        swaption = DiscretizedSwaption(args, ref_date, dc)

        # Build the time grid from the swaption's mandatory times.
        times = swaption.mandatory_times()
        time_grid = TimeGrid.with_mandatory_and_steps(times, self._time_steps)

        # Build the lattice from the model.
        lattice = self._model.tree(time_grid)

        # Stopping times for the exercise.
        assert args.exercise is not None
        stopping_times = [
            dc.year_fraction(ref_date, d) for d in args.exercise.dates()
        ]

        swaption.initialize(lattice, stopping_times[-1])

        # Roll back to the *first non-negative* exercise time.
        next_exercise = next(t for t in stopping_times if t >= 0.0)
        swaption.rollback(next_exercise)

        results.value = swaption.present_value()


__all__ = ["TreeBuildingModelLike", "TreeSwaptionEngine"]
