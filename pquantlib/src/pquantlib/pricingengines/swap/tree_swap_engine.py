"""TreeVanillaSwapEngine — numerical lattice engine for vanilla swaps.

# C++ parity: ql/pricingengines/swap/treeswapengine.{hpp,cpp} (v1.43),
#             ``class TreeVanillaSwapEngine
#               : public LatticeShortRateModelEngine<VanillaSwap::arguments,
#                                                    VanillaSwap::results>``.

The only concrete engine in the library that derives straight from
:class:`~pquantlib.pricingengines.lattice_short_rate_model_engine.LatticeShortRateModelEngine`,
which is why it is the natural exercise for that base class and for
:class:`~pquantlib.pricingengines.generic_model_engine.GenericModelEngine`
underneath it.

``calculate()`` (treeswapengine.cpp:42-75):

1. resolve the reference date + day counter from the model if it is a
   ``TermStructureConsistentModel``, otherwise from the engine's own curve;
2. build a ``DiscretizedSwap`` and take its mandatory times;
3. use the base class's pre-built lattice **if there is one** (the ``TimeGrid``
   constructor), otherwise build ``TimeGrid(times, time_steps)`` and a fresh
   lattice from the model;
4. initialise at ``max(times)``, roll back to 0, read the present value.

It assigns **only** ``results.value``. ``fixed_leg_npv`` / ``floating_leg_npv``
/ ``fair_rate`` / ``fair_spread`` are left unset and therefore raise — pinned
explicitly in the cross-validation so a port does not invent them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.instruments.fixed_vs_floating_swap import (
    FixedVsFloatingSwapArguments,
    FixedVsFloatingSwapResults,
)
from pquantlib.methods.lattices.discretized_swap import DiscretizedSwap
from pquantlib.pricingengines.lattice_short_rate_model_engine import (
    LatticeShortRateModelEngine,
)
from pquantlib.pricingengines.swaption.jamshidian_swaption_engine import (
    TermStructureConsistentModelLike,
)
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.models.model import ShortRateModel
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.date import Date


class TreeVanillaSwapEngine(
    LatticeShortRateModelEngine[FixedVsFloatingSwapArguments, FixedVsFloatingSwapResults]
):
    """Numerical-lattice engine for a ``VanillaSwap``.

    # C++ parity: ``class TreeVanillaSwapEngine`` (treeswapengine.hpp:38-59).
    """

    def __init__(
        self,
        model: ShortRateModel,
        time_steps_or_grid: int | TimeGrid,
        term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        """C++ parity: the two ctors at treeswapengine.cpp:26-40.

        ``term_structure`` is C++'s ``Handle<YieldTermStructure>`` and is only
        needed when the model cannot supply one itself; an empty handle is
        ``None``.
        """
        super().__init__(
            FixedVsFloatingSwapArguments(),
            FixedVsFloatingSwapResults(),
            model,
            time_steps_or_grid,
        )
        self._term_structure: YieldTermStructureProtocol | None = term_structure
        # C++ ``registerWith(termStructure_)``. The protocol does not declare
        # ``register_with``, so probe for it the way DiscountingSwapEngine does.
        register = getattr(term_structure, "register_with", None)
        if register is not None:
            register(self)

    def term_structure(self) -> YieldTermStructureProtocol | None:
        return self._term_structure

    def calculate(self) -> None:
        # C++ parity: treeswapengine.cpp:42-75.
        model = self.model()
        qassert.require(model is not None, "no model specified")
        assert model is not None

        results = self._results
        results.reset()

        reference_date: Date | None = None
        day_counter: DayCounter | None = None
        if isinstance(model, TermStructureConsistentModelLike):
            ts = model.term_structure
            reference_date = ts.reference_date()
            day_counter = ts.day_counter()
        else:
            qassert.require(
                self._term_structure is not None,
                "no term structure available; the model is not term-structure "
                "consistent, so one must be passed to the engine",
            )
            assert self._term_structure is not None
            reference_date = self._term_structure.reference_date()
            day_counter = self._term_structure.day_counter()

        swap = DiscretizedSwap(self._arguments, reference_date, day_counter)
        times = list(swap.mandatory_times())

        lattice = self.lattice()
        if lattice is None:
            # C++ ``TimeGrid timeGrid(times.begin(), times.end(), timeSteps_);``
            lattice = self._build_lattice(
                TimeGrid.with_mandatory_and_steps(times, self.time_steps())
            )

        max_time = max(times)
        swap.initialize(lattice, max_time)
        swap.rollback(0.0)

        # C++ assigns results_.value ONLY; every other VanillaSwap result stays
        # Null and its accessor throws.
        results.value = swap.present_value()


__all__ = ["TreeVanillaSwapEngine"]
