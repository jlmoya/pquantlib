"""LatticeShortRateModelEngine — short-rate-model engine specialised on a lattice.

# C++ parity: ql/pricingengines/latticeshortratemodelengine.hpp (v1.43),
#             ``template <class Arguments, class Results>
#              class LatticeShortRateModelEngine
#                : public GenericModelEngine<ShortRateModel, Arguments, Results>``.

Like :class:`~pquantlib.pricingengines.generic_model_engine.GenericModelEngine`
this is a real base class, not a tag. It owns three things and they are what
distinguish its two constructors:

``LatticeShortRateModelEngine(model, time_steps)``
    stores ``time_steps`` and leaves ``lattice`` **unbuilt**. The derived
    engine's ``calculate()`` is expected to build a ``TimeGrid`` from the
    instrument's own mandatory times plus that step budget, and to build a
    lattice from it on every call.

``LatticeShortRateModelEngine(model, time_grid)``
    stores the grid, sets ``time_steps`` to 0, and builds the lattice
    **eagerly** in the constructor. ``update()`` then rebuilds it from the same
    grid whenever the model notifies. The derived engine's ``calculate()`` must
    use the pre-built lattice as-is — which also means the grid has to contain
    the instrument's mandatory times, or ``TimeGrid.index`` will reject it.

C++ overloads on the second parameter's type; Python takes a union and
dispatches on it, which is the same contract with one fewer entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pquantlib import qassert
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from collections.abc import Callable

    from pquantlib.methods.lattices.tree_lattice_1d import TreeLattice1D
    from pquantlib.models.model import ShortRateModel


class LatticeShortRateModelEngine[
    ArgsT: PricingEngineArguments,
    ResultsT: PricingEngineResults,
](GenericModelEngine["ShortRateModel", ArgsT, ResultsT]):
    """Base for engines that price on a short-rate lattice.

    # C++ parity: ``LatticeShortRateModelEngine``
    # (latticeshortratemodelengine.hpp:36-99).
    """

    def __init__(
        self,
        arguments: ArgsT,
        results: ResultsT,
        model: ShortRateModel,
        time_steps_or_grid: int | TimeGrid,
    ) -> None:
        super().__init__(arguments, results, model)
        self._time_grid: TimeGrid | None = None
        self._time_steps: int = 0
        self._lattice: TreeLattice1D | None = None
        if isinstance(time_steps_or_grid, TimeGrid):
            # C++ ctor #3 (hpp:78-86): timeGrid_(timeGrid), timeSteps_(0),
            # lattice_ = model_->tree(timeGrid).
            self._time_grid = time_steps_or_grid
            self._lattice = self._build_lattice(time_steps_or_grid)
        else:
            # C++ ctors #1/#2 (hpp:57-76): timeSteps_(timeSteps) + QL_REQUIRE.
            qassert.require(
                time_steps_or_grid > 0,
                f"timeSteps must be positive, {time_steps_or_grid} not allowed",
            )
            self._time_steps = int(time_steps_or_grid)

    def _build_lattice(self, grid: TimeGrid) -> TreeLattice1D:
        model = self.model()
        qassert.require(model is not None, "no model specified")
        assert model is not None
        # C++ ``ShortRateModel`` declares ``virtual ext::shared_ptr<Lattice>
        # tree(const TimeGrid&) const = 0`` (model.hpp:144). The Python
        # ``pquantlib.models.model.ShortRateModel`` documents that abstract but
        # does not declare it — ``OneFactorModel`` introduces it instead — so
        # the attribute is invisible to the type checker here. Every concrete
        # one-factor model returns a ``TreeLattice1D``.
        tree = cast("Callable[[TimeGrid], TreeLattice1D]", model.tree)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        return tree(grid)

    # --- inspectors ---------------------------------------------------------

    def time_steps(self) -> int:
        """C++ ``timeSteps_`` — 0 when built from an explicit ``TimeGrid``."""
        return self._time_steps

    def time_grid(self) -> TimeGrid | None:
        """C++ ``timeGrid_`` — empty (``None`` here) in the ``timeSteps`` flavour."""
        return self._time_grid

    def lattice(self) -> TreeLattice1D | None:
        """C++ ``lattice_`` — null in the ``timeSteps`` flavour until built."""
        return self._lattice

    # --- Observer -----------------------------------------------------------

    def update(self) -> None:
        """Rebuild the lattice from the stored grid, then notify.

        C++ parity: latticeshortratemodelengine.hpp:88-95 — the rebuild only
        happens when ``timeGrid_`` is non-empty, i.e. only for the ``TimeGrid``
        constructor.
        """
        if self._time_grid is not None and not self._time_grid.empty():
            self._lattice = self._build_lattice(self._time_grid)
        super().update()


__all__ = ["LatticeShortRateModelEngine"]
