"""TreeCallableFixedRateBondEngine — short-rate lattice engine for callable bonds.

# C++ parity: ql/experimental/callablebonds/treecallablebondengine.{hpp,cpp}
#             (v1.42.1).

Prices a ``CallableBond`` on a short-rate tree (HullWhite /
BlackKarasinski / any one-factor model). Mirrors the
``TreeSwaptionEngine`` / ``TreeCapFloorEngine`` scaffolding:

1. Resolve the discount curve via the model (if it is a
   ``TermStructureConsistentModel``) or the explicitly-passed curve.
2. Build a ``DiscretizedCallableFixedRateBond`` from the arguments.
3. Build the tree from the bond's mandatory times + the step count.
4. If a non-zero continuous ``spread`` is set on the arguments, push it
   onto the tree's short-rate (one-factor trees only — matching C++).
5. Initialise at the redemption time, roll back to 0, read present value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pquantlib import qassert
from pquantlib.instruments.callable_bond import (
    CallableBondArguments,
    CallableBondResults,
)
from pquantlib.methods.lattices.discretized_callable_fixed_rate_bond import (
    DiscretizedCallableFixedRateBond,
)
from pquantlib.models.shortrate.short_rate_tree import ShortRateTree
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swaption.jamshidian_swaption_engine import (
    TermStructureConsistentModelLike,
)
from pquantlib.pricingengines.swaption.tree_swaption_engine import TreeBuildingModelLike
from pquantlib.time.time_grid import TimeGrid

if TYPE_CHECKING:
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class TreeCallableFixedRateBondEngine(
    GenericEngine[CallableBondArguments, CallableBondResults]
):
    """Numerical-lattice callable fixed-rate bond engine.

    # C++ parity: ``class TreeCallableFixedRateBondEngine``
    # (treecallablebondengine.{hpp,cpp}, v1.42.1).
    """

    def __init__(
        self,
        model: TreeBuildingModelLike,
        time_steps: int,
        term_structure: YieldTermStructure | None = None,
    ) -> None:
        super().__init__(CallableBondArguments(), CallableBondResults())
        self._model: TreeBuildingModelLike = model
        self._time_steps: int = int(time_steps)
        self._term_structure: YieldTermStructure | None = term_structure
        # Pre-built lattice (C++ ``lattice_``) — unused in this port; the
        # tree is always built from the time grid.
        self._lattice = None

    def calculate(self) -> None:
        # C++ parity: treecallablebondengine.cpp:46-48.
        self._calculate_with_spread(self._arguments.spread)

    def _calculate_with_spread(self, spread: float) -> None:
        # C++ parity: treecallablebondengine.cpp:50-87.
        results = self._results
        results.reset()

        # Resolve the discount curve — the model takes precedence. The
        # model's term structure is concretely a YieldTermStructure (the
        # protocol return type is structural); the engine needs the richer
        # zero_rate accessor for the spread-adjusted discount in the
        # discretized bond, so we narrow via cast.
        discount_curve: YieldTermStructure | None = None
        if isinstance(self._model, TermStructureConsistentModelLike):
            discount_curve = cast("YieldTermStructure", self._model.term_structure)
        if discount_curve is None:
            discount_curve = self._term_structure
        qassert.require(discount_curve is not None, "no term structure available")
        assert discount_curve is not None

        args = self._arguments
        callable_bond = DiscretizedCallableFixedRateBond(args, discount_curve)

        times = callable_bond.mandatory_times()
        time_grid = TimeGrid.with_mandatory_and_steps(times, self._time_steps)
        lattice = self._model.tree(time_grid)

        if spread != 0.0:
            qassert.require(
                isinstance(lattice, ShortRateTree),
                "Spread is not supported for trees other than OneFactorModel",
            )
            assert isinstance(lattice, ShortRateTree)
            lattice.set_spread(spread)

        reference_date = discount_curve.reference_date()
        day_counter = discount_curve.day_counter()
        redemption_time = day_counter.year_fraction(reference_date, args.redemption_date)

        callable_bond.initialize(lattice, redemption_time)
        callable_bond.rollback(0.0)

        results.value = callable_bond.present_value()
        d = discount_curve.discount(args.settlement_date)
        results.settlement_value = results.value / d


class TreeCallableZeroCouponBondEngine(TreeCallableFixedRateBondEngine):
    """Numerical-lattice callable zero-coupon bond engine.

    # C++ parity: ``class TreeCallableZeroCouponBondEngine``
    # (treecallablebondengine.hpp:61-78) — identical behaviour, distinct type.
    """


__all__ = [
    "TreeCallableFixedRateBondEngine",
    "TreeCallableZeroCouponBondEngine",
]
