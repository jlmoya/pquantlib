"""G2++ two-factor swaption engine.

# C++ parity: ql/pricingengines/swaption/g2swaptionengine.hpp (v1.42.1).

A thin engine that delegates the heavy lifting to the model itself:
the G2 class exposes a ``swaption(arguments, fixed_rate, range,
intervals)`` method which performs the 2-D numerical integration
over the conditional Gaussian.

Divergences from C++:

- The C++ engine is templated over the concrete G2 class
  (``GenericModelEngine<G2, ...>``). PQuantLib uses structural typing
  — any model object exposing the ``G2ModelLike`` Protocol surface
  works. L4-D's ``G2`` will satisfy this Protocol when it lands.
- Only Physical settlement is supported (matches C++).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementType,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


@runtime_checkable
class G2ModelLike(Protocol):
    """Structural surface needed by G2SwaptionEngine.

    L4-D's ``G2`` class will satisfy this Protocol when it lands.
    """

    @property
    def term_structure(self) -> YieldTermStructureProtocol: ...

    def swaption(
        self,
        arguments: SwaptionArguments,
        fixed_rate: float,
        range_: float,
        intervals: int,
    ) -> float:
        """Closed-form (numerical-integration) G2 swaption price.

        # C++ parity: ``G2::swaption`` in
        # ql/models/shortrate/twofactormodels/g2.cpp:218.

        Performs a 2-D numerical integration over the conditional
        bivariate Gaussian on the two factors at exercise.
        """
        ...


class G2SwaptionEngine(GenericEngine[SwaptionArguments, SwaptionResults]):
    """Analytic-via-quadrature G2++ swaption engine.

    # C++ parity: ``class G2SwaptionEngine`` in
    # g2swaptionengine.hpp:39-73 (v1.42.1).
    """

    def __init__(
        self,
        model: G2ModelLike,
        range_: float,
        intervals: int,
    ) -> None:
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._model: G2ModelLike = model
        self._range: float = range_
        self._intervals: int = intervals

    def calculate(self) -> None:
        # # C++ parity: g2swaptionengine.hpp:51-67 (v1.42.1).
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.settlement_type == SettlementType.Physical,
            "cash-settled swaptions not priced with G2 engine",
        )
        qassert.require(args.swap is not None, "swap not set")
        swap = args.swap
        assert swap is not None

        # Re-price the swap on the model's term structure to extract
        # the spread-adjusted fixed rate (matches C++ g2swaptionengine.hpp:60-64).
        swap.set_pricing_engine(
            DiscountingSwapEngine(self._model.term_structure, include_settlement_date_flows=False)
        )
        correction = (
            swap.spread() * abs(swap.floating_leg_bps() / swap.fixed_leg_bps())
            if swap.spread() != 0.0
            else 0.0
        )
        fixed_rate = swap.fixed_rate() - correction

        results.value = self._model.swaption(
            args, fixed_rate, self._range, self._intervals
        )


# Mark SwapType as used to silence unused-import nag (referenced in the
# typed Protocol bound but not directly in this module body).
_ = SwapType

__all__ = ["G2ModelLike", "G2SwaptionEngine"]
