"""DynProgVPPIntrinsicValueEngine — closed-form VPP intrinsic-value engine.

# C++ parity:
# ql/experimental/finitedifferences/dynprogvppintrinsicvalueengine.{hpp,cpp}
# (v1.42.1).

Computes the *intrinsic* (perfect-foresight) NPV of a VPP via dynamic
programming over the fuel + power price arrays:

1. Build a state-axis mesh via :class:`FdmVPPStepConditionFactory`.
2. Initialise the per-state value vector to zero.
3. Walk backwards through the per-hour price arrays from index
   ``N-1`` down to index ``0``. At each step apply the
   :class:`FdmVPPStepCondition` Bellman update using the in-state
   fuel + spark-spread prices (looked up by hour index).
4. The final NPV is :meth:`FdmVPPStepCondition.max_value` of the
   resulting state vector.

The C++ engine wraps two simple FdmInnerValueCalculator namespace-
local helpers (``FuelPrice`` and ``SparkSpreadPrice``) that index
the input arrays by ``Size(t)`` — the engine passes integer hour
indices as the time argument to those calculators. The Python port
uses plain closures over the price arrays for parity.

The C++ ``YieldTermStructure rTS`` is held but unused for the
intrinsic path (no discounting in the dynprog formula); the Python
port likewise stores it but doesn't apply discounting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.fdm_vpp_step_condition import (
    FdmVPPStepConditionMesher,
)
from pquantlib.experimental.finitedifferences.fdm_vpp_step_condition_factory import (
    FdmVPPStepConditionFactory,
)
from pquantlib.experimental.finitedifferences.vanilla_vpp_option import (
    VanillaVPPOptionArguments,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class DynProgVPPIntrinsicValueEngine(
    GenericEngine[VanillaVPPOptionArguments, InstrumentResults]
):
    """Intrinsic-value VPP engine via dynamic programming.

    # C++ parity: ``class DynProgVPPIntrinsicValueEngine : public
    # GenericEngine<VanillaVPPOption::arguments, VanillaVPPOption::results>``.
    """

    def __init__(
        self,
        fuel_prices: Sequence[float],
        power_prices: Sequence[float],
        fuel_cost_addon: float,
        r_ts: YieldTermStructure,
    ) -> None:
        super().__init__(VanillaVPPOptionArguments(), InstrumentResults())
        qassert.require(
            len(tuple(fuel_prices)) == len(tuple(power_prices)),
            "fuel and power price arrays must have the same length",
        )
        self._fuel_prices: list[float] = [float(p) for p in fuel_prices]
        self._power_prices: list[float] = [float(p) for p in power_prices]
        self._fuel_cost_addon: float = float(fuel_cost_addon)
        self._r_ts: YieldTermStructure = r_ts

    def calculate(self) -> None:
        """Run the backward dynamic-programming sweep.

        # C++ parity: ``DynProgVPPIntrinsicValueEngine::calculate``.
        """
        args = self._arguments
        qassert.require(args.heat_rate is not None, "heat_rate not set")
        assert args.heat_rate is not None
        heat_rate = args.heat_rate

        # FuelPrice: t → fuel_prices[int(t)].
        fuel_prices = self._fuel_prices
        power_prices = self._power_prices

        def fuel_price(_iter: FdmLinearOpIterator, t: float) -> float:
            i = int(t)
            qassert.require(0 <= i < len(fuel_prices), "invalid time")
            return fuel_prices[i]

        # SparkSpreadPrice: t → power[i] - heat_rate * fuel[i].
        def spark_spread_price(_iter: FdmLinearOpIterator, t: float) -> float:
            i = int(t)
            qassert.require(0 <= i < len(power_prices), "invalid time")
            return power_prices[i] - heat_rate * fuel_prices[i]

        # Build state mesh via the factory.
        factory = FdmVPPStepConditionFactory(args)
        mesher = FdmMesherComposite(factory.state_mesher())
        mesh = FdmVPPStepConditionMesher(state_direction=0, mesher=mesher)

        step_condition = factory.build(
            mesh=mesh,
            fuel_cost_addon=self._fuel_cost_addon,
            fuel=fuel_price,
            spark=spark_spread_price,
        )

        # Initial state vector: all zeros at maturity.
        state = np.zeros(mesher.layout().size(), dtype=np.float64)

        # Walk backwards from hour ``N-1`` down to ``0``. C++:
        # ``for (Size j=powerPrices.size(); j > 0; --j)``.
        for j in range(len(power_prices), 0, -1):
            step_condition.apply_to(state, float(j - 1))

        # Final NPV: the running max across the state vector.
        self._results.value = step_condition.max_value(state)


__all__ = ["DynProgVPPIntrinsicValueEngine"]
