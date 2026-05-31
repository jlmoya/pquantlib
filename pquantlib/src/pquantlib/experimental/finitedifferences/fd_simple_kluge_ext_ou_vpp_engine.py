"""FdSimpleKlugeExtOUVPPEngine — FD engine for VPP under Kluge+ExtOU.

# C++ parity:
# ql/experimental/finitedifferences/fdsimpleklugeextouvppengine.{hpp,cpp}
# (v1.42.1).

Prices a :class:`VanillaVPPOption` under the Kluge spot-power model
(Extended-OU + Jumps) combined with an Extended-OU spot-fuel model.
The FD grid is 3-D (power LN-OU, jump factor, gas LN-OU) plus a
state-axis dimension carrying the VPP state machine — solved via
``FdmKlugeExtOUOp`` (a 3-D ADI linear operator) and the VPP step
condition factory built by :class:`FdmVPPStepConditionFactory`.

**W5-B scaffold scope.** The constructor is fully wired; the
``calculate()`` path depends on ``FdmKlugeExtOUOp`` /
``FdmKlugeExtOUSolver`` from the W5-A cluster. Until that lands,
:meth:`calculate` raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.experimental.finitedifferences.process_protocols import (
    KlugeExtOUProcessProtocol,
)
from pquantlib.experimental.finitedifferences.vanilla_vpp_option import (
    VanillaVPPOptionArguments,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

# C++ parity: ``typedef FdmExtOUJumpModelInnerValue::Shape Shape;``.
type Shape = Sequence[tuple[float, float]]


class FdSimpleKlugeExtOUVPPEngine(
    GenericEngine[VanillaVPPOptionArguments, InstrumentResults]
):
    """FD engine for VPP options under the Kluge + ExtOU model.

    # C++ parity: ``class FdSimpleKlugeExtOUVPPEngine : public
    # GenericEngine<VanillaVPPOption::arguments,
    #               VanillaVPPOption::results>``.
    """

    def __init__(
        self,
        process: KlugeExtOUProcessProtocol,
        r_ts: YieldTermStructure,
        fuel_shape: Shape | None,
        power_shape: Shape | None,
        fuel_cost_addon: float,
        t_grid: int = 1,
        x_grid: int = 50,
        y_grid: int = 10,
        g_grid: int = 20,
        scheme_desc: object | None = None,
    ) -> None:
        super().__init__(VanillaVPPOptionArguments(), InstrumentResults())
        self._process: KlugeExtOUProcessProtocol = process
        self._r_ts: YieldTermStructure = r_ts
        self._fuel_shape: Shape | None = fuel_shape
        self._power_shape: Shape | None = power_shape
        self._fuel_cost_addon: float = float(fuel_cost_addon)
        self._t_grid: int = int(t_grid)
        self._x_grid: int = int(x_grid)
        self._y_grid: int = int(y_grid)
        self._g_grid: int = int(g_grid)
        self._scheme_desc: object | None = scheme_desc

    def calculate(self) -> None:
        """Run the 4-D (3 spatial + 1 state) FD backward sweep.

        # C++ parity: ``FdSimpleKlugeExtOUVPPEngine::calculate``.
        Currently deferred — depends on ``FdmKlugeExtOUOp`` /
        ``FdmKlugeExtOUSolver`` (W5-A scope).
        """
        raise NotImplementedError(
            "FdSimpleKlugeExtOUVPPEngine.calculate depends on the Kluge + "
            "ExtOU FD operator cluster (W5-A). Tracked as a Phase 11 W5 "
            "carve-out."
        )


__all__ = ["FdSimpleKlugeExtOUVPPEngine", "Shape"]
