"""FdSimpleExtOUJumpSwingEngine — FD engine for swing options under Kluge.

# C++ parity:
# ql/experimental/finitedifferences/fdsimpleextoujumpswingengine.{hpp,cpp}
# (v1.42.1).

Prices a :class:`VanillaSwingOption` under the Kluge spot-power model
(Extended-OU + Jumps) by solving a 3-D FD grid (log-spot + jump factor
+ exercise-rights counter). The exercise-rights counter is a discrete
state axis that records how many rights have been consumed; the
backward sweep applies the standard Bermudan choice (exercise vs
continue) at each event.

**W5-B scaffold scope.** Constructor wired. :meth:`calculate` raises
:class:`NotImplementedError` until the ExtOU+Jumps FD operator
cluster lands (W5-A or later).
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.experimental.finitedifferences.process_protocols import (
    ExtOUWithJumpsProcessProtocol,
)
from pquantlib.experimental.finitedifferences.vanilla_swing_option import (
    VanillaSwingOptionArguments,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

# C++ parity: ``typedef FdmExtOUJumpModelInnerValue::Shape Shape;``.
type Shape = Sequence[tuple[float, float]]


class FdSimpleExtOUJumpSwingEngine(
    GenericEngine[VanillaSwingOptionArguments, InstrumentResults]
):
    """FD engine for swing options under the Kluge model.

    # C++ parity: ``class FdSimpleExtOUJumpSwingEngine : public
    # GenericEngine<VanillaSwingOption::arguments,
    #               VanillaSwingOption::results>``.
    """

    def __init__(
        self,
        process: ExtOUWithJumpsProcessProtocol,
        r_ts: YieldTermStructure,
        t_grid: int = 50,
        x_grid: int = 200,
        y_grid: int = 50,
        shape: Shape | None = None,
        scheme_desc: object | None = None,
    ) -> None:
        super().__init__(VanillaSwingOptionArguments(), InstrumentResults())
        self._process: ExtOUWithJumpsProcessProtocol = process
        self._r_ts: YieldTermStructure = r_ts
        self._t_grid: int = int(t_grid)
        self._x_grid: int = int(x_grid)
        self._y_grid: int = int(y_grid)
        self._shape: Shape | None = shape
        self._scheme_desc: object | None = scheme_desc

    def calculate(self) -> None:
        """Run the 3-D ExtOU+Jumps FD backward sweep.

        # C++ parity: ``FdSimpleExtOUJumpSwingEngine::calculate``.
        Currently deferred — depends on
        ``FdmExtendedOrnsteinUhlenbeckOp`` + ``FdmExpExtOUJumpModelOp``
        (W5-A scope).
        """
        raise NotImplementedError(
            "FdSimpleExtOUJumpSwingEngine.calculate depends on the ExtOU+Jumps "
            "FD operator cluster (W5-A). Tracked as a Phase 11 W5 carve-out."
        )


__all__ = ["FdSimpleExtOUJumpSwingEngine", "Shape"]
