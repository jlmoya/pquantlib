"""FdSimpleExtOUStorageEngine — FD engine for ExtOU storage options.

# C++ parity:
# ql/experimental/finitedifferences/fdsimpleextoustorageengine.{hpp,cpp}
# (v1.42.1).

The C++ engine prices a :class:`VanillaStorageOption` under an
Extended Ornstein-Uhlenbeck (ExtOU) price process by solving a 2-D
finite-difference grid (log-spot + storage-level) with:

* an x-axis mesher built from the ExtOU process at maturity,
* a y-axis "storage" mesher (elevator-style for ``y_grid is None``,
  uniform otherwise),
* a ``FdmSimpleStorageCondition`` Bermudan step condition that
  enforces the change-rate cap per exercise instant.

The full FD solver wires this through ``FdmSimple2dExtOUSolver`` — a
2-D backward-induction solver under the ``FdmKlugeExtOUOp`` linear
operator family that does **not yet exist** in pquantlib.

**W5-B scaffold scope.** The constructor is fully wired (signature
matches C++, dependencies are accepted via process Protocol stubs and
yield curve) so the engine can be instantiated and observed by other
clusters. :meth:`calculate` raises :class:`NotImplementedError`
referencing the ExtOU FD operator dependency — that dependency lands
in the **W5-A** cluster (or whichever subagent owns
``FdmExtendedOrnsteinUhlenbeckOp``).
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib.experimental.finitedifferences.process_protocols import (
    ExtendedOrnsteinUhlenbeckProcessProtocol,
)
from pquantlib.experimental.finitedifferences.vanilla_storage_option import (
    VanillaStorageOptionArguments,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

# C++ parity: ``typedef std::vector<std::pair<Time, Real>> Shape;``.
type Shape = Sequence[tuple[float, float]]


class FdSimpleExtOUStorageEngine(
    GenericEngine[VanillaStorageOptionArguments, InstrumentResults]
):
    """FD engine for simple ExtOU storage options.

    # C++ parity: ``class FdSimpleExtOUStorageEngine : public
    # GenericEngine<VanillaStorageOption::arguments,
    #               VanillaStorageOption::results>``.

    Constructor accepts the full C++ signature; :meth:`calculate`
    raises :class:`NotImplementedError` until the ExtOU FD operator
    cluster lands (W5-A or later).
    """

    def __init__(
        self,
        process: ExtendedOrnsteinUhlenbeckProcessProtocol,
        r_ts: YieldTermStructure,
        t_grid: int = 50,
        x_grid: int = 100,
        y_grid: int | None = None,
        shape: Shape | None = None,
        # C++ ``FdmSchemeDesc`` defaults to ``Douglas()``. The Python
        # port stores the scheme descriptor as an opaque object since
        # the actual scheme dispatch lives in W5-A's FD operator
        # cluster. ``None`` means "use the C++ default".
        scheme_desc: object | None = None,
    ) -> None:
        super().__init__(VanillaStorageOptionArguments(), InstrumentResults())
        self._process: ExtendedOrnsteinUhlenbeckProcessProtocol = process
        self._r_ts: YieldTermStructure = r_ts
        self._t_grid: int = int(t_grid)
        self._x_grid: int = int(x_grid)
        self._y_grid: int | None = y_grid
        self._shape: Shape | None = shape
        self._scheme_desc: object | None = scheme_desc

    def calculate(self) -> None:
        """Run the 2-D ExtOU FD backward sweep.

        # C++ parity: ``FdSimpleExtOUStorageEngine::calculate``.
        Currently deferred — depends on ``FdmExtendedOrnsteinUhlenbeck
        Op`` and ``FdmSimple2dExtOUSolver`` (W5-A scope).
        """
        raise NotImplementedError(
            "FdSimpleExtOUStorageEngine.calculate depends on the ExtOU FD "
            "operator cluster (W5-A). Tracked as a Phase 11 W5 carve-out."
        )


__all__ = ["FdSimpleExtOUStorageEngine", "Shape"]
