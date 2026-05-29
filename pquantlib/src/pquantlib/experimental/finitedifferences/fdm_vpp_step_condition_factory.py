"""FdmVPPStepConditionFactory — build VPP step conditions from arguments.

# C++ parity:
# ql/experimental/finitedifferences/fdmvppstepconditionfactory.{hpp,cpp}
# (v1.42.1).

Classifies a :class:`VanillaVPPOptionArguments` by which of the two
optional limits is set (``n_starts`` or ``n_running_hours``) and
produces:

* ``state_mesher()`` — the 1-D state-axis mesher (uniformly spaced
  over ``[0, 1]`` with cardinality = ``n_states``).
* ``build(mesh, fuel_cost_addon, fuel_calc, spark_calc)`` — a
  configured :class:`FdmVPPStartLimitStepCondition` (both the
  vanilla and start-limit cases route through this same class; the
  vanilla case is ``n_starts=None`` which yields a single
  state-machine cycle).

The C++ ``RunningHourLimit`` enum value is documented but not built
— the C++ source itself raises ``QL_FAIL("vpp type is not supported")``
in that branch. The Python port matches that behavior.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

from pquantlib import qassert
from pquantlib.experimental.finitedifferences.fdm_vpp_start_limit_step_condition import (
    FdmVPPStartLimitStepCondition,
)
from pquantlib.experimental.finitedifferences.fdm_vpp_step_condition import (
    FdmVPPStepCondition,
    FdmVPPStepConditionMesher,
    FdmVPPStepConditionParams,
    InnerValueCalculator,
)
from pquantlib.experimental.finitedifferences.vanilla_vpp_option import (
    VanillaVPPOptionArguments,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)


class _VPPType(IntEnum):
    """C++ ``FdmVPPStepConditionFactory::Type``."""

    Vanilla = 0
    StartLimit = 1
    RunningHourLimit = 2


@final
class FdmVPPStepConditionFactory:
    """Factory for VPP step conditions.

    # C++ parity: ``class FdmVPPStepConditionFactory``.
    """

    def __init__(self, args: VanillaVPPOptionArguments) -> None:
        # C++ parity: the ctor calls ``QL_REQUIRE`` that not both
        # nStarts and nRunningHours are set; arguments.validate() already
        # checks this, but the factory rechecks for robustness.
        qassert.require(
            not (args.n_starts is not None and args.n_running_hours is not None),
            "start and running hour limt together is not supported",
        )
        if args.n_starts is None and args.n_running_hours is None:
            self._type: _VPPType = _VPPType.Vanilla
        elif args.n_running_hours is None:
            self._type = _VPPType.StartLimit
        else:
            self._type = _VPPType.RunningHourLimit
        self._args: VanillaVPPOptionArguments = args

    def state_mesher(self) -> Fdm1dMesher:
        """The 1-D state-axis mesher (uniform on [0, 1]).

        # C++ parity: ``FdmVPPStepConditionFactory::stateMesher``.
        """
        assert self._args.t_min_up is not None
        assert self._args.t_min_down is not None
        if self._type == _VPPType.Vanilla:
            n_states = 2 * self._args.t_min_up + self._args.t_min_down
        elif self._type == _VPPType.StartLimit:
            assert self._args.n_starts is not None
            n_states = FdmVPPStartLimitStepCondition.compute_n_states(
                self._args.t_min_up,
                self._args.t_min_down,
                self._args.n_starts,
            )
        else:
            qassert.fail("vpp type is not supported")
        return Uniform1dMesher(0.0, 1.0, n_states)

    def build(
        self,
        mesh: FdmVPPStepConditionMesher,
        fuel_cost_addon: float,
        fuel: InnerValueCalculator,
        spark: InnerValueCalculator,
    ) -> FdmVPPStepCondition:
        """Build a configured VPP step condition.

        # C++ parity: ``FdmVPPStepConditionFactory::build``.
        """
        assert self._args.heat_rate is not None
        assert self._args.p_min is not None
        assert self._args.p_max is not None
        assert self._args.t_min_up is not None
        assert self._args.t_min_down is not None
        assert self._args.start_up_fuel is not None
        assert self._args.start_up_fix_cost is not None

        params = FdmVPPStepConditionParams(
            heat_rate=self._args.heat_rate,
            p_min=self._args.p_min,
            p_max=self._args.p_max,
            t_min_up=self._args.t_min_up,
            t_min_down=self._args.t_min_down,
            start_up_fuel=self._args.start_up_fuel,
            start_up_fix_cost=self._args.start_up_fix_cost,
            fuel_cost_addon=fuel_cost_addon,
        )
        if self._type in (_VPPType.Vanilla, _VPPType.StartLimit):
            return FdmVPPStartLimitStepCondition(
                params=params,
                n_starts=self._args.n_starts,
                mesh=mesh,
                gas_price=fuel,
                spark_spread_price=spark,
            )
        qassert.fail("vpp type is not supported")
        # qassert.fail raises — unreachable.
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["FdmVPPStepConditionFactory"]
