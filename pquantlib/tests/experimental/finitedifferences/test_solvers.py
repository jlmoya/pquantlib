"""Tests for the FD solver wrappers.

# C++ parity: ql/experimental/finitedifferences/
#   fdmextoujumpsolver.hpp
#   fdmklugeextousolver.hpp
#   fdmsimple2dextousolver.hpp
#   fdmsimple3dextoujumpsolver.hpp

``FdmExtOUJumpSolver`` and ``FdmKlugeExtOUSolver`` are live: they wrap
``Fdm2DimSolver`` / ``FdmNdimSolver``. Their numerical output is
cross-validated end to end through the two engines that use them, in
``test_ext_ou_fd_engines_v143.py``; what is covered here is the wrapper
contract — that ``value_at`` runs, that it is lazy, and that ``n < 3`` is
rejected.

``FdmSimple2dExtOUSolver`` and ``FdmSimple3dExtOUJumpSolver`` are still
carve-outs whose ``value_at`` raises ``NotImplementedError``; their tests
below cover that error path.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_solver import (
    FdmExtOUJumpSolver,
)
from pquantlib.experimental.finitedifferences.fdm_kluge_ext_ou_solver import (
    FdmKlugeExtOUSolver,
)
from pquantlib.experimental.finitedifferences.fdm_simple_2d_ext_ou_solver import (
    FdmSimple2dExtOUSolver,
)
from pquantlib.experimental.finitedifferences.fdm_simple_3d_ext_ou_jump_solver import (
    FdmSimple3dExtOUJumpSolver,
)
from pquantlib.experimental.processes.ext_ou_with_jumps_process import (
    ExtOUWithJumpsProcess,
)
from pquantlib.experimental.processes.extended_ornstein_uhlenbeck_process import (
    ExtendedOrnsteinUhlenbeckProcess,
)
from pquantlib.experimental.processes.kluge_ext_ou_process import KlugeExtOUProcess
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.date import Date, Month


def _make_r_ts() -> FlatForward:
    today = Date.from_ymd(15, Month.January, 2024)
    return FlatForward.from_rate(today, 0.05, Actual365Fixed())


def _make_2d_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(-2.0, 2.0, 5), Uniform1dMesher(0.0, 2.0, 5)
    )


def _make_3d_mesher() -> FdmMesherComposite:
    return FdmMesherComposite(
        Uniform1dMesher(-2.0, 2.0, 5),
        Uniform1dMesher(0.0, 2.0, 4),
        Uniform1dMesher(-2.0, 2.0, 5),
    )


def _trivial_calc(iter_: object, t: float) -> float:
    del iter_, t
    return 0.0


def _desc(mesher: FdmMesherComposite) -> FdmSolverDesc:
    return FdmSolverDesc(
        mesher=mesher,
        condition=FdmStepConditionComposite([], []),
        calculator=_trivial_calc,
        maturity=0.25,
        time_steps=4,
        damping_steps=0,
    )


def test_fdm_ext_ou_jump_solver_rolls_back_a_zero_payoff() -> None:
    """A zero inner value must roll back to zero, whatever the operator does.

    # C++ parity: ``FdmExtOUJumpSolver::valueAt`` -> ``Fdm2DimSolver::
    # interpolateAt`` (fdmextoujumpsolver.cpp:51-54). The engine-level
    # cross-validation lives in test_ext_ou_fd_engines_v143.py; this only pins
    # that the wrapper actually runs the solver instead of raising.
    """
    ou = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    kluge = ExtOUWithJumpsProcess(ou, 0.0, 4.0, 2.0, 4.0)
    solver = FdmExtOUJumpSolver(kluge, _make_r_ts(), _desc(_make_2d_mesher()))
    tolerance.tight(solver.value_at(0.0, 0.5), 0.0)


def test_fdm_ext_ou_jump_solver_is_lazy() -> None:
    """``LazyObject``: the inner solver is built on first use, then cached."""
    ou = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    kluge = ExtOUWithJumpsProcess(ou, 0.0, 4.0, 2.0, 4.0)
    solver = FdmExtOUJumpSolver(kluge, _make_r_ts(), _desc(_make_2d_mesher()))
    assert not solver.is_calculated()
    solver.value_at(0.0, 0.5)
    assert solver.is_calculated()


def test_fdm_kluge_ext_ou_solver_rolls_back_a_zero_payoff() -> None:
    """# C++ parity: ``FdmKlugeExtOUSolver::valueAt`` (fdmklugeextousolver.hpp:52-55)."""
    ou = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    kluge = ExtOUWithJumpsProcess(ou, 0.0, 4.0, 2.0, 4.0)
    ext_ou_b = ExtendedOrnsteinUhlenbeckProcess(0.5, 0.25, 0.0, lambda _t: 0.0)
    kluge_ext_ou = KlugeExtOUProcess(0.4, kluge, ext_ou_b)
    solver = FdmKlugeExtOUSolver(
        kluge_ext_ou, _make_r_ts(), _desc(_make_3d_mesher())
    )
    tolerance.tight(solver.value_at([0.0, 0.5, 0.0]), 0.0)


def test_fdm_kluge_ext_ou_solver_rejects_n_below_3() -> None:
    """N < 3 rejected, matching the C++ ``BOOST_STATIC_ASSERT(N >= 3)``."""
    ou = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    kluge = ExtOUWithJumpsProcess(ou, 0.0, 4.0, 2.0, 4.0)
    ext_ou_b = ExtendedOrnsteinUhlenbeckProcess(0.5, 0.25, 0.0, lambda _t: 0.0)
    kluge_ext_ou = KlugeExtOUProcess(0.4, kluge, ext_ou_b)
    with pytest.raises(LibraryException):
        FdmKlugeExtOUSolver(
            kluge_ext_ou, _make_r_ts(), _desc(_make_3d_mesher()), None, 2
        )


def test_fdm_simple_2d_ext_ou_solver_value_at_raises() -> None:
    """value_at raises NotImplementedError."""
    process = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    solver = FdmSimple2dExtOUSolver(
        process=process,
        r_ts=_make_r_ts(),
        mesher=_make_2d_mesher(),
        condition=None,
        calculator=_trivial_calc,
        maturity=0.25,
        time_steps=10,
    )
    with pytest.raises(NotImplementedError):
        solver.value_at(0.0, 0.0)


def test_fdm_simple_3d_ext_ou_jump_solver_value_at_raises() -> None:
    """value_at raises NotImplementedError."""
    ou = ExtendedOrnsteinUhlenbeckProcess(1.0, 0.3, 0.0, lambda _t: 0.0)
    kluge = ExtOUWithJumpsProcess(ou, 0.0, 4.0, 2.0, 4.0)
    solver = FdmSimple3dExtOUJumpSolver(
        process=kluge,
        r_ts=_make_r_ts(),
        mesher=_make_3d_mesher(),
        condition=None,
        calculator=_trivial_calc,
        maturity=0.25,
        time_steps=10,
    )
    with pytest.raises(NotImplementedError):
        solver.value_at(0.0, 0.0, 0.0)
