"""``FdmSolverDesc.bc_set`` really reaches the scheme — cross-validated.

# C++ parity: migration-harness/cpp/probes/v143_methods_solvers/probe.cpp
# block M (``m1_*`` / ``m2_*``) @ v1.43 (submodule 6b57206e0).

Every other test in this package runs with an empty boundary-condition set,
which cannot tell a solver that threads ``solver_desc.bc_set`` through to
``FdmBackwardSolver`` from one that drops it. These tests put an
``FdmDirichletBoundary`` on one or both faces of the same 1-D and 2-D
problems and pin the result against C++.

The pin has real force: with the upper face knocked to zero the 1-D Douglas
answer at S=100 falls from 9.1948 to 6.8898, so a dropped ``bc_set`` is a
25% error, not a rounding difference.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from pquantlib.methods.finitedifferences.fdm_boundary_condition import BoundaryConditionSide
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import FdmBlackScholesOp
from pquantlib.methods.finitedifferences.operators.fdm_heston_op import FdmHestonOp
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import Fdm1DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.utilities.fdm_dirichlet_boundary import (
    FdmDirichletBoundary,
)

from ._v143_fixtures import (
    SCHEME_NAMES,
    SCHEMES,
    bs_1d_setup,
    bs_process,
    check_tight,
    heston_process,
    heston_setup,
)

LOG_55 = math.log(55.0)
LOG_90 = math.log(90.0)
LOG_100 = math.log(100.0)
LOG_140 = math.log(140.0)
LOG_145 = math.log(145.0)


def _with_bc(desc: FdmSolverDesc, *bcs: FdmDirichletBoundary) -> FdmSolverDesc:
    """``desc`` with ``bcs`` as its boundary-condition set."""
    return dataclasses.replace(desc, bc_set=bcs)


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm1dimsolver_honours_an_upper_dirichlet_face(scheme: str) -> None:
    """1-D call with the upper log-spot face pinned to zero, per scheme.

    # C++ parity: probe.cpp ``block_bcset`` (``m1_<scheme>_*``).
    """
    setup = bs_1d_setup()
    desc = _with_bc(
        setup.desc,
        FdmDirichletBoundary(setup.mesher, 0.0, 0, BoundaryConditionSide.UPPER),
    )
    solver = Fdm1DimSolver(
        desc, SCHEMES[scheme], FdmBlackScholesOp(setup.mesher, bs_process(), 100.0)
    )
    p = f"m1_{scheme}_"
    check_tight(p + "value_100", solver.interpolate_at(LOG_100))
    check_tight(p + "value_140", solver.interpolate_at(LOG_140))
    check_tight(p + "dx_100", solver.derivative_x(LOG_100))


def test_v143_fdm1dimsolver_applies_every_condition_in_the_set() -> None:
    """Two Dirichlet faces at once — both are applied, not just ``bc_set[0]``.

    # C++ parity: probe.cpp ``block_bcset`` (``m1_both_*``).

    The lower face is pinned to 0 and the upper to 7.5, a value the
    unconstrained solution never takes there, so a port that applied only
    the first condition would miss ``m1_both_value_145`` by ~7.
    """
    setup = bs_1d_setup()
    desc = _with_bc(
        setup.desc,
        FdmDirichletBoundary(setup.mesher, 0.0, 0, BoundaryConditionSide.LOWER),
        FdmDirichletBoundary(setup.mesher, 7.5, 0, BoundaryConditionSide.UPPER),
    )
    solver = Fdm1DimSolver(
        desc, SCHEMES["douglas"], FdmBlackScholesOp(setup.mesher, bs_process(), 100.0)
    )
    check_tight("m1_both_value_100", solver.interpolate_at(LOG_100))
    check_tight("m1_both_value_55", solver.interpolate_at(LOG_55))
    check_tight("m1_both_value_145", solver.interpolate_at(LOG_145))


def test_v143_fdm1dimsolver_bcset_survives_the_damping_steps() -> None:
    """The implicit-Euler damping sweep runs under the same boundary set.

    # C++ parity: probe.cpp ``block_bcset`` (``m1_damped_value_100``).
    """
    setup = bs_1d_setup()
    desc = dataclasses.replace(
        _with_bc(
            setup.desc,
            FdmDirichletBoundary(setup.mesher, 0.0, 0, BoundaryConditionSide.UPPER),
        ),
        damping_steps=5,
    )
    solver = Fdm1DimSolver(
        desc, SCHEMES["douglas"], FdmBlackScholesOp(setup.mesher, bs_process(), 100.0)
    )
    check_tight("m1_damped_value_100", solver.interpolate_at(LOG_100))


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm2dimsolver_honours_a_dirichlet_face_on_the_variance_axis(scheme: str) -> None:
    """Heston mesh with the top of the variance axis pinned to 3.0, per scheme.

    # C++ parity: probe.cpp ``block_bcset`` (``m2_<scheme>_*``).

    Direction 1 is used deliberately: ``FdmIndicesOnBoundary`` has to stride
    correctly through the layout to find that face, which a direction-0-only
    implementation would get wrong.
    """
    setup = heston_setup()
    desc = _with_bc(
        setup.desc,
        FdmDirichletBoundary(setup.mesher, 3.0, 1, BoundaryConditionSide.UPPER),
    )
    solver = Fdm2DimSolver(desc, SCHEMES[scheme], FdmHestonOp(setup.mesher, heston_process()))
    p = f"m2_{scheme}_"
    check_tight(p + "value_100_004", solver.interpolate_at(LOG_100, 0.04))
    check_tight(p + "value_90_03", solver.interpolate_at(LOG_90, 0.30))
    check_tight(p + "dy", solver.derivative_y(LOG_100, 0.04))
