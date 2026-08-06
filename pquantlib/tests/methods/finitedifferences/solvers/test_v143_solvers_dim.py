"""Cross-validation of the rank-specific FD solvers against C++ v1.43.

Covers ``Fdm1DimSolver``, ``Fdm2DimSolver``, ``Fdm3DimSolver`` and
``FdmNdimSolver``: the classes that own the backward rollback and the
interpolation of its result.

# C++ parity: migration-harness/cpp/probes/v143_methods_solvers/probe.cpp
# blocks A, D, K and L @ v1.43 (submodule 6b57206e0).

Each rank is exercised under all six scheme descriptors the probe sweeps,
so a solver that ignored its ``FdmSchemeDesc`` could not pass: the six
answers differ in the 4th significant digit.

Cheap intermediates — mesher locations, maturity inner values, one
operator ``apply`` — are pinned at TIGHT first, so a mismatch in a
rolled-back value cannot be blamed on the inputs.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import FdmBlackScholesOp
from pquantlib.methods.finitedifferences.operators.fdm_heston_hull_white_op import (
    FdmHestonHullWhiteOp,
)
from pquantlib.methods.finitedifferences.operators.fdm_heston_op import FdmHestonOp
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import (
    Fdm1DimSolver,
    snapshot_time,
)
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_3dim_solver import Fdm3DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_ndim_solver import FdmNdimSolver
from pquantlib.processes.hull_white_forward_process import HullWhiteForwardProcess
from pquantlib.testing.tolerance import tight

from ._v143_fixtures import (
    SCHEME_NAMES,
    SCHEMES,
    bs_1d_setup,
    bs_process,
    check_cancelling,
    check_tight,
    heston_hull_white_setup,
    heston_process,
    heston_setup,
    hull_white_setup,
    r_ts,
    reference,
)

LOG_90 = math.log(90.0)
LOG_100 = math.log(100.0)
LOG_110 = math.log(110.0)


# --- Block A: Fdm1DimSolver ------------------------------------------------


def test_v143_fdm1dimsolver_grid_intermediates() -> None:
    """Mesher locations, maturity inner values, one operator apply (TIGHT).

    # C++ parity: probe.cpp ``block_fdm1dimsolver`` intermediates.
    """
    setup = bs_1d_setup()
    locations = setup.mesher.locations(0)
    check_tight("a_mesher_loc_0", float(locations[0]))
    check_tight("a_mesher_loc_12", float(locations[12]))
    check_tight("a_mesher_loc_24", float(locations[24]))

    initial = [setup.calculator.avg_inner_value(it, 1.0) for it in setup.mesher.layout().iter()]
    check_tight("a_init_0", initial[0])
    check_tight("a_init_12", initial[12])
    check_tight("a_init_18", initial[18])
    check_tight("a_init_24", initial[24])

    op = FdmBlackScholesOp(setup.mesher, bs_process(), 100.0)
    op.set_time(0.0, 1.0)
    applied = op.apply(np.asarray(initial, dtype=np.float64))
    check_tight("a_op_apply_1", float(applied[1]))
    check_tight("a_op_apply_12", float(applied[12]))
    check_tight("a_op_apply_23", float(applied[23]))


def test_v143_fdm1dimsolver_snapshot_time() -> None:
    """The theta snapshot lands at ``0.99/365`` for a 1y problem with no stopping times.

    # C++ parity: the ``thetaCondition_`` member initialiser.
    """
    tight(snapshot_time(bs_1d_setup().desc), 0.99 / 365.0)


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm1dimsolver(scheme: str) -> None:
    """Values, node value, theta and both spline derivatives, per scheme.

    # C++ parity: probe.cpp ``block_fdm1dimsolver``.
    """
    setup = bs_1d_setup()
    solver = Fdm1DimSolver(
        setup.desc, SCHEMES[scheme], FdmBlackScholesOp(setup.mesher, bs_process(), 100.0)
    )
    p = f"a_{scheme}_"
    check_tight(p + "value_90", solver.interpolate_at(LOG_90))
    check_tight(p + "value_100", solver.interpolate_at(LOG_100))
    check_tight(p + "value_110", solver.interpolate_at(LOG_110))
    # Evaluating exactly on a grid node must return that node's rolled-back
    # value, not an interpolant that merely passes near it.
    check_tight(p + "value_node", solver.interpolate_at(reference()["a_node_x"]))
    check_tight(p + "dx_100", solver.derivative_x(LOG_100))
    check_tight(p + "dxx_100", solver.derivative_xx(LOG_100))
    check_cancelling(p + "theta_100", solver.theta_at(LOG_100))


def test_v143_fdm1dimsolver_scheme_dispatch_reproduces_the_cpp_degeneracy() -> None:
    """Python's six 1-D answers separate exactly where C++'s six separate.

    # C++ parity: probe.cpp ``block_fdm1dimsolver`` (``a_*_value_100``).

    C++ gives **four** distinct values, not six, and the collapses are
    structural rather than coincidental:

    * ``craigsneyd`` equals ``douglas`` *to the last bit*. In
      craigsneydscheme.cpp the mixed-derivative correction
      ``yt = y0 + mu*dt*apply_mixed(y-a)`` is a no-op for a 1-D operator
      (``apply_mixed`` is identically zero), and the scheme's **second**
      splitting loop differences ``apply_direction(i, a)`` — the *original*
      ``a``, not the first loop's output ``y``. The second loop is therefore
      literally the first loop recomputed, so CraigSneyd reduces to Douglas.
      A port that transcribed that second loop as ``apply_direction(i, y)``
      would still look plausible and would break exactly this assertion.
    * ``cn`` agrees with ``douglas`` to rounding (identical at S=100, 8 ulp
      apart at S=90): at theta=0.5 the two are the same scheme written two
      ways.

    ``hundsdorfer``, ``implicit`` and ``explicit`` do separate, so a solver
    that ignored its ``FdmSchemeDesc`` could not pass.
    """
    setup = bs_1d_setup()
    actual = {
        name: Fdm1DimSolver(
            setup.desc, SCHEMES[name], FdmBlackScholesOp(setup.mesher, bs_process(), 100.0)
        ).interpolate_at(LOG_100)
        for name in SCHEME_NAMES
    }
    expected = {name: reference()[f"a_{name}_value_100"] for name in SCHEME_NAMES}

    # The C++ ground truth this test encodes.
    assert len(set(expected.values())) == 4
    assert expected["craigsneyd"] == expected["douglas"]

    # Python must collapse in the same places and separate in the same places.
    assert actual["craigsneyd"] == actual["douglas"]
    for a, b in itertools.combinations(SCHEME_NAMES, 2):
        if expected[a] == expected[b]:
            assert actual[a] == actual[b], f"{a} and {b} are bit-identical in C++"
        else:
            # C++'s smallest genuine gap here is douglas vs hundsdorfer,
            # 2.3e-5 absolute; anything above 1e-9 is unambiguously a real
            # separation rather than accumulated rounding.
            assert abs(actual[a] - actual[b]) > 1e-9, f"{a} and {b} differ in C++"


# --- Block D: Fdm2DimSolver ------------------------------------------------


def test_v143_fdm2dimsolver_grid_intermediates() -> None:
    """Both axes, maturity inner values and one Heston ``apply`` (TIGHT).

    # C++ parity: probe.cpp ``block_fdm2dimsolver`` intermediates.
    """
    setup = heston_setup()
    check_tight("d_mesher_x_loc_0", float(setup.mesher.locations(0)[0]))
    check_tight("d_mesher_x_loc_12", float(setup.mesher.locations(0)[12]))
    check_tight("d_mesher_y_loc_0", float(setup.mesher.locations(1)[0]))
    check_tight("d_mesher_y_loc_13", float(setup.mesher.locations(1)[13]))

    initial = [setup.calculator.avg_inner_value(it, 0.25) for it in setup.mesher.layout().iter()]
    check_tight("d_init_0", initial[0])
    check_tight("d_init_50", initial[50])
    check_tight("d_init_100", initial[100])

    op = FdmHestonOp(setup.mesher, heston_process())
    op.set_time(0.0, 0.25)
    applied = op.apply(np.asarray(initial, dtype=np.float64))
    check_tight("d_op_apply_20", float(applied[20]))
    check_tight("d_op_apply_60", float(applied[60]))


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm2dimsolver(scheme: str) -> None:
    """Values, node value, theta and all five bicubic partials, per scheme.

    # C++ parity: probe.cpp ``block_fdm2dimsolver``.
    """
    setup = heston_setup()
    solver = Fdm2DimSolver(
        setup.desc, SCHEMES[scheme], FdmHestonOp(setup.mesher, heston_process())
    )
    p = f"d_{scheme}_"
    check_tight(p + "value_100_004", solver.interpolate_at(LOG_100, 0.04))
    check_tight(p + "value_90_01", solver.interpolate_at(LOG_90, 0.10))
    check_tight(
        p + "value_node",
        solver.interpolate_at(reference()["d_node_x"], reference()["d_node_y"]),
    )
    check_tight(p + "dx", solver.derivative_x(LOG_100, 0.04))
    check_tight(p + "dy", solver.derivative_y(LOG_100, 0.04))
    check_tight(p + "dxx", solver.derivative_xx(LOG_100, 0.04))
    check_tight(p + "dyy", solver.derivative_yy(LOG_100, 0.04))
    check_tight(p + "dxy", solver.derivative_xy(LOG_100, 0.04))
    check_cancelling(p + "theta_100_004", solver.theta_at(LOG_100, 0.04))


# --- Block K: Fdm3DimSolver ------------------------------------------------


def _hw_process() -> HullWhiteForwardProcess:
    """a = 0.1, sigma = 0.01 — probe.cpp ``HullWhiteProcess(rTS, 0.1, 0.01)``."""
    return HullWhiteForwardProcess(r_ts(), 0.1, 0.01)


def test_v143_fdm3dimsolver_grid_intermediates() -> None:
    """The z axis and two maturity inner values (TIGHT).

    # C++ parity: probe.cpp ``block_3dim`` intermediates.
    """
    setup = heston_hull_white_setup()
    check_tight("k_mesher_z_loc_0", float(setup.mesher.locations(2)[0]))
    check_tight("k_mesher_z_loc_384", float(setup.mesher.locations(2)[384]))

    initial = [setup.calculator.avg_inner_value(it, 0.25) for it in setup.mesher.layout().iter()]
    check_tight("k_init_0", initial[0])
    check_tight("k_init_200", initial[200])


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm3dimsolver(scheme: str) -> None:
    """Two interior points and theta on the 11x7x5 mesh, per scheme.

    # C++ parity: probe.cpp ``block_3dim`` (``k3_`` keys).
    """
    setup = heston_hull_white_setup()
    solver = Fdm3DimSolver(
        setup.desc,
        SCHEMES[scheme],
        FdmHestonHullWhiteOp(setup.mesher, heston_process(), _hw_process(), -0.2),
    )
    p = f"k3_{scheme}_"
    check_tight(p + "value", solver.interpolate_at(LOG_100, 0.04, 0.05))
    check_tight(p + "value_2", solver.interpolate_at(LOG_90, 0.10, 0.02))
    check_cancelling(p + "theta", solver.theta_at(LOG_100, 0.04, 0.05))


# --- Block L: FdmNdimSolver ------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdmndimsolver(scheme: str) -> None:
    """The N=2 instantiation on the Heston mesh, per scheme.

    # C++ parity: probe.cpp ``block_fdmndimsolver``.

    Same rollback as ``Fdm2DimSolver`` but read out through
    ``MultiCubicSpline`` instead of ``BicubicSpline`` — the two are
    different functions off the grid, so this also pins that the right one
    is used.
    """
    setup = heston_setup()
    solver = FdmNdimSolver(
        setup.desc, SCHEMES[scheme], FdmHestonOp(setup.mesher, heston_process()), 2
    )
    p = f"l_{scheme}_"
    check_tight(p + "value_100_004", solver.interpolate_at([LOG_100, 0.04]))
    check_tight(p + "value_90_01", solver.interpolate_at([LOG_90, 0.10]))
    check_cancelling(p + "theta_100_004", solver.theta_at([LOG_100, 0.04]))


def test_v143_fdmndimsolver_rejects_wrong_rank() -> None:
    """``n`` mismatching the layout rank raises.

    # C++ parity: ``QL_REQUIRE(layout->dim().size() == N, ...)``.
    """
    setup = hull_white_setup()
    with pytest.raises(Exception, match="does not fit to layout dim"):
        FdmNdimSolver(
            setup.desc,
            SCHEMES["douglas"],
            FdmBlackScholesOp(setup.mesher, bs_process(), 1.0),
            3,
        )
