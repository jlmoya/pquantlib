"""Cross-validation of the Black-Scholes FD solvers against C++ v1.43.

Covers ``FdmBlackScholesSolver``, ``FdmSimple2dBSSolver`` and
``Fdm2dBlackScholesSolver``.

# C++ parity: migration-harness/cpp/probes/v143_methods_solvers/probe.cpp
# blocks B, F and G @ v1.43 (submodule 6b57206e0).
"""

from __future__ import annotations

import pytest

from pquantlib.methods.finitedifferences.solvers.fdm_2d_black_scholes_solver import (
    Fdm2dBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_black_scholes_solver import (
    FdmBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_simple_2d_bs_solver import (
    FdmSimple2dBSSolver,
)

from ._v143_fixtures import (
    SCHEME_NAMES,
    SCHEMES,
    bs_1d_damped_setup,
    bs_1d_setup,
    bs_process,
    check_cancelling,
    check_tight,
    second_asset_process,
    simple_2d_bs_setup,
    two_asset_setup,
)

# --- Block B: FdmBlackScholesSolver ---------------------------------------


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm_black_scholes_solver(scheme: str) -> None:
    """Value, delta, gamma and theta at three spots, per scheme.

    # C++ parity: probe.cpp ``block_fdmblackscholessolver``.
    """
    setup = bs_1d_setup()
    solver = FdmBlackScholesSolver(bs_process(), 100.0, setup.desc, SCHEMES[scheme])
    p = f"b_{scheme}_"
    check_tight(p + "value_90", solver.value_at(90.0))
    check_tight(p + "value_100", solver.value_at(100.0))
    check_tight(p + "value_110", solver.value_at(110.0))
    check_tight(p + "delta_100", solver.delta_at(100.0))
    check_tight(p + "gamma_100", solver.gamma_at(100.0))
    check_cancelling(p + "theta_100", solver.theta_at(100.0))


def test_v143_fdm_black_scholes_solver_damping_steps() -> None:
    """Five implicit-Euler damping steps ahead of the Douglas sweep.

    # C++ parity: probe.cpp ``block_fdmblackscholessolver`` (``b_damped_*``).
    The damped answer differs from the undamped one in the 4th digit, so
    this pins that ``damping_steps`` reaches the backward solver.
    """
    solver = FdmBlackScholesSolver(
        bs_process(), 100.0, bs_1d_damped_setup().desc, SCHEMES["douglas"]
    )
    check_tight("b_damped_value_100", solver.value_at(100.0))
    check_tight("b_damped_delta_100", solver.delta_at(100.0))


# --- Block F: FdmSimple2dBSSolver -----------------------------------------


def test_v143_fdm_simple_2d_bs_solver_intermediates() -> None:
    """The auxiliary axis and three max-basket maturity values (TIGHT).

    # C++ parity: probe.cpp ``block_fdmsimple2dbssolver`` intermediates.
    """
    setup = simple_2d_bs_setup()
    check_tight("f_mesher_y_loc_21", float(setup.mesher.locations(1)[21]))
    initial = [setup.calculator.avg_inner_value(it, 1.0) for it in setup.mesher.layout().iter()]
    check_tight("f_init_0", initial[0])
    check_tight("f_init_100", initial[100])
    check_tight("f_init_200", initial[200])


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm_simple_2d_bs_solver(scheme: str) -> None:
    """Values at two (spot, average) pairs plus bumped greeks and theta.

    # C++ parity: probe.cpp ``block_fdmsimple2dbssolver``.
    """
    setup = simple_2d_bs_setup()
    solver = FdmSimple2dBSSolver(bs_process(), 100.0, setup.desc, SCHEMES[scheme])
    p = f"f_{scheme}_"
    check_tight(p + "value_100_100", solver.value_at(100.0, 100.0))
    check_tight(p + "value_90_110", solver.value_at(90.0, 110.0))
    check_cancelling(p + "delta", solver.delta_at(100.0, 100.0, 0.5))
    check_cancelling(p + "gamma", solver.gamma_at(100.0, 100.0, 0.5))
    check_cancelling(p + "theta", solver.theta_at(100.0, 100.0))


# --- Block G: Fdm2dBlackScholesSolver -------------------------------------


def test_v143_fdm_2d_black_scholes_solver_intermediates() -> None:
    """The second log-spot axis and three rainbow maturity values (TIGHT).

    # C++ parity: probe.cpp ``block_fdm2dblackscholessolver`` intermediates.
    """
    setup = two_asset_setup()
    check_tight("g_mesher_y_loc_13", float(setup.mesher.locations(1)[13]))
    initial = [setup.calculator.avg_inner_value(it, 0.25) for it in setup.mesher.layout().iter()]
    check_tight("g_init_0", initial[0])
    check_tight("g_init_84", initial[84])
    check_tight("g_init_168", initial[168])


@pytest.mark.parametrize("scheme", SCHEME_NAMES)
def test_v143_fdm_2d_black_scholes_solver(scheme: str) -> None:
    """Values, theta and all five greeks of the correlated two-asset solver.

    # C++ parity: probe.cpp ``block_fdm2dblackscholessolver``.

    The payoff is a max-basket, so the second asset genuinely drives the
    solution: ``delta_y`` and ``gamma_xy`` are both non-zero here, unlike
    a single-direction payoff where they would vanish identically and the
    y axis would go untested.
    """
    setup = two_asset_setup()
    solver = Fdm2dBlackScholesSolver(
        bs_process(), second_asset_process(), 0.30, setup.desc, SCHEMES[scheme]
    )
    p = f"g_{scheme}_"
    check_tight(p + "value_100_95", solver.value_at(100.0, 95.0))
    check_tight(p + "value_110_90", solver.value_at(110.0, 90.0))
    check_tight(p + "delta_x", solver.delta_x_at(100.0, 95.0))
    check_tight(p + "delta_y", solver.delta_y_at(100.0, 95.0))
    check_tight(p + "gamma_x", solver.gamma_x_at(100.0, 95.0))
    check_tight(p + "gamma_y", solver.gamma_y_at(100.0, 95.0))
    check_tight(p + "gamma_xy", solver.gamma_xy_at(100.0, 95.0))
    check_cancelling(p + "theta", solver.theta_at(100.0, 95.0))
