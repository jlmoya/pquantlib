"""Tests for RatePseudoRootJacobian + AllElements + Numerical (W11-D batch a).

Cross-validates against ``migration-harness/references/cluster/w11d.json``.

C++ parity:
  ql/models/marketmodels/pathwisegreeks/ratepseudorootjacobian.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pquantlib.models.marketmodels.pathwisegreeks.rate_pseudo_root_jacobian import (
    RatePseudoRootJacobian,
    RatePseudoRootJacobianAllElements,
    RatePseudoRootJacobianNumerical,
)
from pquantlib.testing.tolerance import tight

from .conftest import N_FACTORS, N_RATES, reshape, reshape_stack


def _bumps() -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    """The 4 unit pseudo-bumps the probe used: (0,0),(2,1),(4,2),(5,0)."""
    spots = [(0, 0), (2, 1), (4, 2), (5, 0)]
    out: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for r, f in spots:
        b = np.zeros((N_RATES, N_FACTORS), dtype=np.float64)
        b[r][f] = 1.0
        out.append(b)
    return out


def _inputs(ref: dict[str, Any]) -> tuple[Any, ...]:
    pseudo_root = reshape(ref["jac_pseudo_root"], N_RATES, N_FACTORS)
    old_rates = list(ref["jac_old_rates"])
    new_rates = list(ref["jac_new_rates"])
    gaussians = list(ref["jac_gaussians"])
    one_step_dfs = list(ref["jac_one_step_dfs"])
    taus = list(ref["rate_taus"])
    displacements = list(ref["displacements"])
    return (
        pseudo_root,
        old_rates,
        new_rates,
        gaussians,
        one_step_dfs,
        taus,
        displacements,
    )


def test_rate_pseudo_root_jacobian_analytic(ref: dict[str, Any]) -> None:
    """Analytic ``B`` matrix matches C++ (TIGHT — pure linear algebra)."""
    pr, old, new, g, dfs, taus, disp = _inputs(ref)
    alive = 0  # step 0 => first alive rate 0; numeraire = alive (MM account)
    n_bumps = ref["jac_n_bumps"]
    jac = RatePseudoRootJacobian(pr, alive, alive, taus, _bumps(), disp)
    b_out = np.zeros((n_bumps, N_RATES), dtype=np.float64)
    jac.get_bumps(old, dfs, new, g, b_out)

    expected = reshape(ref["jac_B"], n_bumps, N_RATES)
    for i in range(n_bumps):
        for j in range(N_RATES):
            tight(float(b_out[i][j]), float(expected[i][j]))


def test_rate_pseudo_root_jacobian_all_elements(ref: dict[str, Any]) -> None:
    """Full element-wise Jacobian matches C++ (TIGHT)."""
    pr, old, new, g, dfs, taus, disp = _inputs(ref)
    alive = 0
    jac = RatePseudoRootJacobianAllElements(pr, alive, alive, taus, disp)
    b_out = [
        np.zeros((N_RATES, N_FACTORS), dtype=np.float64) for _ in range(N_RATES)
    ]
    jac.get_bumps(old, dfs, new, g, b_out)

    expected = reshape_stack(ref["jac_B_all"], N_RATES, N_RATES, N_FACTORS)
    for rate in range(N_RATES):
        for k in range(N_RATES):
            for f in range(N_FACTORS):
                tight(float(b_out[rate][k][f]), float(expected[rate][k][f]))


def test_rate_pseudo_root_jacobian_numerical(ref: dict[str, Any]) -> None:
    """Finite-difference (Numerical) ``B`` matrix matches C++ (TIGHT).

    The numerical class re-evolves the bumped pseudo-root via the LMM drift
    calculator, so this exercises the whole drift -> log-Euler kernel under a
    full unit bump; it cross-validates bit-for-bit against C++.
    """
    pr, old, new, g, dfs, taus, disp = _inputs(ref)
    alive = 0
    n_bumps = ref["jac_n_bumps"]
    num = RatePseudoRootJacobianNumerical(pr, alive, alive, taus, _bumps(), disp)
    b_out = np.zeros((n_bumps, N_RATES), dtype=np.float64)
    num.get_bumps(old, dfs, new, g, b_out)

    expected = reshape(ref["jac_B_numerical"], n_bumps, N_RATES)
    for i in range(n_bumps):
        for j in range(N_RATES):
            tight(float(b_out[i][j]), float(expected[i][j]))


def test_analytic_vs_numerical_consistency(ref: dict[str, Any]) -> None:
    """Analytic Jacobian matches a centered finite difference (LOOSE).

    The C++ ``testPathwiseVegas`` validates the analytic Jacobian against the
    finite-difference bump of the one-step evolution. We reproduce that here
    with a *centered* difference of the evolved rate w.r.t. each pseudo-root
    element: ``(R(+eps) - R(-eps)) / (2*eps) ~= dR/d(pseudoRoot element)``.

    Because the analytic Jacobian's ``discountRatios`` term assumes the *exact*
    one-step discount ratios, we drive the difference off the genuine evolved
    rates (a zero-bump Numerical pass) rather than the synthetic ``new_rates``
    used for the bit-exact ``getBumps`` cross-validation above.
    """
    pr, old, _new, g, dfs, taus, disp = _inputs(ref)
    alive = 0
    n_bumps = ref["jac_n_bumps"]
    spots = [(0, 0), (2, 1), (4, 2), (5, 0)]

    # true one-step evolved rates under the *unbumped* pseudo-root: run the
    # Numerical class with a zero bump and read bumpedRates = evolved - 0.
    zero_bump = [np.zeros((N_RATES, N_FACTORS), dtype=np.float64)]
    base = RatePseudoRootJacobianNumerical(pr, alive, alive, taus, zero_bump, disp)
    zero_out = np.zeros((1, N_RATES), dtype=np.float64)
    # pass new_rates = 0 so the result IS the evolved rate
    base.get_bumps(old, dfs, [0.0] * N_RATES, g, zero_out)
    evolved = [float(zero_out[0][j]) for j in range(N_RATES)]

    # analytic Jacobian evaluated at the true evolved rates
    jac = RatePseudoRootJacobian(pr, alive, alive, taus, _bumps(), disp)
    b_an = np.zeros((n_bumps, N_RATES), dtype=np.float64)
    jac.get_bumps(old, dfs, evolved, g, b_an)

    eps = 1e-6
    for i, (r, f) in enumerate(spots):
        up = pr.copy()
        up[r][f] += eps
        dn = pr.copy()
        dn[r][f] -= eps
        # re-evolve with +eps / -eps and read evolved rate (new_rates = 0)
        num_up = RatePseudoRootJacobianNumerical(up, alive, alive, taus, zero_bump, disp)
        num_dn = RatePseudoRootJacobianNumerical(dn, alive, alive, taus, zero_bump, disp)
        o_up = np.zeros((1, N_RATES), dtype=np.float64)
        o_dn = np.zeros((1, N_RATES), dtype=np.float64)
        num_up.get_bumps(old, dfs, [0.0] * N_RATES, g, o_up)
        num_dn.get_bumps(old, dfs, [0.0] * N_RATES, g, o_dn)
        for j in range(N_RATES):
            fd = (float(o_up[0][j]) - float(o_dn[0][j])) / (2.0 * eps)
            analytic = float(b_an[i][j])
            assert abs(fd - analytic) <= 1e-5 + 1e-4 * abs(analytic)

