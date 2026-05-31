"""Tests for SwaptionPseudoDerivative + CapPseudoDerivative (W11-D batch b).

Cross-validates against ``migration-harness/references/cluster/w11d.json``.

The pseudo-root derivatives are *linear* in the pseudo-root, hence sensitive to
the per-column spectral eigenvector sign — which C++'s Jacobi
``SymmetricSchurDecomposition`` and numpy's LAPACK ``eigh`` pin differently for
the dead-rate-truncated covariance matrices (``B @ B.T`` is identical; see
``align(pseudo_sqrt)`` W10-B). We therefore cross-validate the signed
derivatives against the *exact C++ pseudo-roots* fed through a
``PseudoRootFacade`` (isolating the derivative algorithm), and additionally
confirm against the natively-built ``FlatVol`` that the sign-invariant scalars
(variance, implied vol, expiry) match and the derivative *magnitudes* agree.

C++ parity:
  ql/models/marketmodels/pathwisegreeks/swaptionpseudojacobian.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pquantlib.models.marketmodels.pathwisegreeks.swaption_pseudo_jacobian import (
    CapPseudoDerivative,
    SwaptionPseudoDerivative,
)
from pquantlib.testing.tolerance import tight

from .conftest import N_FACTORS, N_RATES, make_facade_setup, make_setup, reshape_stack


def test_swaption_pseudo_derivative_scalars_facade(ref: dict[str, Any]) -> None:
    """variance / implied vol / expiry match C++ (TIGHT)."""
    facade = make_facade_setup(ref["pseudo_roots"])
    sp = SwaptionPseudoDerivative(facade, ref["swpt_start"], ref["swpt_end"])
    tight(sp.variance(), ref["swpt_variance"])
    tight(sp.implied_volatility(), ref["swpt_implied_vol"])
    tight(sp.expiry(), ref["swpt_expiry"])


def test_swaption_pseudo_derivative_matrices_facade(ref: dict[str, Any]) -> None:
    """variance + volatility derivative matrices match C++ signed (TIGHT).

    Driven by the exact C++ pseudo-roots (via the facade), so the sign is
    pinned to C++ and the full signed Jacobian cross-validates bit-for-bit.
    """
    facade = make_facade_setup(ref["pseudo_roots"])
    sp = SwaptionPseudoDerivative(facade, ref["swpt_start"], ref["swpt_end"])
    steps = ref["n_steps"]
    cpp_var = reshape_stack(ref["swpt_variance_derivs"], steps, N_RATES, N_FACTORS)
    cpp_vol = reshape_stack(ref["swpt_volatility_derivs"], steps, N_RATES, N_FACTORS)
    for i in range(steps):
        var_d = np.asarray(sp.variance_derivative(i))
        vol_d = np.asarray(sp.volatility_derivative(i))
        for r in range(N_RATES):
            for f in range(N_FACTORS):
                tight(float(var_d[r][f]), float(cpp_var[i][r][f]))
                tight(float(vol_d[r][f]), float(cpp_vol[i][r][f]))


def test_swaption_pseudo_derivative_flatvol_magnitude(ref: dict[str, Any]) -> None:
    """Native FlatVol: scalars match; derivative magnitudes match C++ (TIGHT).

    The natively-built FlatVol uses numpy's spectral signs, which differ from
    C++ per (step, factor) column. The sign-invariant scalars match exactly and
    the element-wise magnitudes equal C++'s, confirming the only divergence is
    the documented spectral sign.
    """
    model, _ = make_setup()
    sp = SwaptionPseudoDerivative(model, ref["swpt_start"], ref["swpt_end"])
    tight(sp.variance(), ref["swpt_variance"])
    tight(sp.implied_volatility(), ref["swpt_implied_vol"])

    steps = ref["n_steps"]
    cpp_var = reshape_stack(ref["swpt_variance_derivs"], steps, N_RATES, N_FACTORS)
    cpp_vol = reshape_stack(ref["swpt_volatility_derivs"], steps, N_RATES, N_FACTORS)
    for i in range(steps):
        var_d = np.abs(np.asarray(sp.variance_derivative(i)))
        vol_d = np.abs(np.asarray(sp.volatility_derivative(i)))
        for r in range(N_RATES):
            for f in range(N_FACTORS):
                tight(float(var_d[r][f]), abs(float(cpp_var[i][r][f])))
                tight(float(vol_d[r][f]), abs(float(cpp_vol[i][r][f])))


def test_cap_pseudo_derivative_scalars_facade(ref: dict[str, Any]) -> None:
    """Cap implied vol (Brent solve over sum-of-caplets) matches C++ (TIGHT)."""
    facade = make_facade_setup(ref["pseudo_roots"])
    cp = CapPseudoDerivative(
        facade,
        ref["cap_strike"],
        ref["cap_start"],
        ref["cap_end"],
        ref["cap_first_df"],
    )
    tight(cp.implied_volatility(), ref["cap_implied_vol"])


def test_cap_pseudo_derivative_matrices_facade(ref: dict[str, Any]) -> None:
    """Cap price + volatility derivative matrices match C++ signed (TIGHT)."""
    facade = make_facade_setup(ref["pseudo_roots"])
    cp = CapPseudoDerivative(
        facade,
        ref["cap_strike"],
        ref["cap_start"],
        ref["cap_end"],
        ref["cap_first_df"],
    )
    steps = ref["n_steps"]
    cpp_pd = reshape_stack(ref["cap_price_derivs"], steps, N_RATES, N_FACTORS)
    cpp_vd = reshape_stack(ref["cap_volatility_derivs"], steps, N_RATES, N_FACTORS)
    for i in range(steps):
        price_d = np.asarray(cp.price_derivative(i))
        vol_d = np.asarray(cp.volatility_derivative(i))
        for r in range(N_RATES):
            for f in range(N_FACTORS):
                tight(float(price_d[r][f]), float(cpp_pd[i][r][f]))
                tight(float(vol_d[r][f]), float(cpp_vd[i][r][f]))


def test_cap_pseudo_derivative_flatvol_magnitude(ref: dict[str, Any]) -> None:
    """Native FlatVol cap: implied vol matches; derivative magnitudes match."""
    model, _ = make_setup()
    cp = CapPseudoDerivative(
        model,
        ref["cap_strike"],
        ref["cap_start"],
        ref["cap_end"],
        ref["cap_first_df"],
    )
    tight(cp.implied_volatility(), ref["cap_implied_vol"])

    steps = ref["n_steps"]
    cpp_pd = reshape_stack(ref["cap_price_derivs"], steps, N_RATES, N_FACTORS)
    cpp_vd = reshape_stack(ref["cap_volatility_derivs"], steps, N_RATES, N_FACTORS)
    for i in range(steps):
        price_d = np.abs(np.asarray(cp.price_derivative(i)))
        vol_d = np.abs(np.asarray(cp.volatility_derivative(i)))
        for r in range(N_RATES):
            for f in range(N_FACTORS):
                tight(float(price_d[r][f]), abs(float(cpp_pd[i][r][f])))
                tight(float(vol_d[r][f]), abs(float(cpp_vd[i][r][f])))
