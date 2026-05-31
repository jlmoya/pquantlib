"""Tests for VegaBumpCluster/Collection + instrument jacobian (W11-D batch c).

Cross-validates against ``migration-harness/references/cluster/w11d.json``.

The clustering schema (intersection / compatibility / coverage booleans + bump
counts) is sign-invariant and tested against the natively-built FlatVol. The
instrument-level vega-bump Jacobian and the orthogonalised bumps are linear in
the pseudo-root (sign-sensitive), so they are cross-validated signed against the
exact C++ pseudo-roots via a ``PseudoRootFacade`` (see the swaption test module
for the spectral-sign rationale).

C++ parity:
  ql/models/marketmodels/pathwisegreeks/vegabumpcluster.{hpp,cpp}
  ql/models/marketmodels/pathwisegreeks/bumpinstrumentjacobian.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pquantlib.models.marketmodels.pathwisegreeks.bump_instrument_jacobian import (
    Cap,
    OrthogonalizedBumpFinder,
    Swaption,
    VolatilityBumpInstrumentJacobian,
)
from pquantlib.models.marketmodels.pathwisegreeks.vega_bump_cluster import (
    VegaBumpCluster,
    VegaBumpCollection,
)
from pquantlib.testing.tolerance import tight

from .conftest import N_FACTORS, N_RATES, make_facade_setup, make_setup, reshape, reshape_stack


def test_vega_bump_cluster_intersection_compatibility(ref: dict[str, Any]) -> None:
    """doesIntersect / isCompatible reproduce C++ (TIGHT — deterministic)."""
    model, _ = make_setup()
    a = VegaBumpCluster(0, 2, 1, 3, 0, 1)
    b = VegaBumpCluster(1, 3, 2, 4, 0, 1)
    c = VegaBumpCluster(0, 1, 4, 5, 2, 3)
    big = VegaBumpCluster(0, 1, 0, 100, 0, 1)

    assert a.does_intersect(b) is ref["clus_a_intersect_b"]
    assert a.does_intersect(c) is ref["clus_a_intersect_c"]
    assert b.does_intersect(c) is ref["clus_b_intersect_c"]
    assert a.is_compatible(model) is ref["clus_a_compatible"]
    assert big.is_compatible(model) is ref["clus_big_compatible"]


def test_vega_bump_collection_factorwise(ref: dict[str, Any]) -> None:
    """Factor-wise collection: count + coverage booleans match C++ (TIGHT)."""
    model, _ = make_setup()
    coll = VegaBumpCollection(model, True)
    assert coll.number_bumps() == ref["coll_factor_numberBumps"]
    assert coll.is_full() is ref["coll_factor_isFull"]
    assert coll.is_non_overlapping() is ref["coll_factor_isNonOverlapping"]
    assert coll.is_sensible() is ref["coll_factor_isSensible"]


def test_vega_bump_collection_whole(ref: dict[str, Any]) -> None:
    """Non-factor-wise collection count matches C++ (TIGHT)."""
    model, _ = make_setup()
    coll = VegaBumpCollection(model, False)
    assert coll.number_bumps() == ref["coll_whole_numberBumps"]


def test_vega_bump_collection_manual(ref: dict[str, Any]) -> None:
    """Manually-built collection: count + coverage quirk match C++ (TIGHT).

    Built via ``from_clusters`` (``checked_ = False``) so ``is_full`` /
    ``is_non_overlapping`` run the real computation. The collection covers every
    alive element exactly once, so (per the C++ ``numberFailures > 0`` quirk)
    both return ``False``.
    """
    model, ev = make_setup()
    bumps = [
        VegaBumpCluster(0, N_FACTORS, r, r + 1, st, st + 1)
        for st in range(ev.number_of_steps())
        for r in range(ev.first_alive_rate()[st], N_RATES)
    ]
    coll = VegaBumpCollection.from_clusters(bumps, model)
    assert coll.number_bumps() == ref["coll_manual_numberBumps"]
    assert coll.is_full() is ref["coll_manual_isFull"]
    assert coll.is_non_overlapping() is ref["coll_manual_isNonOverlapping"]


def test_instrument_jacobian_derivatives_facade(ref: dict[str, Any]) -> None:
    """derivativesVolatility (swaption + cap) match C++ signed (TIGHT)."""
    facade = make_facade_setup(ref["pseudo_roots"])
    bumps = VegaBumpCollection(facade, False)
    assert bumps.number_bumps() == ref["inst_numberBumps"]

    jac = VolatilityBumpInstrumentJacobian(bumps, [Swaption(1, 5)], [Cap(1, 5, 0.04)])

    dv_swaption = jac.derivatives_volatility(0)
    dv_cap = jac.derivatives_volatility(1)
    cpp_swaption = ref["inst_deriv_swaption"]
    cpp_cap = ref["inst_deriv_cap"]
    for k in range(bumps.number_bumps()):
        tight(dv_swaption[k], cpp_swaption[k])
        tight(dv_cap[k], cpp_cap[k])


def test_instrument_jacobian_one_percent_bumps_facade(ref: dict[str, Any]) -> None:
    """onePercentBump + getAllOnePercentBumps match C++ signed (TIGHT)."""
    facade = make_facade_setup(ref["pseudo_roots"])
    bumps = VegaBumpCollection(facade, False)
    jac = VolatilityBumpInstrumentJacobian(bumps, [Swaption(1, 5)], [Cap(1, 5, 0.04)])

    op_swaption = jac.one_percent_bump(0)
    op_cap = jac.one_percent_bump(1)
    cpp_op_swaption = ref["inst_onepct_swaption"]
    cpp_op_cap = ref["inst_onepct_cap"]
    for k in range(bumps.number_bumps()):
        tight(op_swaption[k], cpp_op_swaption[k])
        tight(op_cap[k], cpp_op_cap[k])

    all_bumps = np.asarray(jac.get_all_one_percent_bumps())
    cpp_all = reshape(ref["inst_all_onepct"], 2, bumps.number_bumps())
    for i in range(2):
        for k in range(bumps.number_bumps()):
            tight(float(all_bumps[i][k]), float(cpp_all[i][k]))


def test_orthogonalized_bump_finder_facade(ref: dict[str, Any]) -> None:
    """Orthogonalised pseudo-root bumps match C++ signed (TIGHT).

    The orthogonalised bump directions reproduce the target instrument vegas;
    against the exact C++ pseudo-roots they match bit-for-bit. The step/bump
    grid shape (and the number of *valid* — non-discarded — bumps) matches C++.
    """
    facade = make_facade_setup(ref["pseudo_roots"])
    bumps = VegaBumpCollection(facade, False)
    finder = OrthogonalizedBumpFinder(
        bumps, [Swaption(1, 5)], [Cap(1, 5, 0.04)], 100.0, 1e-8
    )
    the_bumps = finder.get_vega_bumps()

    assert len(the_bumps) == ref["orth_n_steps"]
    assert len(the_bumps[0]) == ref["orth_n_bumps"]

    flat = [m for per_step in the_bumps for m in per_step]
    cpp_orth = reshape_stack(ref["orth_bumps"], len(flat), N_RATES, N_FACTORS)
    for i in range(len(flat)):
        mat = np.asarray(flat[i])
        for r in range(N_RATES):
            for f in range(N_FACTORS):
                tight(float(mat[r][f]), float(cpp_orth[i][r][f]))
