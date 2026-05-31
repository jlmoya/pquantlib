"""Tests for the W10-B SquareRootAndersen vol process (batch c).

Cross-validates against ``migration-harness/references/cluster/w10b.json``. The
QE discretization is fed FIXED variates (so the draws are deterministic), so the
sub-step states + stepSd match C++ TIGHT regardless of any RNG.

C++ parity:
  ql/models/marketmodels/evolvers/marketmodelvolprocess.hpp
  ql/models/marketmodels/evolvers/volprocesses/squarerootandersen.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.models.marketmodels.evolvers.volprocesses.square_root_andersen import (
    SquareRootAndersen,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10b")


def _make_sra() -> SquareRootAndersen:
    evol_times = [0.5 * (i + 1) for i in range(5)]
    return SquareRootAndersen(0.04, 1.0, 0.3, 0.04, evol_times, 2, 0.5, 0.5, 1.5)


def test_sra_metadata(ref: dict[str, Any]) -> None:
    sra = _make_sra()
    assert sra.variates_per_step() == int(ref["sra_variates_per_step"])
    assert sra.number_state_variables() == int(ref["sra_number_state_vars"])


def test_sra_quadratic_branch_steps(ref: dict[str, Any]) -> None:
    # steps 1 & 2 use the quadratic (psi <= cut) branch of the QE scheme.
    sra = _make_sra()
    sra.next_path()

    w1 = sra.next_step([0.3, -0.5])
    tight(w1, ref["sra_step1_weight"])
    tight(sra.step_sd(), ref["sra_step1_sd"])
    tight(sra.state_variables()[0], ref["sra_step1_state"])

    w2 = sra.next_step([-1.0, 0.8])
    tight(w2, ref["sra_step2_weight"])
    tight(sra.step_sd(), ref["sra_step2_sd"])
    tight(sra.state_variables()[0], ref["sra_step2_state"])


def test_sra_exponential_branch_step(ref: dict[str, Any]) -> None:
    # step 3 drives the psi > cut (exponential) branch + the u < p -> v=0 arm.
    sra = _make_sra()
    sra.next_path()
    sra.next_step([0.3, -0.5])
    sra.next_step([-1.0, 0.8])

    w3 = sra.next_step([-3.0, -3.0])
    tight(w3, ref["sra_step3_weight"])
    tight(sra.step_sd(), ref["sra_step3_sd"])
    tight(sra.state_variables()[0], ref["sra_step3_state"])
