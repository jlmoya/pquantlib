"""Cross-validate Loss against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.credit.loss import Loss
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def test_loss_round_trips_match_cpp(cpp_ref: dict[str, Any]) -> None:
    l1 = Loss(amount=100.0, time=2.5)
    l2 = Loss(amount=50.0, time=3.0)
    l3 = Loss(amount=999.0, time=2.5)

    ref = cpp_ref["loss"]
    tolerance.tight(l1.time, ref["l1_time"])
    tolerance.tight(l1.amount, ref["l1_amount"])
    assert (l1 < l2) == ref["l1_lt_l2"]
    # # C++ parity: operator== compares time only — divergent amounts still equal.
    assert (l1 == l3) == ref["l1_eq_l3"]


def test_loss_hash_by_time() -> None:
    l1 = Loss(amount=100.0, time=2.5)
    l2 = Loss(amount=200.0, time=2.5)
    # Same time → same hash (and equal).
    assert hash(l1) == hash(l2)
    assert l1 == l2


def test_loss_ne_returns_negation_of_eq() -> None:
    l1 = Loss(amount=100.0, time=2.5)
    l2 = Loss(amount=200.0, time=2.6)
    assert l1 != l2
    eq = l1 == l2
    assert not eq


def test_loss_optional_name_field() -> None:
    loss = Loss(amount=100.0, time=2.5, name="AcmeCorp")
    assert loss.name == "AcmeCorp"
    loss2 = Loss(amount=100.0, time=2.5)
    assert loss2.name is None
