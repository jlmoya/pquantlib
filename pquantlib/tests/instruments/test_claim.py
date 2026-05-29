"""Cross-validate Claim hierarchy against C++.

Probe source: migration-harness/cpp/probes/cluster_l8b/probe.cpp
Reference:    migration-harness/references/cluster/l8b.json (key: "face_value_claim")
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.instruments.claim import FaceValueClaim
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/l8b")["face_value_claim"]


def test_face_value_claim_amount(cpp_ref: dict[str, Any]) -> None:
    claim = FaceValueClaim()
    d = Date.from_ymd(15, Month.June, 2026)
    # N=10M, recovery=40% → 10M * 0.6 = 6M.
    # EXACT: closed-form multiplication of finite floats.
    tolerance.exact(claim.amount(d, 10_000_000.0, 0.4), cpp_ref["amount_N10M_R40"])


def test_face_value_claim_independent_of_date() -> None:
    """FaceValueClaim ignores the default date."""
    claim = FaceValueClaim()
    d1 = Date.from_ymd(15, Month.June, 2026)
    d2 = Date.from_ymd(31, Month.December, 2030)
    assert claim.amount(d1, 10_000_000.0, 0.4) == claim.amount(d2, 10_000_000.0, 0.4)


def test_face_value_claim_zero_recovery() -> None:
    claim = FaceValueClaim()
    d = Date.from_ymd(15, Month.June, 2026)
    # 100% loss → claim = notional.
    assert claim.amount(d, 1_000_000.0, 0.0) == 1_000_000.0


def test_face_value_claim_full_recovery() -> None:
    claim = FaceValueClaim()
    d = Date.from_ymd(15, Month.June, 2026)
    # 100% recovery → claim = 0.
    assert claim.amount(d, 1_000_000.0, 1.0) == 0.0
