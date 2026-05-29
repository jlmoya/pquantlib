"""Cross-validate Pool / HomogeneousPool / InhomogeneousPool.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    Restructuring,
    Seniority,
)
from pquantlib.experimental.credit.homogeneous_pool import HomogeneousPool
from pquantlib.experimental.credit.inhomogeneous_pool import InhomogeneousPool
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.experimental.credit.pool import Pool
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def _make_issuer() -> tuple[Issuer, DefaultProbKey]:
    usd = USDCurrency()
    today = Date.from_ymd(15, Month.January, 2024)
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    et = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(et,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    return Issuer(probabilities=[(key, curve)]), key


def test_pool_size_names_lookup_match_cpp(cpp_ref: dict[str, Any]) -> None:
    issuer, key = _make_issuer()
    pool = Pool()
    pool.add("AcmeCorp", issuer, key)
    pool.add("Globex", issuer, key)
    pool.set_time("AcmeCorp", 2.5)

    ref = cpp_ref["pool"]
    assert pool.size() == ref["size"]
    assert pool.has("AcmeCorp") == ref["has_acme"]
    assert pool.has("Unknown") == ref["has_unknown"]
    tolerance.tight(pool.get_time("AcmeCorp"), ref["time_acme"])
    tolerance.tight(pool.get_time("Globex"), ref["time_globex"])
    assert pool.names()[0] == ref["names_first"]
    assert pool.names()[1] == ref["names_second"]


def test_pool_add_is_noop_for_duplicate_names() -> None:
    issuer, key = _make_issuer()
    other_issuer = Issuer()
    pool = Pool()
    pool.add("AcmeCorp", issuer, key)
    pool.add("AcmeCorp", other_issuer, key)  # no-op
    assert pool.size() == 1
    # The original issuer is still registered (not overwritten).
    assert pool.get("AcmeCorp") is issuer


def test_pool_get_missing_raises() -> None:
    pool = Pool()
    with pytest.raises(LibraryException, match="not found"):
        pool.get("Unknown")
    with pytest.raises(LibraryException, match="not found"):
        pool.default_key("Unknown")
    with pytest.raises(LibraryException, match="not found"):
        pool.get_time("Unknown")


def test_pool_clear_drops_all() -> None:
    issuer, key = _make_issuer()
    pool = Pool()
    pool.add("AcmeCorp", issuer, key)
    pool.add("Globex", issuer, key)
    assert pool.size() == 2
    pool.clear()
    assert pool.size() == 0
    assert pool.names() == []


def test_pool_default_keys_iteration() -> None:
    issuer, key = _make_issuer()
    pool = Pool()
    pool.add("AcmeCorp", issuer, key)
    pool.add("Globex", issuer, key)
    keys = pool.default_keys()
    assert len(keys) == 2
    assert keys[0] == key
    assert keys[1] == key


def test_homogeneous_pool_scalar_attributes() -> None:
    pool = HomogeneousPool(uniform_notional=2.5, uniform_recovery_rate=0.35)
    tolerance.exact(pool.uniform_notional(), 2.5)
    tolerance.exact(pool.uniform_recovery_rate(), 0.35)

    issuer, key = _make_issuer()
    pool.add("AcmeCorp", issuer, key)
    pool.add("Globex", issuer, key)
    assert pool.notionals() == [2.5, 2.5]
    assert pool.recovery_rates() == [0.35, 0.35]


def test_inhomogeneous_pool_arrays() -> None:
    pool = InhomogeneousPool()
    issuer, key = _make_issuer()
    pool.add_with_attributes("AcmeCorp", issuer, 100.0, 0.4, key)
    pool.add_with_attributes("Globex", issuer, 200.0, 0.3, key)
    assert pool.size() == 2
    assert pool.notionals() == [100.0, 200.0]
    assert pool.recovery_rates() == [0.4, 0.3]


def test_inhomogeneous_pool_duplicate_add_is_noop() -> None:
    pool = InhomogeneousPool()
    issuer, key = _make_issuer()
    pool.add_with_attributes("AcmeCorp", issuer, 100.0, 0.4, key)
    pool.add_with_attributes("AcmeCorp", issuer, 999.0, 0.5, key)  # ignored
    assert pool.size() == 1
    assert pool.notionals() == [100.0]
    assert pool.recovery_rates() == [0.4]


def test_inhomogeneous_pool_clear() -> None:
    pool = InhomogeneousPool()
    issuer, key = _make_issuer()
    pool.add_with_attributes("AcmeCorp", issuer, 100.0, 0.4, key)
    assert pool.size() == 1
    pool.clear()
    assert pool.size() == 0
    assert pool.notionals() == []
    assert pool.recovery_rates() == []


def test_pool_default_trigger_falls_back_to_empty_key() -> None:
    # If caller doesn't pass a contract_trigger, an empty DefaultProbKey
    # is registered. This is the foundation for downstream basket code
    # that overrides per-name keys later.
    pool = Pool()
    pool.add("AcmeCorp", Issuer())
    key = pool.default_key("AcmeCorp")
    assert key.size() == 0
    assert key.seniority == Seniority.NoSeniority
