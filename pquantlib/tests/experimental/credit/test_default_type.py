"""Cross-validate DefaultType / Seniority / AtomicDefault / Restructuring enums.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    FailureToPay,
    Restructuring,
    Seniority,
)
from pquantlib.testing import reference_reader
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def test_seniority_enum_indices_match_cpp(cpp_ref: dict[str, Any]) -> None:
    # Probe uses SnrFor (=1) as seniority on the default event.
    assert int(Seniority.SnrFor) == cpp_ref["default_event"]["seniority_idx"]
    # Markit aliases must land on the canonical underlying value.
    assert Seniority.SeniorSec == Seniority.SecDom  # type: ignore[attr-defined]
    assert Seniority.SeniorUnSec == Seniority.SnrFor  # type: ignore[attr-defined]
    assert Seniority.SubTier1 == Seniority.PrefT1  # type: ignore[attr-defined]
    assert Seniority.SubUpperTier2 == Seniority.JrSubT2  # type: ignore[attr-defined]
    assert Seniority.SubLoweTier2 == Seniority.SubLT2  # type: ignore[attr-defined]


def test_atomic_default_enum_indices_match_cpp(cpp_ref: dict[str, Any]) -> None:
    # Probe uses Bankruptcy (=1) as the default type on the event.
    assert int(AtomicDefault.Bankruptcy) == cpp_ref["default_event"]["default_type_idx"]
    # Synonyms.
    assert AtomicDefault.ObligationAcceleration == AtomicDefault.Acceleration  # type: ignore[attr-defined]
    assert AtomicDefault.ObligationDefault == AtomicDefault.Default  # type: ignore[attr-defined]
    assert AtomicDefault.CrossDefault == AtomicDefault.Default  # type: ignore[attr-defined]


def test_restructuring_enum_indices_match_cpp(cpp_ref: dict[str, Any]) -> None:
    # Probe uses XR (=NoRestructuring=0) on the event.
    assert (
        int(Restructuring.NoRestructuring)
        == cpp_ref["default_event"]["restructuring_type_idx"]
    )
    # Markit aliases.
    assert Restructuring.NoRestructuring == Restructuring.XR  # type: ignore[attr-defined]
    assert Restructuring.ModifiedRestructuring == Restructuring.MR  # type: ignore[attr-defined]
    assert Restructuring.ModifiedModifiedRestructuring == Restructuring.MM  # type: ignore[attr-defined]
    assert Restructuring.FullRestructuring == Restructuring.CR  # type: ignore[attr-defined]


def test_default_type_basics() -> None:
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    assert dt.default_type == AtomicDefault.Bankruptcy
    assert dt.restructuring_type == Restructuring.NoRestructuring
    assert not dt.is_restructuring()
    assert dt.contains_default_type(AtomicDefault.Bankruptcy)
    assert not dt.contains_default_type(AtomicDefault.FailureToPay)
    # AnyRestructuring is wildcard.
    assert dt.contains_restructuring_type(Restructuring.NoRestructuring)
    assert dt.contains_restructuring_type(Restructuring.AnyRestructuring)
    assert not dt.contains_restructuring_type(Restructuring.FullRestructuring)


def test_default_type_with_restructuring() -> None:
    dt = DefaultType(AtomicDefault.Restructuring, Restructuring.FullRestructuring)
    assert dt.is_restructuring()
    assert dt.contains_restructuring_type(Restructuring.FullRestructuring)


def test_default_type_equality() -> None:
    a = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    b = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    c = DefaultType(AtomicDefault.FailureToPay, Restructuring.NoRestructuring)
    assert a == b
    assert a != c
    # Hashable.
    assert hash(a) == hash(b)


def test_failure_to_pay_subclass() -> None:
    ftp = FailureToPay(grace_period=Period(30, TimeUnit.Days), amount_required=1.0e6)
    assert ftp.default_type == AtomicDefault.FailureToPay
    assert ftp.restructuring_type == Restructuring.NoRestructuring
    assert ftp.grace_period == Period(30, TimeUnit.Days)
    assert ftp.amount_required == 1.0e6
    assert not ftp.is_restructuring()


def test_failure_to_pay_default_values() -> None:
    ftp = FailureToPay()
    assert ftp.default_type == AtomicDefault.FailureToPay
    # Default ``grace_period`` is a null Period(0, Days) ≡ Period().
    assert ftp.grace_period == Period()
    assert ftp.amount_required == 1.0e6
