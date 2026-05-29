"""Cross-validate ConstantRecoveryModel against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import Seniority
from pquantlib.experimental.credit.recovery_rate_model import ConstantRecoveryModel
from pquantlib.experimental.credit.recovery_rate_quote import RecoveryRateQuote
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def test_constant_recovery_model_from_rate_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    rm = ConstantRecoveryModel.from_rate(0.37, Seniority.SnrFor)
    today = Date.from_ymd(15, Month.January, 2024)
    r = rm.recovery_value(today)
    assert r is not None
    # EXACT: scalar stored verbatim.
    tolerance.exact(r, cpp_ref["constant_recovery_model_real_ctor"])


def test_constant_recovery_model_from_quote_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    quote = RecoveryRateQuote(0.55, Seniority.SubLT2)
    rm = ConstantRecoveryModel(quote)
    today = Date.from_ymd(15, Month.January, 2024)
    r = rm.recovery_value(today)
    assert r is not None
    tolerance.tight(r, cpp_ref["constant_recovery_model_quote_ctor"])


def test_constant_recovery_model_applies_to_any_seniority() -> None:
    rm = ConstantRecoveryModel.from_rate(0.4, Seniority.SnrFor)
    for sen in (
        Seniority.SecDom,
        Seniority.SnrFor,
        Seniority.SubLT2,
        Seniority.JrSubT2,
        Seniority.PrefT1,
        Seniority.NoSeniority,
    ):
        assert rm.applies_to_seniority(sen) is True


def test_constant_recovery_model_passes_through_quote_invalidation() -> None:
    quote = RecoveryRateQuote(0.4, Seniority.SnrFor)
    rm = ConstantRecoveryModel(quote)
    today = Date.from_ymd(15, Month.January, 2024)
    assert rm.recovery_value(today) is not None
    quote.reset()
    assert rm.recovery_value(today) is None


def test_constant_recovery_model_with_explicit_default_key() -> None:
    rm = ConstantRecoveryModel.from_rate(0.42, Seniority.SnrFor)
    today = Date.from_ymd(15, Month.January, 2024)
    # # C++ parity: recoveryValueImpl ignores the key argument.
    r = rm.recovery_value(today, DefaultProbKey())
    assert r is not None
    tolerance.exact(r, 0.42)
