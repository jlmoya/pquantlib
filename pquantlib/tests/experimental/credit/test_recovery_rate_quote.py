"""Cross-validate RecoveryRateQuote against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.default_type import Seniority
from pquantlib.experimental.credit.recovery_rate_quote import RecoveryRateQuote
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def test_conventional_recovery_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["isda_conv_recoveries"]
    # TIGHT: closed-form scalar lookup.
    tolerance.tight(RecoveryRateQuote.conventional_recovery(Seniority.SecDom), ref["secdom"])
    tolerance.tight(RecoveryRateQuote.conventional_recovery(Seniority.SnrFor), ref["snrfor"])
    tolerance.tight(RecoveryRateQuote.conventional_recovery(Seniority.SubLT2), ref["sublt2"])
    tolerance.tight(RecoveryRateQuote.conventional_recovery(Seniority.JrSubT2), ref["jrsubt2"])
    tolerance.tight(RecoveryRateQuote.conventional_recovery(Seniority.PrefT1), ref["preft1"])


def test_recovery_rate_quote_basics(cpp_ref: dict[str, Any]) -> None:
    q = RecoveryRateQuote(0.42, Seniority.SnrFor)
    ref = cpp_ref["recovery_rate_quote"]
    tolerance.tight(q.value(), ref["value"])
    assert int(q.seniority()) == ref["seniority_idx"]
    assert q.is_valid() == ref["is_valid"]


def test_recovery_rate_quote_invalid_initial() -> None:
    q = RecoveryRateQuote()
    assert not q.is_valid()
    with pytest.raises(LibraryException, match="invalid Recovery"):
        q.value()


def test_recovery_rate_quote_set_value_diff() -> None:
    q = RecoveryRateQuote(0.5, Seniority.SnrFor)
    diff = q.set_value(0.6)
    tolerance.tight(diff, 0.1)
    tolerance.tight(q.value(), 0.6)
    # Non-change returns 0
    diff = q.set_value(0.6)
    assert diff == 0.0


def test_recovery_rate_quote_reset_drops_seniority() -> None:
    q = RecoveryRateQuote(0.5, Seniority.SnrFor)
    q.reset()
    assert not q.is_valid()
    assert q.seniority() == Seniority.NoSeniority


def test_recovery_rate_quote_rejects_out_of_range() -> None:
    with pytest.raises(LibraryException, match="fractional unit"):
        RecoveryRateQuote(1.5, Seniority.SnrFor)
    with pytest.raises(LibraryException, match="fractional unit"):
        RecoveryRateQuote(-0.1, Seniority.SnrFor)


def test_recovery_rate_quote_set_value_to_none_invalidates() -> None:
    q = RecoveryRateQuote(0.5, Seniority.SnrFor)
    assert q.is_valid()
    q.set_value(None)
    assert not q.is_valid()
