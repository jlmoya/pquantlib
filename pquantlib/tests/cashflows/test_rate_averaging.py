"""Tests for the RateAveraging enum.

The ordinal values matter: C++ code (and any reference JSON that ever emits
one) uses the underlying int, and ``RateAveraging::Compound`` is the default
averaging method throughout QuantLib's overnight machinery.
"""

from __future__ import annotations

from pquantlib.cashflows.rate_averaging import RateAveraging


def test_ordinals_match_cpp() -> None:
    """C++ ql/cashflows/rateaveraging.hpp declares ``{ Simple, Compound }``."""
    assert int(RateAveraging.Simple) == 0
    assert int(RateAveraging.Compound) == 1


def test_members_are_exhaustive() -> None:
    assert [m.name for m in RateAveraging] == ["Simple", "Compound"]
