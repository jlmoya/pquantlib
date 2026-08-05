"""Cross-validate Histogram against the C++ probe.

Reference: ``migration-harness/references/v143/math/statistics.json`` —
``histogram``. Every bin rule is pinned, not just one: Sturges, FD and
Scott disagree about the bin count on the same 50-point data set, and the
FD rule in particular depends on QuantLib's own Hyndman-Fan type-8
quantile rather than on any library default.

The ``explicit_breaks`` case pins a genuine quirk: ``bins`` is fixed at
``len(breaks) + 1`` before the near-duplicate break points are collapsed
with ``close_enough``, so the de-duplicated input leaves a permanently
empty bin. The port reproduces it rather than correcting it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.statistics.histogram import Histogram
from pquantlib.testing import reference_reader, tolerance

RULE_CASES: dict[str, Callable[[list[float]], Histogram]] = {
    "sturges": lambda data: Histogram(data, algorithm=Histogram.Algorithm.Sturges),
    "fd": lambda data: Histogram(data, algorithm=Histogram.Algorithm.FD),
    "scott": lambda data: Histogram(data, algorithm=Histogram.Algorithm.Scott),
    "breaks_count_4": lambda data: Histogram(data, breaks=4),
    "breaks_count_1": lambda data: Histogram(data, breaks=1),
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/statistics")


def assert_histogram(histogram: Histogram, block: dict[str, Any]) -> None:
    """Compare every observable of a built histogram with the C++ block."""
    assert histogram.bins() == int(block["bins"])
    assert int(histogram.algorithm()) == int(block["algorithm"])
    assert histogram.empty() is bool(block["empty"])

    breaks = histogram.breaks()
    assert len(breaks) == len(block["breaks"])
    for got, want in zip(breaks, block["breaks"], strict=True):
        # TIGHT: computed breaks are ``min + (i+1)*(max-min)/bins``, one
        # multiply-add per break; explicit breaks are echoed unchanged.
        tolerance.tight(got, float(want))

    counts = [histogram.counts(i) for i in range(histogram.bins())]
    assert counts == [int(c) for c in block["counts"]]

    for i, want in enumerate(block["frequency"]):
        # EXACT: frequency is one division of two exactly-representable
        # integers, so both sides round identically.
        tolerance.exact(histogram.frequency(i), float(want))


@pytest.mark.parametrize("case", list(RULE_CASES))
def test_bin_rules(cpp: dict[str, Any], case: str) -> None:
    """Each bin rule reproduces C++'s breaks, counts and frequencies."""
    block = cpp["histogram"]
    data = [float(v) for v in block["data"]]
    assert_histogram(RULE_CASES[case](data), block[case])


def test_rules_disagree_on_bin_count(cpp: dict[str, Any]) -> None:
    """The three algorithms are genuinely different, not aliases."""
    block = cpp["histogram"]
    counts = {rule: int(block[rule]["bins"]) for rule in ("sturges", "fd", "scott")}
    assert len(set(counts.values())) > 1, counts


def test_explicit_breaks_are_sorted_and_deduplicated(cpp: dict[str, Any]) -> None:
    """Explicit breaks are sorted, collapsed by close_enough — and bins is stale.

    The input is ``[3, 1, 2, 2+5e-16]``: sorted it becomes
    ``[1, 2, 2+5e-16, 3]``, the pair around 2 collapses, and three breaks
    remain — while ``bins`` stays at the pre-collapse ``4 + 1``, leaving one
    bin that can never receive a sample.
    """
    block = cpp["histogram"]
    data = [float(v) for v in block["data"]]
    breaks = [float(b) for b in block["explicit_breaks_input"]]
    expected = block["explicit_breaks"]

    histogram = Histogram(data, breaks=breaks)
    assert_histogram(histogram, expected)

    assert len(histogram.breaks()) == 3
    assert histogram.bins() == 5
    assert histogram.counts(3) == 0


def test_default_constructed_is_empty(cpp: dict[str, Any]) -> None:
    """The default constructor stores the out-of-range ``Algorithm(-1)``."""
    block = cpp["histogram"]["default_constructed"]
    histogram = Histogram()
    assert histogram.bins() == int(block["bins"])
    assert int(histogram.algorithm()) == int(block["algorithm"])
    assert histogram.algorithm() is Histogram.Algorithm.Unset
    assert histogram.empty() is bool(block["empty"])


def test_none_algorithm_rejected(cpp: dict[str, Any]) -> None:
    """``Algorithm.None_`` is not a rule and must fail."""
    block = cpp["histogram"]
    assert bool(block["none_algorithm_throws"])
    with pytest.raises(LibraryException):
        Histogram([float(v) for v in block["data"]], algorithm=Histogram.Algorithm.None_)


def test_empty_data_rejected() -> None:
    """``calculate`` requires data."""
    with pytest.raises(LibraryException):
        Histogram([], breaks=3)


def test_breaks_and_algorithm_are_mutually_exclusive() -> None:
    """Python cannot overload on the third argument, so the pair is checked."""
    with pytest.raises(LibraryException):
        Histogram([1.0, 2.0], breaks=2, algorithm=Histogram.Algorithm.Sturges)
    with pytest.raises(LibraryException):
        Histogram([1.0, 2.0])


def test_binning_is_left_open(cpp: dict[str, Any]) -> None:
    """A sample equal to a break point falls in the bin *above* it."""
    histogram = Histogram([0.5, 1.0, 1.5], breaks=[1.0])
    assert histogram.bins() == 2
    assert histogram.counts(0) == 1
    assert histogram.counts(1) == 2
