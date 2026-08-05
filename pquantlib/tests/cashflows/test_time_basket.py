"""Cross-validate ``TimeBasket`` against C++ QuantLib v1.43.

Probe:     ``migration-harness/cpp/probes/v143_cf_timebasket/probe.cpp``
Reference: ``v143/cf/timebasket``

``TimeBasket`` privately derives from ``std::map<Date, Real>``, so the map
semantics are as much part of the contract as the algebra: ascending-date
iteration, and an ``operator[]`` that inserts ``0.0`` on a missing key. Every
case is therefore compared as the full ``(date-serial, amount)`` listing, not
as a headline number — a basket can hold the right total while two entries sit
on the wrong dates.

Amounts are exact sums / linear splits of the inputs, so TIGHT applies
throughout (the ``rebin`` splits are one multiply and one divide over
identical integer day counts).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.time_basket import TimeBasket
from pquantlib.exceptions import LibraryException
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# Deliberately unsorted, matching the probe: the map must re-order them.
_DATES = [
    Date.from_ymd(15, Month.March, 2026),
    Date.from_ymd(15, Month.January, 2026),
    Date.from_ymd(15, Month.June, 2026),
    Date.from_ymd(15, Month.September, 2026),
]
_VALUES = [2000.0, 1000.0, 3000.0, 4000.0]

_OTHER_DATES = [
    Date.from_ymd(15, Month.March, 2026),
    Date.from_ymd(15, Month.September, 2026),
    Date.from_ymd(1, Month.January, 2026),
    Date.from_ymd(31, Month.December, 2026),
]
_OTHER_VALUES = [500.0, -250.0, 125.0, 77.5]


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/timebasket")


def _base() -> TimeBasket:
    return TimeBasket(_DATES, _VALUES)


def _other() -> TimeBasket:
    return TimeBasket(_OTHER_DATES, _OTHER_VALUES)


def _check_basket(basket: TimeBasket, ref: dict[str, Any]) -> None:
    """Compare size, the full date listing (in order) and the amounts."""
    assert basket.size() == ref["size"]
    assert len(basket) == ref["size"]
    ordered_keys = basket.keys()  # C++ map order, not insertion order
    assert [d.serial_number() for d in ordered_keys] == ref["dates"]
    assert [d.serial_number() for d in basket] == ref["dates"]
    assert [d.serial_number() for d, _ in basket.items()] == ref["dates"]
    for got, want in zip(basket.values(), ref["amounts"], strict=True):
        tolerance.tight(got, float(want))


# --- construction ----------------------------------------------------------


def test_ctor_listing(cpp: dict[str, Any]) -> None:
    _check_basket(_base(), cpp["ctor"])


def test_ctor_duplicate_date_keeps_last_value(cpp: dict[str, Any]) -> None:
    """Construction assigns rather than accumulates (timebasket.cpp:32-33)."""
    dates = [
        Date.from_ymd(15, Month.January, 2026),
        Date.from_ymd(15, Month.March, 2026),
        Date.from_ymd(15, Month.January, 2026),
    ]
    _check_basket(TimeBasket(dates, [10.0, 20.0, 99.0]), cpp["ctor_duplicate_date"])


def test_default_ctor_is_empty() -> None:
    assert TimeBasket().size() == 0
    assert TimeBasket().items() == []


def test_ctor_size_mismatch_raises(cpp: dict[str, Any]) -> None:
    assert cpp["ctor_size_mismatch_raises"]["raises"] is True
    with pytest.raises(LibraryException, match="number of dates differs"):
        TimeBasket(_DATES, _VALUES[:-1])


# --- map interface ---------------------------------------------------------


def test_has_date(cpp: dict[str, Any]) -> None:
    basket = _base()
    assert basket.has_date(Date.from_ymd(15, Month.March, 2026)) is cpp["has_date_present"]
    assert basket.has_date(Date.from_ymd(16, Month.March, 2026)) is cpp["has_date_absent"]
    # has_date must NOT insert (unlike operator[]).
    assert basket.size() == cpp["ctor"]["size"]
    assert (Date.from_ymd(15, Month.March, 2026) in basket) is True


def test_lookup_present(cpp: dict[str, Any]) -> None:
    tolerance.tight(_base()[Date.from_ymd(15, Month.June, 2026)], float(cpp["lookup_present"]))


def test_lookup_missing_inserts_zero(cpp: dict[str, Any]) -> None:
    """``std::map::operator[]`` value-initialises AND inserts a missing key."""
    basket = _base()
    assert basket.size() == int(cpp["lookup_missing_size_before"])
    tolerance.tight(basket[Date.from_ymd(1, Month.December, 2026)], float(cpp["lookup_missing_value"]))
    assert basket.size() == int(cpp["lookup_missing_size_after"])
    _check_basket(basket, cpp["lookup_missing_basket"])


def test_reversed_is_descending(cpp: dict[str, Any]) -> None:
    """``rbegin()``/``rend()`` walk the map backwards."""
    assert [d.serial_number() for d in reversed(_base())] == list(reversed(cpp["ctor"]["dates"]))


def test_setitem_overwrites() -> None:
    basket = _base()
    basket[Date.from_ymd(15, Month.March, 2026)] = -1.0
    tolerance.tight(basket[Date.from_ymd(15, Month.March, 2026)], -1.0)
    assert basket.size() == 4


# --- algebra ---------------------------------------------------------------


def test_other_listing(cpp: dict[str, Any]) -> None:
    _check_basket(_other(), cpp["other"])


def test_plus_equals(cpp: dict[str, Any]) -> None:
    basket = _base()
    basket += _other()
    _check_basket(basket, cpp["plus_equals"])


def test_minus_equals(cpp: dict[str, Any]) -> None:
    basket = _base()
    basket -= _other()
    _check_basket(basket, cpp["minus_equals"])


def test_plus_then_minus_keeps_introduced_keys(cpp: dict[str, Any]) -> None:
    """``+=`` then ``-=`` restores the amounts but not the original size."""
    basket = _base()
    basket += _other()
    basket -= _other()
    _check_basket(basket, cpp["plus_then_minus"])
    assert basket.size() > _base().size()


def test_plus_equals_leaves_operand_untouched(cpp: dict[str, Any]) -> None:
    basket = _base()
    other = _other()
    basket += other
    _check_basket(other, cpp["other"])


# --- rebin -----------------------------------------------------------------


def test_rebin_coarser_unsorted_buckets(cpp: dict[str, Any]) -> None:
    """Buckets are sorted internally; out-of-range entries collapse onto an end."""
    buckets = [Date.from_ymd(1, Month.July, 2026), Date.from_ymd(1, Month.February, 2026)]
    _check_basket(_base().rebin(buckets), cpp["rebin_coarser"])


def test_rebin_finer(cpp: dict[str, Any]) -> None:
    """A bucket coinciding with an entry takes the whole value (``pDate == date``)."""
    buckets = [
        Date.from_ymd(15, Month.January, 2026),
        Date.from_ymd(15, Month.February, 2026),
        Date.from_ymd(15, Month.March, 2026),
        Date.from_ymd(15, Month.April, 2026),
        Date.from_ymd(15, Month.May, 2026),
        Date.from_ymd(15, Month.June, 2026),
        Date.from_ymd(15, Month.July, 2026),
        Date.from_ymd(15, Month.August, 2026),
        Date.from_ymd(15, Month.September, 2026),
    ]
    _check_basket(_base().rebin(buckets), cpp["rebin_finer"])


def test_rebin_partial_overlap(cpp: dict[str, Any]) -> None:
    """Entries before the first and after the last bucket both collapse onto an end."""
    buckets = [
        Date.from_ymd(1, Month.March, 2026),
        Date.from_ymd(1, Month.May, 2026),
        Date.from_ymd(1, Month.July, 2026),
    ]
    _check_basket(_base().rebin(buckets), cpp["rebin_partial_overlap"])


def test_rebin_single_bucket(cpp: dict[str, Any]) -> None:
    _check_basket(_base().rebin([Date.from_ymd(1, Month.May, 2026)]), cpp["rebin_single_bucket"])


def test_rebin_conserves_total(cpp: dict[str, Any]) -> None:
    """The linear split is value-conserving, so every grid keeps the total."""
    total = float(sum(float(a) for a in cpp["ctor"]["amounts"]))
    for key, buckets in (
        (
            "rebin_coarser",
            [Date.from_ymd(1, Month.July, 2026), Date.from_ymd(1, Month.February, 2026)],
        ),
        ("rebin_single_bucket", [Date.from_ymd(1, Month.May, 2026)]),
    ):
        tolerance.tight(float(sum(float(a) for a in cpp[key]["amounts"])), total)
        tolerance.tight(sum(_base().rebin(buckets).values()), total)


def test_rebin_source_unchanged(cpp: dict[str, Any]) -> None:
    basket = _base()
    basket.rebin([Date.from_ymd(1, Month.May, 2026)])
    _check_basket(basket, cpp["rebin_source_unchanged"])


def test_rebin_empty_buckets_raises(cpp: dict[str, Any]) -> None:
    assert cpp["rebin_empty_buckets_raises"]["raises"] is True
    with pytest.raises(LibraryException, match="empty bucket structure"):
        _base().rebin([])
