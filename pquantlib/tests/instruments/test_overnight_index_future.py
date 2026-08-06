"""Cross-validate OvernightIndexFuture against C++ QuantLib v1.43.

Probe: ``v143/inst/swaps`` (key ``overnight_index_future``).

Optional constructor arguments swept at NON-default values:

- ``convexity_adjustment`` — absent (empty handle in C++, ``None`` here) *and*
  a live quote at 12bp.  The quote must reach the price: the test asserts the
  exact ``-100 * ca`` shift as well as the pinned C++ NPV, so an adjustment
  that is accepted and dropped fails twice over.
- ``averaging_method`` — ``Simple`` *and* ``Compound``; the default is
  ``Compound`` and that default is itself pinned against C++.
- Past fixings present in the ``IndexManager``, with the evaluation date moved
  inside the reference period so both the historical-accrual loop and the
  "today's fixing is already known" branch run.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.exceptions import LibraryException
from pquantlib.indexes.ibor.sofr import Sofr
from pquantlib.indexes.index_manager import IndexManager
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.instruments.overnight_index_future import OvernightIndexFuture
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month

_EVAL = Date.from_ymd(17, Month.January, 2024)
_FORWARD_RATE = 0.035
_CONVEXITY = 0.0012


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    IndexManager().clear_histories()
    ObservableSettings().evaluation_date = _EVAL
    yield
    IndexManager().clear_histories()
    ObservableSettings().evaluation_date = None


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/swaps")


def _sofr() -> OvernightIndex:
    curve = FlatForward.from_rate(_EVAL, _FORWARD_RATE, Actual360(), Compounding.Continuous, Frequency.Annual)
    return Sofr(curve)


def _dates(ref: dict[str, Any]) -> tuple[Date, Date]:
    return Date(int(ref["value_date_serial"])), Date(int(ref["maturity_date_serial"]))


def test_value_and_maturity_dates(cpp: dict[str, Any]) -> None:
    ref = cpp["overnight_index_future"]
    value_date, maturity_date = _dates(ref)
    future = OvernightIndexFuture(_sofr(), value_date, maturity_date)
    assert future.value_date() == value_date
    assert future.maturity_date() == maturity_date
    assert future.overnight_index().name().startswith("SOFR")


@pytest.mark.parametrize(
    ("key", "averaging", "with_quote"),
    [
        ("simple_zero_ca", RateAveraging.Simple, False),
        ("compound_zero_ca", RateAveraging.Compound, False),
        ("simple_nonzero_ca", RateAveraging.Simple, True),
        ("compound_nonzero_ca", RateAveraging.Compound, True),
    ],
)
def test_npv_before_the_reference_period(
    cpp: dict[str, Any], key: str, averaging: RateAveraging, with_quote: bool
) -> None:
    ref = cpp["overnight_index_future"]
    value_date, maturity_date = _dates(ref)
    quote = SimpleQuote(_CONVEXITY) if with_quote else None
    future = OvernightIndexFuture(_sofr(), value_date, maturity_date, quote, averaging)
    tight(future.npv(), ref[key], reason=f"{key} NPV")


def test_default_arguments_match_cpp(cpp: dict[str, Any]) -> None:
    """The C++ defaults are an empty convexity handle and ``Compound``."""
    ref = cpp["overnight_index_future"]
    value_date, maturity_date = _dates(ref)
    future = OvernightIndexFuture(_sofr(), value_date, maturity_date)
    tight(future.npv(), ref["defaults"], reason="default-argument NPV")
    tight(ref["defaults"], ref["compound_zero_ca"], reason="C++ default is Compound")
    assert future.averaging_method() == RateAveraging.Compound
    assert future.convexity_adjustment() == 0.0


def test_averaging_method_is_observable(cpp: dict[str, Any]) -> None:
    ref = cpp["overnight_index_future"]
    assert ref["simple_zero_ca"] != ref["compound_zero_ca"]
    value_date, maturity_date = _dates(ref)
    simple = OvernightIndexFuture(_sofr(), value_date, maturity_date, None, RateAveraging.Simple)
    compound = OvernightIndexFuture(_sofr(), value_date, maturity_date, None, RateAveraging.Compound)
    assert simple.npv() != compound.npv()


@pytest.mark.parametrize(
    ("zero_key", "ca_key", "averaging"),
    [
        ("simple_zero_ca", "simple_nonzero_ca", RateAveraging.Simple),
        ("compound_zero_ca", "compound_nonzero_ca", RateAveraging.Compound),
    ],
)
def test_convexity_quote_reaches_the_price(
    cpp: dict[str, Any], zero_key: str, ca_key: str, averaging: RateAveraging
) -> None:
    """``NPV = 100 (1 - (ca + rate))`` — a non-zero quote shifts by ``-100 ca``."""
    ref = cpp["overnight_index_future"]
    tight(ref[ca_key], ref[zero_key] - 100.0 * _CONVEXITY, reason="C++ convexity shift")

    value_date, maturity_date = _dates(ref)
    quote = SimpleQuote(_CONVEXITY)
    future = OvernightIndexFuture(_sofr(), value_date, maturity_date, quote, averaging)
    assert future.convexity_adjustment() == _CONVEXITY
    tight(future.npv(), ref[ca_key])

    # The quote is live: moving it must move the price (observer wiring).
    quote.set_value(0.0)
    tight(future.npv(), ref[zero_key], reason="quote update propagates")


def test_convexity_adjustment_inspector(cpp: dict[str, Any]) -> None:
    ref = cpp["overnight_index_future"]
    value_date, maturity_date = _dates(ref)
    assert (
        OvernightIndexFuture(_sofr(), value_date, maturity_date).convexity_adjustment()
        == ref["convexity_adjustment_empty"]
    )
    with_quote = OvernightIndexFuture(_sofr(), value_date, maturity_date, SimpleQuote(_CONVEXITY))
    tight(with_quote.convexity_adjustment(), ref["convexity_adjustment_quote"])


def _load_history(index: OvernightIndex, ref: dict[str, Any]) -> None:
    for entry in ref["history"]:
        index.add_fixing(Date(int(entry["date_serial"])), float(entry["fixing"]))


@pytest.mark.parametrize(
    ("key", "averaging", "with_quote"),
    [
        ("simple_past_fixings", RateAveraging.Simple, False),
        ("compound_past_fixings", RateAveraging.Compound, False),
        ("compound_past_fixings_ca", RateAveraging.Compound, True),
    ],
)
def test_npv_inside_the_reference_period(
    cpp: dict[str, Any], key: str, averaging: RateAveraging, with_quote: bool
) -> None:
    """Valued inside the period, off the fixings held by the IndexManager."""
    ref = cpp["overnight_index_future"]
    value_date, maturity_date = _dates(ref)
    index = _sofr()
    _load_history(index, ref)
    ObservableSettings().evaluation_date = Date(int(ref["inside_eval_serial"]))
    quote = SimpleQuote(_CONVEXITY) if with_quote else None
    future = OvernightIndexFuture(index, value_date, maturity_date, quote, averaging)
    tight(future.npv(), ref[key], reason=f"{key} NPV")


def test_past_fixings_change_the_price(cpp: dict[str, Any]) -> None:
    """The historical fixings must actually be consumed, not merely required."""
    ref = cpp["overnight_index_future"]
    assert ref["compound_past_fixings"] != ref["compound_zero_ca"]
    assert ref["simple_past_fixings"] != ref["simple_zero_ca"]


def test_is_expired(cpp: dict[str, Any]) -> None:
    ref = cpp["overnight_index_future"]
    value_date, maturity_date = _dates(ref)
    future = OvernightIndexFuture(_sofr(), value_date, maturity_date)
    assert future.is_expired() is ref["is_expired_before"]

    ObservableSettings().evaluation_date = Date.from_ymd(1, Month.August, 2024)
    expired = OvernightIndexFuture(_sofr(), value_date, maturity_date)
    assert expired.is_expired() is ref["is_expired_after"]


def test_null_index_raises(cpp: dict[str, Any]) -> None:
    ref = cpp["overnight_index_future"]
    assert cpp["overnight_index_future_raises"]["null_index"]["raises"] is True
    value_date, maturity_date = _dates(ref)
    with pytest.raises(LibraryException, match="null overnight index"):
        OvernightIndexFuture(None, value_date, maturity_date)  # pyright: ignore[reportArgumentType]


def test_missing_past_fixing_raises(cpp: dict[str, Any]) -> None:
    ref = cpp["overnight_index_future"]
    assert cpp["overnight_index_future_raises"]["missing_past_fixing"]["raises"] is True
    value_date, maturity_date = _dates(ref)
    ObservableSettings().evaluation_date = Date(int(ref["inside_eval_serial"]))
    future = OvernightIndexFuture(_sofr(), value_date, maturity_date, None, RateAveraging.Compound)
    with pytest.raises(LibraryException, match="missing rate on"):
        future.npv()
