"""Tests for VarianceSwap.

# C++ parity: ql/instruments/varianceswap.{hpp,cpp} @ v1.43.

Structural coverage only — inspectors, argument plumbing, every
``validate()`` branch and the whole ``is_expired`` boundary matrix, all pinned
against ``migration-harness/references/v143/inst/lookbackvarswap``. The priced
cross-validation lives with the engine, in
``tests/pricingengines/forward/test_replicating_variance_swap_engine.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.instruments.variance_swap import (
    VarianceSwap,
    VarianceSwapArguments,
    VarianceSwapResults,
)
from pquantlib.option import OptionArguments
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.position import PositionType
from pquantlib.testing import reference_reader
from pquantlib.time.date import Date
from pquantlib.time.month import Month

TODAY: Final[Date] = Date.from_ymd(1, Month.March, 2025)
MATURITY: Final[Date] = TODAY + 90
NULL_DATE: Final[Date] = Date()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/lookbackvarswap")


@pytest.fixture(autouse=True)
def _pinned_settings() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """``is_expired`` reads two globals; pin both and restore them after."""
    settings = ObservableSettings()
    previous_date = settings.evaluation_date
    previous_flag = settings.include_reference_date_events
    settings.evaluation_date = TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous_date
        settings.include_reference_date_events = previous_flag


def _swap(
    position: PositionType = PositionType.Long,
    strike: float = 0.04,
    notional: float = 50000.0,
    start: Date = TODAY,
    maturity: Date = MATURITY,
) -> VarianceSwap:
    return VarianceSwap(position, strike, notional, start, maturity)


# --- inspectors --------------------------------------------------------------


def test_inspectors_return_the_constructor_arguments() -> None:
    """All five arguments are non-default and each must come back distinct.

    ``start_date`` and ``maturity_date`` are adjacent same-typed arguments, so
    they are given clearly different values to catch a swap.
    """
    swap = _swap(
        position=PositionType.Short,
        strike=0.0625,
        notional=1234567.0,
        start=TODAY + 3,
        maturity=TODAY + 400,
    )
    assert swap.position() == PositionType.Short
    assert swap.strike() == 0.0625
    assert swap.notional() == 1234567.0
    assert swap.start_date() == TODAY + 3
    assert swap.maturity_date() == TODAY + 400


# --- argument plumbing -------------------------------------------------------


def test_setup_arguments_populates_every_field() -> None:
    swap = _swap(
        position=PositionType.Short,
        strike=0.09,
        notional=250000.0,
        start=TODAY + 3,
        maturity=TODAY + 400,
    )
    args = VarianceSwapArguments()
    swap.setup_arguments(args)
    assert args.position == PositionType.Short
    assert args.strike == 0.09
    assert args.notional == 250000.0
    assert args.start_date == TODAY + 3
    assert args.maturity_date == TODAY + 400


def test_setup_arguments_rejects_a_foreign_carrier() -> None:
    """A subclass of the right carrier is fine; an unrelated one is not."""

    class _Subclass(VarianceSwapArguments):
        pass

    _swap().setup_arguments(_Subclass())
    with pytest.raises(LibraryException, match="wrong argument type"):
        _swap().setup_arguments(OptionArguments())


def test_arguments_start_out_null() -> None:
    """# C++ parity: ``arguments()`` inits strike/notional to ``Null<Real>()``."""
    args = VarianceSwapArguments()
    assert args.strike is None
    assert args.notional is None
    assert args.start_date == NULL_DATE
    assert args.maturity_date == NULL_DATE
    assert args.position == PositionType.Long


# --- validate ----------------------------------------------------------------


def _args(
    strike: float | None = 0.04,
    notional: float | None = 50000.0,
    start: Date = TODAY,
    maturity: Date = MATURITY,
) -> VarianceSwapArguments:
    args = VarianceSwapArguments()
    args.position = PositionType.Long
    args.strike = strike
    args.notional = notional
    args.start_date = start
    args.maturity_date = maturity
    return args


def test_validate_accepts_a_well_formed_argument_set(cpp: dict[str, Any]) -> None:
    assert cpp["vs_args_validate_ok"]["expected"]["raises"] is False
    _args().validate()


def test_validate_rejects_missing_strike(cpp: dict[str, Any]) -> None:
    assert cpp["vs_validate_null_strike"]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="no strike given"):
        _args(strike=None).validate()


@pytest.mark.parametrize(
    ("strike", "case"),
    [(0.0, "vs_validate_zero_strike"), (-0.04, "vs_validate_negative_strike")],
)
def test_validate_rejects_nonpositive_strike(cpp: dict[str, Any], strike: float, case: str) -> None:
    assert cpp[case]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="negative or null strike given"):
        _args(strike=strike).validate()


def test_validate_rejects_missing_notional(cpp: dict[str, Any]) -> None:
    assert cpp["vs_validate_null_notional"]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="no notional given"):
        _args(notional=None).validate()


@pytest.mark.parametrize(
    ("notional", "case"),
    [(0.0, "vs_validate_zero_notional"), (-50000.0, "vs_validate_negative_notional")],
)
def test_validate_rejects_nonpositive_notional(cpp: dict[str, Any], notional: float, case: str) -> None:
    assert cpp[case]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="negative or null notional given"):
        _args(notional=notional).validate()


def test_validate_rejects_null_start_date(cpp: dict[str, Any]) -> None:
    assert cpp["vs_args_validate_null_start_date"]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="null start date given"):
        _args(start=NULL_DATE).validate()


def test_validate_rejects_null_maturity_date(cpp: dict[str, Any]) -> None:
    assert cpp["vs_args_validate_null_maturity_date"]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="null maturity date given"):
        _args(maturity=NULL_DATE).validate()


# --- results -----------------------------------------------------------------


def test_variance_without_an_engine_raises() -> None:
    with pytest.raises(LibraryException, match="null pricing engine"):
        _swap().variance()


def test_results_reset_clears_the_variance() -> None:
    results = VarianceSwapResults()
    results.variance = 0.04
    results.value = 100.0
    results.reset()
    assert results.variance is None
    assert results.value is None


# --- is_expired --------------------------------------------------------------


def test_is_expired_boundary_matrix(cpp: dict[str, Any]) -> None:
    """All six {ref vs maturity} x {include_reference_date_events} cases.

    The inclusion flag is observable ONLY on the maturity date itself:
    ``maturity < ref`` when set, ``maturity <= ref`` when not.
    """
    settings = ObservableSettings()
    checked = 0
    for name, case in cpp.items():
        if not name.startswith("vs_expired_"):
            continue
        inputs, expected = case["inputs"], case["expected"]
        settings.include_reference_date_events = bool(inputs["include_reference_date_events"])
        settings.evaluation_date = Date(int(inputs["evaluation_date_serial"]))
        swap = _swap(maturity=Date(int(inputs["maturity_serial"])))
        assert swap.is_expired() is bool(expected["is_expired"]), name
        checked += 1
    assert checked == 6, f"expected the full boundary matrix, got {checked}"


def test_null_maturity_reports_expired_rather_than_failing_validation(
    cpp: dict[str, Any],
) -> None:
    """A null maturity is swallowed by ``is_expired`` before ``validate`` runs.

    The null date's serial is 0, so it is ``<=`` any evaluation date and the
    swap short-circuits to the expired path. C++ does the same — the
    ``no maturity date given`` requirement is only reachable by calling
    ``validate()`` directly, which is what
    ``test_validate_rejects_null_maturity_date`` above does.
    """
    assert cpp["vs_null_maturity_is_expired"]["expected"]["is_expired"] is True
    assert cpp["vs_null_maturity_swallowed_by_is_expired"]["expected"]["raises"] is False
    assert _swap(maturity=NULL_DATE).is_expired() is True


def test_setup_expired_clears_npv_and_variance() -> None:
    swap = _swap(maturity=NULL_DATE)
    swap.setup_expired()
    assert swap.npv() == 0.0
    with pytest.raises(LibraryException, match="result not available"):
        swap.variance()
