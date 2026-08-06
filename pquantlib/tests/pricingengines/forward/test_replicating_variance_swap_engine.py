"""Cross-validate ReplicatingVarianceSwapEngine against the C++ probe.

Reference: ``migration-harness/references/v143/inst/lookbackvarswap``.

Covers the Demeterfi-Derman-Kamal-Zou (1999) static-replication engine over two
markets: the DDKZ paper's own skewed 18-strike surface (market ``A``, whose
fair variance the probe reproduces at 0.041889 against the paper's published
0.04189) and a flat-vol market with different spot, rates and maturity (market
``B``), so that a wiring error which happens to cancel under the skew has
nowhere to hide.

Each probe case carries its whole configuration inline — position, variance
strike, notional, start and maturity dates, ``dk`` and both strike ladders — so
the sweeps below reconstruct it. The sweeps exist because of what this engine
invites: ``notional`` and ``position`` are pure multipliers on the NPV and
completely invisible in ``variance()``, and ``dk`` / the two ladders reach the
price only through the weight strip. So each is moved off its default and
asserted against C++, and ``variance()`` is separately asserted to be
INVARIANT across position, notional and strike — a port that let any of them
leak into the fair variance fails there rather than nowhere.

Tolerances
----------
* ``option_weights`` — EXACT. The weights are absolute slope differences of a
  log payoff over rational strike steps: the same operations in the same order
  on the same doubles. Measured over all 108 weights in the reference the
  agreement is bit-identical, so anything looser would be throwing away a
  free assertion.
* ``variance()`` — TIGHT. A finite sum of closed-form Black prices over the
  strip, discounted; a short chain of arithmetic with no iteration and no
  quadrature. Worst measured relative disagreement 8.8e-15, 114x inside the
  tier.
* ``NPV()`` — TIGHT amplified by the strike cancellation, via ``custom``.
  ``NPV = +-df * N * (V - K)``, so

      |dNPV| / |NPV| = |dV| / |V - K| = (|dV| / |V|) * (|V| / |V - K|)

  i.e. the NPV inherits the fair variance's *relative* error multiplied by
  ``amp = |V| / |V - K|``, which is 22.2 at the base case and 16.0 at the
  shortest maturity. The bound used is therefore ``TIGHT_rel * amp`` with
  ``amp`` computed per case from the reference's own ``variance`` and
  ``strike``. This is not a loosening to force green — the identity is exact:
  measured ``npv_rel / (var_rel * amp)`` is 1.00 to two decimals on every case,
  and the realised worst NPV error is 1.4e-13, a further 150x inside the
  derived bound.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.variance_swap import VarianceSwap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType, PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.position import PositionType
from pquantlib.pricingengines.forward.replicating_variance_swap_engine import (
    ReplicatingVarianceSwapEngine,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.volatility.equity_fx.black_variance_surface import (
    BlackVarianceSurface,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

TODAY: Final[Date] = Date.from_ymd(1, Month.March, 2025)

# The Demeterfi-Derman-Kamal-Zou (1999) skew, verbatim from the v1.43
# test-suite case testReplicatingVarianceSwap.
DDKZ_STRIKES: Final[list[float]] = [
    50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
    95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0,
]  # fmt: skip
DDKZ_VOLS: Final[list[float]] = [
    0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22,
    0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13,
]  # fmt: skip
DDKZ_CALLS: Final[list[float]] = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]
DDKZ_PUTS: Final[list[float]] = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]

MATURITY_A: Final[Date] = TODAY + 90

# TIGHT's relative tier; the NPV bound below is this amplified by the
# variance-minus-strike cancellation.
_TIGHT_REL: Final[float] = 1.0e-12


@pytest.fixture(autouse=True)
def _pinned_evaluation_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """VarianceSwap.is_expired() reads the global evaluation date.

    Without pinning it the probe's 2025 maturities are in the past, every swap
    reports itself expired and prices to zero. Restored afterwards so the
    singleton does not leak into other modules.
    """
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = TODAY
    try:
        yield
    finally:
        settings.evaluation_date = previous


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/lookbackvarswap")


# --- market construction: mirrors the probe exactly --------------------------


def _flat_curve(rate: float) -> FlatForward:
    return FlatForward.from_rate(reference_date=TODAY, forward_rate=rate, day_counter=Actual365Fixed())


def _ddkz_vol() -> BlackVolTermStructure:
    return BlackVarianceSurface(
        reference_date=TODAY,
        calendar=NullCalendar(),
        dates=[MATURITY_A],
        strikes=DDKZ_STRIKES,
        black_vol_matrix=np.asarray([[v] for v in DDKZ_VOLS], dtype=np.float64),
        day_counter=Actual365Fixed(),
    )


def _market(name: str) -> GeneralizedBlackScholesProcess:
    if name == "A":
        # DDKZ example: spot 100, no dividend, 5% risk-free, skewed surface.
        return BlackScholesMertonProcess(
            x0=SimpleQuote(100.0),
            dividend_ts=_flat_curve(0.0),
            risk_free_ts=_flat_curve(0.05),
            black_vol_ts=_ddkz_vol(),
        )
    # Flat vol, different spot / rates / maturity.
    return BlackScholesMertonProcess(
        x0=SimpleQuote(120.0),
        dividend_ts=_flat_curve(0.03),
        risk_free_ts=_flat_curve(0.02),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY,
            calendar=NullCalendar(),
            day_counter=Actual365Fixed(),
            volatility=0.22,
        ),
    )


def _engine(process: GeneralizedBlackScholesProcess, inputs: dict[str, Any]) -> ReplicatingVarianceSwapEngine:
    return ReplicatingVarianceSwapEngine(
        process,
        float(inputs["dk"]),
        [float(k) for k in inputs["call_strikes"]],
        [float(k) for k in inputs["put_strikes"]],
    )


def _swap(inputs: dict[str, Any]) -> VarianceSwap:
    return VarianceSwap(
        PositionType.Long if inputs["position"] == "Long" else PositionType.Short,
        float(inputs["strike"]),
        float(inputs["notional"]),
        Date(int(inputs["start_serial"])),
        Date(int(inputs["maturity_serial"])),
    )


def _priced(inputs: dict[str, Any]) -> VarianceSwap:
    process = _market(str(inputs["market"]))
    swap = _swap(inputs)
    swap.set_pricing_engine(_engine(process, inputs))
    return swap


def _cases(cpp: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(k, v) for k, v in cpp.items() if v["inputs"].get("kind") == "variance_swap"]


def _npv_bound(expected: dict[str, Any], inputs: dict[str, Any]) -> float:
    """``TIGHT_rel * |V| / |V - K|`` — see the module docstring derivation."""
    variance = float(expected["variance"])
    strike = float(inputs["strike"])
    return _TIGHT_REL * abs(variance) / abs(variance - strike)


# --- the sweep ---------------------------------------------------------------


def test_variance_swap_npv_and_variance(cpp: dict[str, Any]) -> None:
    """Every variance-swap case in the probe, NPV and fair variance."""
    checked = 0
    for name, case in _cases(cpp):
        inputs, expected = case["inputs"], case["expected"]
        swap = _priced(inputs)

        tight(swap.variance(), float(expected["variance"]), reason=f"{name}: variance")
        tolerance.custom(
            swap.npv(),
            float(expected["npv"]),
            abs_tol=0.0,
            rel_tol=_npv_bound(expected, inputs),
            reason=f"{name}: NPV inherits the fair variance's relative error amplified by |V| / |V - K|",
        )
        checked += 1
    assert checked > 15, f"expected a variance-swap sweep, got {checked} cases"


def test_variance_swap_inspectors_round_trip(cpp: dict[str, Any]) -> None:
    """Position, strike, notional and both dates survive construction.

    ``start_date`` in particular is stored and never read by any engine, which
    is exactly the shape of argument that gets quietly swapped with its
    neighbour.
    """
    checked = 0
    for name, case in _cases(cpp):
        inputs, expected = case["inputs"], case["expected"]
        swap = _priced(inputs)
        assert str(swap.position()) == expected["inspector_position"], name
        assert swap.strike() == float(expected["inspector_strike"]), name
        assert swap.notional() == float(expected["inspector_notional"]), name
        assert swap.start_date().serial_number() == int(expected["inspector_start_serial"]), name
        assert swap.maturity_date().serial_number() == int(expected["inspector_maturity_serial"]), name
        assert swap.is_expired() is bool(expected["is_expired"]), name
        checked += 1
    assert checked > 15


def test_option_weights_ladder(cpp: dict[str, Any]) -> None:
    """The full replicating strip: type, strike and weight, in emission order.

    Calls ascending then puts descending, one leg per input strike (the ``dk``
    end strike supplies the last slope and gets no option of its own).
    """
    checked = 0
    for name, case in _cases(cpp):
        expected = case["expected"]
        if "option_weights" not in expected:
            continue
        swap = _priced(case["inputs"])
        swap.npv()
        weights: list[tuple[StrikedTypePayoff, float]] = swap.additional_results()["optionWeights"]
        reference: list[dict[str, Any]] = expected["option_weights"]
        assert len(weights) == len(reference), name
        for (payoff, weight), ref in zip(weights, reference, strict=True):
            assert isinstance(payoff, PlainVanillaPayoff), name
            assert payoff.option_type() == (OptionType.Call if ref["type"] == "Call" else OptionType.Put), (
                name
            )
            exact(payoff.strike(), float(ref["strike"]), reason=f"{name}: strike")
            exact(weight, float(ref["weight"]), reason=f"{name}: weight")
        checked += 1
    assert checked >= 10, f"expected weight ladders in the reference, got {checked}"


def test_dk_moves_the_ladder(cpp: dict[str, Any]) -> None:
    """``dk`` is an optional argument with a 5.0 default and it must be used.

    Same market, same ladders, same swap: only ``dk`` differs, and both the
    weights and the fair variance must move with it.
    """
    base = cpp["vs_a_long_base"]
    for other_name in ("vs_a_dk0", "vs_a_dk1"):
        other = cpp[other_name]
        assert float(other["inputs"]["dk"]) != float(base["inputs"]["dk"])
        assert float(other["expected"]["variance"]) != float(base["expected"]["variance"])
        assert _priced(other["inputs"]).variance() != _priced(base["inputs"]).variance()


def test_dk_defaults_to_five(cpp: dict[str, Any]) -> None:
    """Omitting ``dk`` must give the same answer as passing C++'s 5.0 default.

    ``vs_a_long_base`` is generated with dk == 5.0, so the constructed-with-
    default engine has to reproduce it exactly — a port that defaulted to
    anything else (or to 0, or to None) shows up here rather than only in the
    NPV of some future caller.
    """
    inputs = cpp["vs_a_long_base"]["inputs"]
    assert float(inputs["dk"]) == 5.0
    defaulted = VarianceSwap(
        PositionType.Long,
        float(inputs["strike"]),
        float(inputs["notional"]),
        Date(int(inputs["start_serial"])),
        Date(int(inputs["maturity_serial"])),
    )
    defaulted.set_pricing_engine(
        ReplicatingVarianceSwapEngine(
            _market("A"),
            call_strikes=[float(k) for k in inputs["call_strikes"]],
            put_strikes=[float(k) for k in inputs["put_strikes"]],
        )
    )
    explicit = _priced(inputs)
    exact(defaulted.variance(), explicit.variance())
    exact(defaulted.npv(), explicit.npv())


def test_ladder_is_sorted_and_deduplicated(cpp: dict[str, Any]) -> None:
    """An unsorted or duplicated ladder must give the sorted-unique answer.

    Pins the ``std::sort`` + ``std::unique`` pair inside
    ``computeOptionWeights``; the engine constructor deliberately does NOT
    require sorted input.
    """
    reference = _priced(cpp["vs_a_ladder_short"]["inputs"])
    for name in ("vs_a_ladder_unsorted", "vs_a_ladder_duplicates"):
        swap = _priced(cpp[name]["inputs"])
        exact(swap.variance(), reference.variance(), reason=f"{name}: variance")
        exact(swap.npv(), reference.npv(), reason=f"{name}: NPV")


def test_variance_is_invariant_to_position_notional_and_strike(
    cpp: dict[str, Any],
) -> None:
    """``variance()`` sees none of the three NPV multipliers.

    A port that let the position sign, the notional or the variance strike leak
    into the fair variance would pass every NPV assertion above by accident and
    fail here.
    """
    reference = _priced(cpp["vs_a_long_base"]["inputs"]).variance()
    for name in (
        "vs_a_short_base",
        "vs_a_notional0",
        "vs_a_notional1",
        "vs_a_strike0",
        "vs_a_strike1",
        "vs_a_short_strike_high",
        "vs_a_startdate_shifted",
    ):
        exact(_priced(cpp[name]["inputs"]).variance(), reference, reason=name)


def test_position_flips_the_npv_sign(cpp: dict[str, Any]) -> None:
    """Long and Short at the same strike/notional are exact negatives."""
    long_npv = _priced(cpp["vs_a_long_base"]["inputs"]).npv()
    short_npv = _priced(cpp["vs_a_short_base"]["inputs"]).npv()
    exact(short_npv, -long_npv)

    long_b = _priced(cpp["vs_b_long_base"]["inputs"]).npv()
    short_b = _priced(cpp["vs_b_short_base"]["inputs"]).npv()
    exact(short_b, -long_b)


def test_start_date_does_not_reach_the_price(cpp: dict[str, Any]) -> None:
    """The engine reads maturity only; shifting the start must change nothing."""
    base = cpp["vs_a_long_base"]
    shifted = cpp["vs_a_startdate_shifted"]
    assert int(shifted["inputs"]["start_serial"]) != int(base["inputs"]["start_serial"])
    exact(_priced(shifted["inputs"]).npv(), _priced(base["inputs"]).npv())
    assert _priced(shifted["inputs"]).start_date() != _priced(base["inputs"]).start_date()


# --- engine constructor guards ----------------------------------------------


def _ctor(call_strikes: list[float], put_strikes: list[float]) -> ReplicatingVarianceSwapEngine:
    return ReplicatingVarianceSwapEngine(_market("A"), 5.0, call_strikes, put_strikes)


def test_engine_requires_both_ladders(cpp: dict[str, Any]) -> None:
    assert cpp["vs_engine_empty_call_strikes"]["expected"]["raises"] is True
    assert cpp["vs_engine_empty_put_strikes"]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="no strike"):
        _ctor([], DDKZ_PUTS)
    with pytest.raises(LibraryException, match="no strike"):
        _ctor(DDKZ_CALLS, [])


def test_engine_requires_positive_put_strikes(cpp: dict[str, Any]) -> None:
    assert cpp["vs_engine_nonpositive_put_strike"]["expected"]["raises"] is True
    assert cpp["vs_engine_negative_put_strike"]["expected"]["raises"] is True
    for puts in ([0.0, 100.0], [-5.0, 100.0]):
        with pytest.raises(LibraryException, match="min put strike must be positive"):
            _ctor(DDKZ_CALLS, puts)


def test_engine_requires_ladders_to_meet(cpp: dict[str, Any]) -> None:
    assert cpp["vs_engine_min_call_max_put_differ"]["expected"]["raises"] is True
    with pytest.raises(LibraryException, match="min call and max put strikes differ"):
        _ctor([105.0, 110.0], [90.0, 100.0])


def test_engine_accepts_unsorted_ladders(cpp: dict[str, Any]) -> None:
    """Ordering is not a constructor constraint in C++ — do not invent one."""
    assert cpp["vs_engine_unsorted_ladders_ok"]["expected"]["raises"] is False
    _ctor([110.0, 100.0], [90.0, 100.0])
