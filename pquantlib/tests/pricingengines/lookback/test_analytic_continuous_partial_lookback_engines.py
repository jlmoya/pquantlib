"""Cross-validate the two partial-time lookback engines against the C++ probe.

Reference: ``migration-harness/references/v143/inst/lookbackvarswap``.

Covers :class:`AnalyticContinuousPartialFloatingLookbackEngine` and
:class:`AnalyticContinuousPartialFixedLookbackEngine` (both Heynen-Kat 1994,
Haug 2nd ed. pp.146 and 148).

Each probe case carries its whole market inline — spot, both rates, the vol
level or a flag selecting the skew, the running extremum or strike, the
lookback boundary date and (for the floating variant) lambda — so the sweeps
below reconstruct it rather than restating it. The sweeps move ONE argument at
a time off a common base case, which is what makes them able to catch the
failure mode these two classes invite: ``lambda`` and the lookback boundary
date are the only things that distinguish them from the plain lookbacks, and
an implementation that accepted them and dropped them would still return a
plausible price. ``pflt_*_degenerate`` is the corner where the partial-time
price legitimately equals the plain one, and is asserted to agree with
:class:`AnalyticContinuousFloatingLookbackEngine` for exactly that reason.

Tolerances
----------
TIGHT throughout. Both closed forms are a fixed chain of arithmetic over
identical inputs; the only step that is not is the bivariate normal CDF, where
the port integrates through scipy's Genz-Bretz and C++ through its own
Genz-2004 hybrid. Those two disagree at the O(1e-16) level, and each CDF enters
the price multiplied by terms of order the underlying (~100) against an NPV of
order 10, so the lever from CDF error to relative NPV error is about 10x.
Measured over all 94 lookback cases in the reference the worst relative
disagreement is 2.2e-14 — 45x inside the TIGHT tier — and the degenerate
|rho| == 1 branches (which both engines reach, and which the port evaluates in
closed form) agree to 2.2e-16 absolute. LOOSE would discard four orders of
magnitude of genuine agreement and would stop the sweeps from detecting a
dropped argument at all.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.lookback_option import (
    ContinuousFloatingLookbackOption,
    ContinuousPartialFixedLookbackOption,
    ContinuousPartialFloatingLookbackOption,
)
from pquantlib.payoffs import (
    CashOrNothingPayoff,
    FloatingTypePayoff,
    OptionType,
    PlainVanillaPayoff,
)
from pquantlib.pricingengines.lookback.analytic_continuous_floating_lookback_engine import (
    AnalyticContinuousFloatingLookbackEngine,
)
from pquantlib.pricingengines.lookback.analytic_continuous_partial_fixed_lookback_engine import (
    AnalyticContinuousPartialFixedLookbackEngine,
)
from pquantlib.pricingengines.lookback.analytic_continuous_partial_floating_lookback_engine import (
    AnalyticContinuousPartialFloatingLookbackEngine,
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
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

TODAY: Final[Date] = Date.from_ymd(1, Month.March, 2025)
EXPIRY: Final[Date] = Date.from_ymd(1, Month.March, 2026)

# The skew used by the ``*_smile`` cases: 5 strikes at the single exercise
# tenor. It exists so that "which strike does the engine look the vol up at"
# becomes observable — with a flat vol it never is.
SMILE_STRIKES: Final[list[float]] = [80.0, 90.0, 100.0, 110.0, 120.0]
SMILE_VOLS: Final[list[float]] = [0.34, 0.29, 0.25, 0.22, 0.20]


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/inst/lookbackvarswap")


# --- market construction: mirrors the probe exactly --------------------------


def _smile_vol() -> BlackVolTermStructure:
    return BlackVarianceSurface(
        reference_date=TODAY,
        calendar=NullCalendar(),
        dates=[EXPIRY],
        strikes=SMILE_STRIKES,
        black_vol_matrix=np.asarray([[v] for v in SMILE_VOLS], dtype=np.float64),
        day_counter=Actual365Fixed(),
    )


def _process(spot: float, r: float, q: float, vol: float, smile: bool) -> GeneralizedBlackScholesProcess:
    day_counter = Actual365Fixed()
    black_vol: BlackVolTermStructure = (
        _smile_vol()
        if smile
        else BlackConstantVol(
            reference_date=TODAY,
            calendar=NullCalendar(),
            day_counter=day_counter,
            volatility=vol,
        )
    )
    return BlackScholesMertonProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(reference_date=TODAY, forward_rate=q, day_counter=day_counter),
        risk_free_ts=FlatForward.from_rate(reference_date=TODAY, forward_rate=r, day_counter=day_counter),
        black_vol_ts=black_vol,
    )


def _process_from(inputs: dict[str, Any]) -> GeneralizedBlackScholesProcess:
    return _process(
        float(inputs["spot"]),
        float(inputs["r"]),
        float(inputs["q"]),
        float(inputs["vol"]),
        bool(inputs["smile"]),
    )


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _floating_option(inputs: dict[str, Any]) -> ContinuousPartialFloatingLookbackOption:
    return ContinuousPartialFloatingLookbackOption(
        float(inputs["minmax"]),
        float(inputs["lambda"]),
        Date(int(inputs["lookback_end_serial"])),
        FloatingTypePayoff(_option_type(str(inputs["option_type"]))),
        EuropeanExercise(Date(int(inputs["expiry_serial"]))),
    )


def _fixed_option(inputs: dict[str, Any]) -> ContinuousPartialFixedLookbackOption:
    return ContinuousPartialFixedLookbackOption(
        Date(int(inputs["lookback_start_serial"])),
        PlainVanillaPayoff(_option_type(str(inputs["option_type"])), float(inputs["strike"])),
        EuropeanExercise(Date(int(inputs["expiry_serial"]))),
    )


def _cases(cpp: dict[str, Any], kind: str) -> list[tuple[str, dict[str, Any]]]:
    return [(k, v) for k, v in cpp.items() if v["inputs"].get("kind") == kind]


def _check_market_wiring(
    process: GeneralizedBlackScholesProcess,
    inputs: dict[str, Any],
    expected: dict[str, Any],
    lookup_strike: float,
    name: str,
) -> None:
    """The inputs the closed form is built on, pinned separately from the NPV.

    An NPV can match while two errors cancel; these turn a whole column of NPV
    failures into a single obvious "the vol lookup strike is wrong" or "the
    lookback boundary landed on the wrong day".
    """
    lookback_key = "lookback_end_serial" if "lookback_end_serial" in inputs else "lookback_start_serial"
    t = process.time(Date(int(inputs["expiry_serial"])))
    tight(t, float(expected["residual_time"]), reason=f"{name}: residual time")
    tight(
        process.time(Date(int(inputs[lookback_key]))),
        float(expected["lookback_time"]),
        reason=f"{name}: lookback boundary time",
    )
    tight(process.risk_free_rate().discount(t), float(expected["risk_free_discount"]))
    tight(process.dividend_yield().discount(t), float(expected["dividend_discount"]))
    tight(
        process.black_volatility().black_vol_at_time(t, lookup_strike),
        float(expected["vol_used"]),
        reason=f"{name}: volatility lookup strike",
    )


# --- partial-time FLOATING strike --------------------------------------------


def test_partial_floating_npv(cpp: dict[str, Any]) -> None:
    """Every partial-time floating case in the probe.

    The sweep moves lambda, the lookback end date, the running extremum, spot,
    vol and both rates one at a time, and covers both engine branches: the
    seven-term expression when the window ends before expiry and the three-term
    "simpler calculation" when it runs to expiry.
    """
    checked = 0
    for name, case in _cases(cpp, "partial_floating"):
        inputs, expected = case["inputs"], case["expected"]
        process = _process_from(inputs)
        option = _floating_option(inputs)
        option.set_pricing_engine(AnalyticContinuousPartialFloatingLookbackEngine(process))
        tight(option.npv(), float(expected["npv"]), reason=name)
        _check_market_wiring(process, inputs, expected, float(inputs["minmax"]), name)
        checked += 1
    assert checked > 40, f"expected a partial-floating sweep, got {checked} cases"


def test_partial_floating_arguments_carry_lambda_and_end_date(
    cpp: dict[str, Any],
) -> None:
    """``lambda`` and ``lookback_period_end`` reach the engine arguments.

    This is the argument-dropping guard at the plumbing level: the NPV sweep
    above proves they reach the *formula*, this proves they reach the carrier
    (with non-default values) and survive ``setup_arguments``.
    """
    checked = 0
    for name, case in _cases(cpp, "partial_floating"):
        inputs, expected = case["inputs"], case["expected"]
        option = _floating_option(inputs)
        engine = AnalyticContinuousPartialFloatingLookbackEngine(_process_from(inputs))
        option.set_pricing_engine(engine)
        option.npv()
        args = engine.get_arguments()
        assert args.minmax == float(expected["args_minmax"]), name
        assert args.lambda_ == float(expected["args_lambda"]), name
        assert args.lookback_period_end.serial_number() == int(expected["args_lookback_end_serial"]), name
        # And the instrument's own inspectors agree with what it handed over.
        assert option.lambda_() == args.lambda_
        assert option.lookback_period_end() == args.lookback_period_end
        assert option.minmax() == args.minmax
        checked += 1
    assert checked > 40


def test_partial_floating_degenerate_equals_plain_lookback(cpp: dict[str, Any]) -> None:
    """lambda == 1 with the window running to expiry IS the plain lookback.

    This is the one configuration where a port that dropped both extra
    arguments would be right, and it is the value such a port would return
    everywhere. Asserting it against the *existing* Conze-Viswanathan engine
    (not just against C++) fixes the diagnosis: if this passes and the sweep
    above fails, the arguments are being ignored.
    """
    for option_type in ("call", "put"):
        case = cpp[f"pflt_{option_type}_degenerate"]
        inputs, expected = case["inputs"], case["expected"]
        assert float(inputs["lambda"]) == 1.0
        assert int(inputs["lookback_end_serial"]) == int(inputs["expiry_serial"])

        plain = ContinuousFloatingLookbackOption(
            float(inputs["minmax"]),
            FloatingTypePayoff(_option_type(str(inputs["option_type"]))),
            EuropeanExercise(Date(int(inputs["expiry_serial"]))),
        )
        plain.set_pricing_engine(AnalyticContinuousFloatingLookbackEngine(_process_from(inputs)))
        tight(plain.npv(), float(expected["npv"]), reason=f"pflt_{option_type}")


def test_partial_floating_fills_no_greeks(cpp: dict[str, Any]) -> None:
    """The engine assigns only ``value``; every greek accessor must raise."""
    inputs = cpp["pflt_call_base"]["inputs"]
    option = _floating_option(inputs)
    option.set_pricing_engine(AnalyticContinuousPartialFloatingLookbackEngine(_process_from(inputs)))
    option.npv()
    for greek, accessor in (
        ("delta", option.delta),
        ("gamma", option.gamma),
        ("theta", option.theta),
        ("vega", option.vega),
        ("rho", option.rho),
        ("dividendRho", option.dividend_rho),
    ):
        assert cpp[f"pflt_greeks_unavailable_{greek}"]["expected"]["raises"] is True
        with pytest.raises(LibraryException, match="not provided"):
            accessor()


# --- partial-time FIXED strike -----------------------------------------------


def test_partial_fixed_npv(cpp: dict[str, Any]) -> None:
    """Every partial-time fixed case in the probe.

    The sweep moves the lookback start date, the strike, spot, vol and both
    rates one at a time, and covers both settings of
    ``different_start_of_lookback``.
    """
    checked = 0
    for name, case in _cases(cpp, "partial_fixed"):
        inputs, expected = case["inputs"], case["expected"]
        process = _process_from(inputs)
        option = _fixed_option(inputs)
        option.set_pricing_engine(AnalyticContinuousPartialFixedLookbackEngine(process))
        tight(option.npv(), float(expected["npv"]), reason=name)
        _check_market_wiring(process, inputs, expected, float(inputs["strike"]), name)
        checked += 1
    assert checked > 40, f"expected a partial-fixed sweep, got {checked} cases"


def test_partial_fixed_arguments_carry_start_date_and_zero_minmax(
    cpp: dict[str, Any],
) -> None:
    """``lookback_period_start`` reaches the arguments; ``minmax`` is a hard 0.

    The fixed partial-time variant takes no running-extremum argument at all —
    C++ forwards a literal ``0`` to the base class — so a port that grew one
    would show up here.
    """
    checked = 0
    for name, case in _cases(cpp, "partial_fixed"):
        inputs, expected = case["inputs"], case["expected"]
        option = _fixed_option(inputs)
        engine = AnalyticContinuousPartialFixedLookbackEngine(_process_from(inputs))
        option.set_pricing_engine(engine)
        option.npv()
        args = engine.get_arguments()
        assert args.minmax == float(expected["args_minmax"]) == 0.0, name
        assert args.lookback_period_start.serial_number() == int(expected["args_lookback_start_serial"]), name
        assert option.lookback_period_start() == args.lookback_period_start
        assert option.minmax() == 0.0
        checked += 1
    assert checked > 40


def test_partial_fixed_fills_no_greeks(cpp: dict[str, Any]) -> None:
    """The engine assigns only ``value``; every greek accessor must raise."""
    inputs = cpp["pfix_call_base"]["inputs"]
    option = _fixed_option(inputs)
    option.set_pricing_engine(AnalyticContinuousPartialFixedLookbackEngine(_process_from(inputs)))
    option.npv()
    for greek, accessor in (
        ("delta", option.delta),
        ("gamma", option.gamma),
        ("theta", option.theta),
        ("vega", option.vega),
        ("rho", option.rho),
        ("dividendRho", option.dividend_rho),
    ):
        assert cpp[f"pfix_greeks_unavailable_{greek}"]["expected"]["raises"] is True
        with pytest.raises(LibraryException, match="not provided"):
            accessor()


# --- engine guards -----------------------------------------------------------


def _flat_market(spot: float = 100.0) -> GeneralizedBlackScholesProcess:
    return _process(spot, 0.05, 0.02, 0.25, False)


def test_floating_engine_rejects_nonpositive_underlying(cpp: dict[str, Any]) -> None:
    assert cpp["pflt_engine_nonpositive_underlying"]["expected"]["raises"] is True
    option = ContinuousPartialFloatingLookbackOption(
        100.0,
        1.0,
        Date.from_ymd(1, Month.September, 2025),
        FloatingTypePayoff(OptionType.Call),
        EuropeanExercise(EXPIRY),
    )
    option.set_pricing_engine(AnalyticContinuousPartialFloatingLookbackEngine(_flat_market(0.0)))
    with pytest.raises(LibraryException, match="negative or null underlying"):
        option.npv()


def test_fixed_engine_rejects_nonpositive_underlying(cpp: dict[str, Any]) -> None:
    assert cpp["pfix_engine_nonpositive_underlying"]["expected"]["raises"] is True
    option = ContinuousPartialFixedLookbackOption(
        Date.from_ymd(1, Month.September, 2025),
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(EXPIRY),
    )
    option.set_pricing_engine(AnalyticContinuousPartialFixedLookbackEngine(_flat_market(0.0)))
    with pytest.raises(LibraryException, match="negative or null underlying"):
        option.npv()


def test_fixed_engine_rejects_non_plain_payoff(cpp: dict[str, Any]) -> None:
    """A striked payoff is not enough — C++ requires a PLAIN vanilla one."""
    assert cpp["pfix_engine_non_plain_payoff"]["expected"]["raises"] is True
    option = ContinuousPartialFixedLookbackOption(
        Date.from_ymd(1, Month.September, 2025),
        CashOrNothingPayoff(OptionType.Call, 100.0, 10.0),
        EuropeanExercise(EXPIRY),
    )
    option.set_pricing_engine(AnalyticContinuousPartialFixedLookbackEngine(_flat_market()))
    with pytest.raises(LibraryException, match="Non-plain payoff"):
        option.npv()


def test_fixed_engine_strike_guard_is_asymmetric(cpp: dict[str, Any]) -> None:
    """Zero strike: legal for a call (``>= 0``), illegal for a put (``> 0``)."""
    assert cpp["pfix_engine_negative_strike_call"]["expected"]["raises"] is True
    assert cpp["pfix_engine_zero_strike_put"]["expected"]["raises"] is True
    assert cpp["pfix_engine_negative_strike_put"]["expected"]["raises"] is True

    lookback_start = Date.from_ymd(1, Month.September, 2025)

    negative_call = ContinuousPartialFixedLookbackOption(
        lookback_start, PlainVanillaPayoff(OptionType.Call, -10.0), EuropeanExercise(EXPIRY)
    )
    negative_call.set_pricing_engine(AnalyticContinuousPartialFixedLookbackEngine(_flat_market()))
    with pytest.raises(LibraryException, match="Strike must be positive or null"):
        negative_call.npv()

    for strike in (0.0, -10.0):
        put = ContinuousPartialFixedLookbackOption(
            lookback_start,
            PlainVanillaPayoff(OptionType.Put, strike),
            EuropeanExercise(EXPIRY),
        )
        put.set_pricing_engine(AnalyticContinuousPartialFixedLookbackEngine(_flat_market()))
        with pytest.raises(LibraryException, match="Strike must be positive"):
            put.npv()
