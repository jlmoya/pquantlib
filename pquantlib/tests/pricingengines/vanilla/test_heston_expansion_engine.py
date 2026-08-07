"""Cross-validate HestonExpansionEngine and its three expansions against C++ v1.43.

Reference: ``migration-harness/references/v143/pe/heston`` (``expansion_*``
cases, 472 of them), produced by ``migration-harness/cpp/probes/
v143_pe_heston/probe.cpp``. Every one is asserted here;
``test_every_expansion_case_is_covered`` fails if the probe grows a case this
module does not read.

What these tests are actually defending
---------------------------------------
1. **All three formulas, through the engine AND standalone.** ``LPP2``,
   ``LPP3`` and ``Forde`` are each priced through ``HestonExpansionEngine``
   over two markets / three parameter sets / four maturities / five strikes /
   both option types, and each ``HestonExpansion`` subclass is *also* queried
   directly through ``impliedVolatility(strike, forward)`` — which is how they
   are used during calibration, one expansion per expiry and many strikes, with
   no market involved at all.
2. **That the enum is not accepted and discarded.** At 130 of the 150
   (market, params, maturity, type, strike) groups C++ gives three prices that
   differ by more than 1e-8 relative; ``test_the_formula_selector_changes_the
   _price`` requires the port to produce the same number of distinct prices in
   every group.
3. **Forde's small-time regime and its long-maturity breakdown.** Terms span
   1 week to 10 years. The C++ test-suite's own comment says Forde "breaks down
   for long maturities" and allows 100% error there; the reference pins the
   breakdown exactly and this module reproduces it rather than repairing it.
4. **The two different clamps.** LPP2/LPP3 return ``max(1e-8, vol)`` — the
   clamp is on the VOLATILITY. Forde returns ``sqrt(max(1e-8, var))`` — the
   clamp is on the VARIANCE, so its floor is 1e-4, not 1e-8. The reference
   contains 35 clamped values (33 Forde at 1e-4, one LPP2 and one LPP3 at
   1e-8); ``test_the_two_clamps_are_reproduced_exactly`` asserts every one of
   them bit-exactly and asserts that Forde never returns 1e-8, which is what a
   port that clamped after the square root would produce.
5. **The engine's wiring.** ``blackFormula(payoff, forward, vol*sqrt(term),
   riskFreeDiscount, 0)`` — displacement zero, the RISK-FREE discount, and a
   forward that already carries the dividend discount. Asserted bit-exactly for
   all 450 priced cases against an independently assembled Black price.
6. **That no Greeks are invented** and both guards fire.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel) everywhere except the *leading* LPP coefficient
at small ``kappa*term``, where the pinned C++ digits are themselves not
accurate to TIGHT.

``LPP2HestonExpansion::z2`` (hestonexpansionengine.cpp:184) and
``LPP3HestonExpansion::z3`` (.cpp:660) are Mathematica-generated rational
functions whose numerators, at small ``kappa*t``, are near-total cancellations.
Re-evaluating the *same* expressions at 60 significant digits (mpmath) gives,
for this port's double evaluation:

    kappa*term  0.022    0.038    0.077  | 0.115    0.200    0.400  | >= 1.15
    LPP2 |dz2|  1.7e-11  2.2e-11  6.2e-11| 2.3e-13  2.7e-13  2.1e-13| < 2e-16
    LPP3 |dz3|  4.1e-5   2.6e-5   8.1e-5 | 5.3e-9   4.7e-10  7.2e-10| < 3e-15

The disagreement with C++ lives *entirely* in that one coefficient. Dividing
the observed volatility gaps by ``|log(K/F)|**order`` gives the same number at
every strike to three or four significant figures — LPP2: 1.86e-11 (forde),
2.70e-11 (lewis), 2.73e-11 (fellerbad); LPP3: 5.95e-6 / 5.68e-5 / 2.00e-4 — at
``term = 1/52``. A transliteration error would not have that shape, and would
not vanish to 1e-15 by ``term = 1``, which it does for every parameter set.
Since the port's own ``z3`` error is 4.1e-5 / 8.1e-5 / 2.6e-5 there, C++'s must
run to 2.0e-4: at ``fellerbad, term = 1/52, K = 140`` the pinned C++ implied
vol is 0.026497681 against a true 0.026506281 — wrong in its fourth significant
figure, and less accurate than this port's 0.026505308. Demanding TIGHT would
demand reproduction of C++'s rounding, not of its mathematics.

The relaxation is therefore expressed as an error in the leading coefficient,
propagated:

* implied volatility: ``abs_tol = 1e-14 + dz * |log(K/F)|**order``
* NPV: ``abs_tol = 1e-14 + |Black(vol + dv) - Black(vol - dv)| / 2`` with
  ``dv = dz * |log(K/F)|**order``, i.e. the vega of that volatility
  uncertainty at the case's own forward, discount and term

with ``order`` 2 for LPP2 and 3 for LPP3, and

    dz          kappa*term < 0.1     0.1 <= kappa*term < 1   >= 1
    LPP2        2e-10 (3x of 6.2e-11) 1e-12 (4x of 2.7e-13)  0
    LPP3        5e-4  (2.5x of 2e-4)  2e-8  (4x of 5.3e-9)   0

Forde is unconditionally TIGHT: its coefficients are direct polynomials in
``(v0, sigma, rho, kappa, theta)`` with no cancellation, and all 133 pinned
Forde values hold TIGHT. ``dz = 0`` — i.e. plain TIGHT — also covers every LPP
case at ``kappa*term >= 1``, which is all of 1y and 5y and every wing case. The
relative criterion stays 1e-12 throughout; the bound collapses to TIGHT at the
money (``x = 0``) and wherever the ``max(1e-8, vol)`` clamp is active, since
both implementations then return exactly the clamp.

The widest bound this produces anywhere is 1.7e-4 absolute on a price of 2.72
(``expansion_npv_A_lewis_1w_LPP3_Call_k160``, observed gap 1.5e-5); the
narrowest non-zero one is 4.5e-9 absolute on a price of 0.115
(``expansion_npv_A_fellerbad_3m_LPP3_Put_k60``, observed gap 4.6e-12).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import AmericanExercise, EuropeanExercise
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import CashOrNothingPayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.pricingengines.vanilla.heston_expansion_engine import (
    FordeHestonExpansion,
    HestonExpansion,
    HestonExpansionEngine,
    HestonExpansionFormula,
    LPP2HestonExpansion,
    LPP3HestonExpansion,
)
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month

CPP: dict[str, Any] = reference_reader.load("v143/pe/heston")

# probe.cpp:216 — const Date kToday(1, March, 2025);
TODAY: Date = Date.from_ymd(1, Month.March, 2025)
DAY_COUNTER: DayCounter = Actual365Fixed()

# probe.cpp:242 — const Market kMarketA{"A", 100.0, 0.05, 0.02}. The
# ``expansion_iv_*`` cases take an explicit ``forward`` and carry no market;
# only the model parameters matter to them.
MARKET_A_SPOT: float = 100.0
MARKET_A_R: float = 0.05
MARKET_A_Q: float = 0.02

# See "Tolerance" in the module docstring: the double-evaluation error of each
# expansion's LEADING polynomial coefficient, banded by kappa*term as
# (kappa*term < 0.1, 0.1 <= kappa*term < 1), together with the power of
# log(K/F) that carries it into the volatility.
_LEADING_COEFFICIENT_ORDER: dict[str, int] = {"LPP2": 2, "LPP3": 3, "Forde": 0}
_LEADING_COEFFICIENT_NOISE: dict[str, tuple[float, float]] = {
    "LPP2": (2.0e-10, 1.0e-12),
    "LPP3": (5.0e-4, 2.0e-8),
    "Forde": (0.0, 0.0),
}
_LPP_VOL_CLAMP: float = 1e-8
_FORDE_VOL_FLOOR: float = 1e-4  # sqrt(max(1e-8, var))

# ``(kappa, theta, sigma, v0, rho, term) -> HestonExpansion``. Spelled as a
# factory rather than ``type[HestonExpansion]`` because the abstract base takes
# no constructor arguments; all three concretes take these six, in this order.
ExpansionFactory = Callable[[float, float, float, float, float, float], HestonExpansion]

EXPANSION_CLASSES: dict[str, ExpansionFactory] = {
    "LPP2": LPP2HestonExpansion,
    "LPP3": LPP3HestonExpansion,
    "Forde": FordeHestonExpansion,
}
VOL_FIELDS: dict[str, str] = {
    "lpp2_vols": "LPP2",
    "lpp3_vols": "LPP3",
    "forde_vols": "Forde",
}

IV_CASES: list[str] = sorted(n for n in CPP if n.startswith("expansion_iv_"))
NPV_CASES: list[str] = sorted(n for n in CPP if n.startswith("expansion_npv_"))
GUARD_CASES: list[str] = [
    "expansion_no_greeks",
    "expansion_rejects_non_plain_vanilla_payoff",
    "expansion_rejects_non_european_exercise",
]

GREEK_ACCESSORS: tuple[str, ...] = (
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "dividend_rho",
)


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """probe.cpp:1365 — ``Settings::instance().evaluationDate() = kToday``."""
    settings = ObservableSettings()
    saved = settings.evaluation_date
    settings.evaluation_date = TODAY
    yield
    settings.evaluation_date = saved


def _flat_curve(rate: float) -> FlatForward:
    """probe.cpp:223-226 — ``FlatForward(kToday, r, Actual365Fixed())``."""
    return FlatForward.from_rate(
        reference_date=TODAY, forward_rate=rate, day_counter=DAY_COUNTER
    )


def _heston_process(inputs: dict[str, Any]) -> HestonProcess:
    """probe.cpp:284-287 — ``hestonProcess(market, params)``."""
    return HestonProcess(
        risk_free_rate=_flat_curve(inputs.get("r", MARKET_A_R)),
        dividend_yield=_flat_curve(inputs.get("q", MARKET_A_Q)),
        s0=SimpleQuote(inputs.get("s0", MARKET_A_SPOT)),
        v0=inputs["v0"],
        kappa=inputs["kappa"],
        theta=inputs["theta"],
        sigma=inputs["sigma"],
        rho=inputs["rho"],
    )


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _maturity_date(inputs: dict[str, Any]) -> Date:
    return TODAY + int(inputs["maturity_days"])


def _expansion(formula: str, inputs: dict[str, Any], term: float) -> HestonExpansion:
    """One expansion for one expiry.

    # C++ parity: hestonexpansionengine.cpp:66-98 — the ctor argument order is
    # ``(kappa, theta, sigma, v0, rho, term)``, which is NOT the order the
    # private ``z*`` helpers receive.
    """
    return EXPANSION_CLASSES[formula](
        inputs["kappa"], inputs["theta"], inputs["sigma"], inputs["v0"], inputs["rho"], term
    )


def _priced_option(inputs: dict[str, Any]) -> VanillaOption:
    option = VanillaOption(
        PlainVanillaPayoff(_option_type(inputs["type"]), inputs["strike"]),
        EuropeanExercise(_maturity_date(inputs)),
    )
    option.set_pricing_engine(
        HestonExpansionEngine(
            HestonModel(_heston_process(inputs)),
            HestonExpansionFormula[inputs["formula"]],
        )
    )
    return option


def _volatility_uncertainty(
    formula: str, kappa_term: float, strike: float, forward: float
) -> float:
    """``dz * |log(K/F)|**order`` — the leading coefficient's rounding, propagated.

    Zero (i.e. plain TIGHT) for Forde at any term, for any expansion at
    ``kappa*term >= 1``, and at the money. See "Tolerance" in the module
    docstring for the 60-digit measurements the ``dz`` values are read off.
    """
    cancellation_limited, transitional = _LEADING_COEFFICIENT_NOISE[formula]
    if kappa_term < 0.1:
        noise = cancellation_limited
    elif kappa_term < 1.0:
        noise = transitional
    else:
        noise = 0.0
    if noise == 0.0:
        return 0.0
    return noise * abs(math.log(strike / forward)) ** _LEADING_COEFFICIENT_ORDER[formula]


def test_every_expansion_case_is_covered() -> None:
    """No ``expansion_*`` case may be silently skipped."""
    covered = set(IV_CASES) | set(NPV_CASES) | set(GUARD_CASES)
    assert covered == {n for n in CPP if n.startswith("expansion_")}
    assert len(covered) == 472


@pytest.mark.parametrize("case_name", IV_CASES)
@pytest.mark.parametrize("field", sorted(VOL_FIELDS))
def test_implied_volatility_matches_cpp(case_name: str, field: str) -> None:
    """Each ``HestonExpansion`` subclass, called directly, over the strike grid.

    No market is involved: ``impliedVolatility(strike, forward)`` takes the
    forward as an argument, so these cases pin the expansions themselves rather
    than the engine that wraps them.
    """
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    assert set(expected) == set(VOL_FIELDS), f"{case_name} pins an unread field"

    formula = VOL_FIELDS[field]
    term = inputs["term"]
    forward = inputs["forward"]
    expansion = _expansion(formula, inputs, term)
    kappa_term = inputs["kappa"] * term

    for strike, want in zip(inputs["strikes"], expected[field], strict=True):
        got = expansion.implied_volatility(strike, forward)
        if want in (_LPP_VOL_CLAMP, _FORDE_VOL_FLOOR):
            # The clamp is active: both implementations return the constant, so
            # there is no rounding to accommodate.
            tolerance.exact(got, want, reason=f"{case_name}.{field}[K={strike}] clamp")
            continue
        # abs_tol = TIGHT floor + the leading coefficient's rounding carried
        # through |log(K/F)|**order. Zero for Forde and for every
        # kappa*term >= 1, so this is plain TIGHT for 91 of the 133 values.
        uncertainty = _volatility_uncertainty(formula, kappa_term, strike, forward)
        tolerance.custom(
            got,
            want,
            abs_tol=1e-14 + uncertainty,
            rel_tol=1e-12,
            reason=(
                f"{case_name}.{field}[K={strike}]: kappa*term={kappa_term:.4g}, "
                f"leading-coefficient volatility uncertainty {uncertainty:.3e}"
            ),
        )


def test_the_two_clamps_are_reproduced_exactly() -> None:
    """LPP2/LPP3 floor the volatility at 1e-8; Forde floors the VARIANCE.

    # C++ parity: ``max(1e-8, vol)`` in LPP2/LPP3::impliedVolatility
    # (.cpp:121-126, :733-738) versus ``sqrt(max(1e-8, var))`` in
    # FordeHestonExpansion::impliedVolatility (.cpp:216-222). Forde's floor is
    # therefore 1e-4. A port that clamped Forde *after* the square root would
    # return 1e-8 wherever C++ returns 1e-4, so both the presence of 1e-4 and
    # the absence of 1e-8 in the Forde column are asserted.
    """
    hits = {"lpp2_vols": 0, "lpp3_vols": 0, "forde_vols": 0}
    for case_name in IV_CASES:
        inputs, expected = CPP[case_name]["inputs"], CPP[case_name]["expected"]
        for field, formula in VOL_FIELDS.items():
            expansion = _expansion(formula, inputs, inputs["term"])
            for strike, want in zip(inputs["strikes"], expected[field], strict=True):
                if formula == "Forde":
                    assert want != _LPP_VOL_CLAMP, (
                        f"{case_name}[K={strike}]: C++ Forde returned the 1e-8 "
                        "volatility clamp — it clamps the variance, not the vol"
                    )
                floor = _FORDE_VOL_FLOOR if formula == "Forde" else _LPP_VOL_CLAMP
                if want != floor:
                    continue
                hits[field] += 1
                tolerance.exact(
                    expansion.implied_volatility(strike, inputs["forward"]),
                    want,
                    reason=f"{case_name}.{field}[K={strike}]",
                )

    assert hits == {"lpp2_vols": 1, "lpp3_vols": 1, "forde_vols": 33}


def test_forde_breaks_down_at_long_maturities() -> None:
    """The reference pins the breakdown; the port must reproduce, not repair it.

    At ``term = 10`` with the Lewis parameters Forde's quartic in ``x`` goes
    negative over most of the strike grid and the ``max(1e-8, var)`` floor takes
    over, while LPP2/LPP3 stay in the 0.4-0.6 range. That is the behaviour the
    C++ test-suite documents with "forde breaks down for long maturities".
    """
    case = CPP["expansion_iv_lewis_term10"]
    inputs, expected = case["inputs"], case["expected"]
    floored = [v for v in expected["forde_vols"] if v == _FORDE_VOL_FLOOR]
    assert len(floored) == 4, "the reference no longer exhibits the breakdown"
    assert all(0.2 < v < 1.5 for v in expected["lpp2_vols"])

    forde = _expansion("Forde", inputs, inputs["term"])
    for strike, want in zip(inputs["strikes"], expected["forde_vols"], strict=True):
        got = forde.implied_volatility(strike, inputs["forward"])
        tolerance.tight(got, want, reason=f"K={strike}")


@pytest.mark.parametrize("case_name", NPV_CASES)
def test_npv_matches_cpp(case_name: str) -> None:
    """Price through the engine, all three formulas, both markets."""
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    got = _priced_option(inputs).npv()

    process = _heston_process(inputs)
    maturity_date = _maturity_date(inputs)
    term = process.time(maturity_date)
    discount = process.risk_free_rate().discount(maturity_date)
    forward = process.s0().value() * process.dividend_yield().discount(maturity_date) / discount
    strike = inputs["strike"]
    delta_vol = _volatility_uncertainty(
        inputs["formula"], inputs["kappa"] * term, strike, forward
    )
    vol = _expansion(inputs["formula"], inputs, term).implied_volatility(strike, forward)
    if delta_vol == 0.0 or vol <= _LPP_VOL_CLAMP:
        # Either the expansion is well conditioned at this kappa*term, or the
        # max(1e-8, vol) clamp is active and both implementations return
        # exactly 1e-8, so no coefficient rounding can reach the price.
        tolerance.tight(got, expected["npv"], reason=case_name)
        return

    # Propagate the coefficient uncertainty through Black at this case's own
    # forward, discount and term: half the spread between the two perturbed
    # prices is the vega of that volatility uncertainty.
    option_type = _option_type(inputs["type"])
    sqrt_term = math.sqrt(term)
    upper = black_formula(
        option_type, strike, forward, (vol + delta_vol) * sqrt_term, discount, 0.0
    )
    lower = black_formula(
        option_type, strike, forward, max(0.0, vol - delta_vol) * sqrt_term, discount, 0.0
    )
    tolerance.custom(
        got,
        expected["npv"],
        abs_tol=1e-14 + 0.5 * abs(upper - lower),
        rel_tol=1e-12,
        reason=(
            f"{case_name}: kappa*term={inputs['kappa'] * term:.4g}, "
            f"delta_vol={delta_vol:.3e} from the leading-coefficient bound"
        ),
    )


@pytest.mark.parametrize("case_name", NPV_CASES)
def test_engine_price_is_black_of_the_expansion_volatility(case_name: str) -> None:
    """``blackFormula(payoff, forward, vol*sqrt(term), riskFreeDiscount, 0)``.

    # C++ parity: hestonexpansionengine.cpp:101-103. Three things are wrong
    # often enough to be worth pinning bit-exactly: the discount is the
    # RISK-FREE one (the dividend discount is already inside ``forward``), the
    # displacement is 0, and the volatility comes from the standalone expansion
    # at ``(strike, forward)`` — the same object a caller would build directly.
    """
    inputs = CPP[case_name]["inputs"]
    process = _heston_process(inputs)
    maturity_date = _maturity_date(inputs)
    term = process.time(maturity_date)
    discount = process.risk_free_rate().discount(maturity_date)
    forward = process.s0().value() * process.dividend_yield().discount(maturity_date) / discount
    vol = _expansion(inputs["formula"], inputs, term).implied_volatility(
        inputs["strike"], forward
    )
    expected = black_formula(
        _option_type(inputs["type"]),
        inputs["strike"],
        forward,
        vol * math.sqrt(term),
        discount,
        0.0,
    )
    tolerance.exact(_priced_option(inputs).npv(), expected, reason=case_name)


def _distinct_price_count(prices: list[float]) -> int:
    """Prices differing by more than 1e-8 relative (1e-14 absolute near zero)."""
    kept: list[float] = []
    for price in sorted(prices):
        if not any(math.isclose(price, seen, rel_tol=1e-8, abs_tol=1e-14) for seen in kept):
            kept.append(price)
    return len(kept)


def test_the_formula_selector_changes_the_price() -> None:
    """LPP2 / LPP3 / Forde must genuinely disagree.

    An engine that stored ``HestonExpansionFormula`` and priced with one fixed
    expansion would return the same number three times. In 130 of the 150
    (market, params, maturity, type, strike) groups C++'s three prices are all
    distinct at 1e-8 relative; the remaining 20 are deep in the wings where all
    three collapse onto intrinsic or onto zero. The port must reproduce the
    same partition, group by group.
    """
    groups: dict[tuple[str, str, str, str, float], dict[str, str]] = {}
    for name in NPV_CASES:
        inputs = CPP[name]["inputs"]
        key = (
            inputs["market"],
            inputs["params"],
            inputs["maturity"],
            inputs["type"],
            inputs["strike"],
        )
        groups.setdefault(key, {})[inputs["formula"]] = name

    assert len(groups) == 150
    assert {frozenset(g) for g in groups.values()} == {frozenset({"LPP2", "LPP3", "Forde"})}

    all_distinct = 0
    for key, names in groups.items():
        cpp = [CPP[name]["expected"]["npv"] for name in names.values()]
        port = [_priced_option(CPP[name]["inputs"]).npv() for name in names.values()]
        cpp_count = _distinct_price_count(cpp)
        assert _distinct_price_count(port) == cpp_count, (
            f"{key}: C++ gives {cpp_count} distinct prices across the three "
            f"formulas, the port gives {_distinct_price_count(port)}"
        )
        all_distinct += cpp_count == 3

    assert all_distinct == 130


def test_no_greeks_are_provided() -> None:
    """C++ writes ``results_.value`` and nothing else (.cpp:48-104)."""
    case = CPP["expansion_no_greeks"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["has_any_greek"] is False
    assert inputs["formula"] == "LPP2"

    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        EuropeanExercise(_maturity_date(inputs)),
    )
    option.set_pricing_engine(
        HestonExpansionEngine(
            HestonModel(_heston_process(inputs)), HestonExpansionFormula.LPP2
        )
    )
    assert option.npv() > 0.0
    for accessor in GREEK_ACCESSORS:
        with pytest.raises(LibraryException):
            getattr(option, accessor)()


def test_rejects_non_plain_vanilla_payoff() -> None:
    """# C++ parity: ``QL_REQUIRE(payoff, "non plain vanilla payoff given")``.

    The probe uses the LPP3 formula for this case; the guard sits in
    ``calculate()`` ahead of the expansion, so the formula is irrelevant.
    """
    case = CPP["expansion_rejects_non_plain_vanilla_payoff"]
    assert case["inputs"]["payoff"] == "CashOrNothingPayoff"
    assert case["expected"]["throws"] is True

    option = VanillaOption(
        CashOrNothingPayoff(OptionType.Call, 100.0, 1.0),
        EuropeanExercise(TODAY + 365),
    )
    option.set_pricing_engine(
        HestonExpansionEngine(
            HestonModel(_heston_process(CPP["expansion_no_greeks"]["inputs"])),
            HestonExpansionFormula.LPP3,
        )
    )
    with pytest.raises(LibraryException, match="non plain vanilla payoff given"):
        option.npv()


def test_rejects_non_european_exercise() -> None:
    """# C++ parity: ``QL_REQUIRE(exercise->type() == Exercise::European)``."""
    case = CPP["expansion_rejects_non_european_exercise"]
    assert case["inputs"]["exercise"] == "AmericanExercise"
    assert case["expected"]["throws"] is True

    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + 365),
    )
    option.set_pricing_engine(
        HestonExpansionEngine(
            HestonModel(_heston_process(CPP["expansion_no_greeks"]["inputs"])),
            HestonExpansionFormula.Forde,
        )
    )
    with pytest.raises(LibraryException, match="not an European option"):
        option.npv()
