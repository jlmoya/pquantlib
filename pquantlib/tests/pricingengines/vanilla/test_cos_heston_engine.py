"""Cross-validate COSHestonEngine against the C++ v1.43 probe.

Reference: ``migration-harness/references/v143/pe/heston`` (``cos_*`` cases,
408 of them), produced by ``migration-harness/cpp/probes/v143_pe_heston/
probe.cpp``. Every one of those 408 is asserted here; ``test_every_cos_case_is
_covered`` fails if a future probe adds one this module does not read.

What these tests are actually defending
---------------------------------------
1. **The cumulant surface, directly.** ``c1``..``c4`` / ``mu`` / ``var`` /
   ``skew`` / ``kurtosis`` are public in C++ and are pinned on their own, with
   no market attached, because they depend only on
   ``(v0, kappa, theta, sigma, rho, t)``. They are the easiest place for this
   engine to be wrong invisibly: ``c1`` and ``c2`` only set the integration
   window ``[a, b]``, so a re-derived (rather than transliterated) ``c2`` still
   produces plausible-looking prices. ``c3`` and ``c4`` do not enter pricing at
   all in v1.43 and would otherwise never be exercised.
2. **That the cumulants are DRIFTLESS.** ``COSHestonEngine::muT()`` — the drift
   ``log(qDiscount / rDiscount)`` — is private and never called in v1.43;
   ``mu() == c1()`` is the driftless first cumulant. The probe emits the same
   four numbers for market A (r = 5%, q = 2%) and market B (r = 35%, q = 17%),
   and ``test_cumulants_ignore_spot_and_rates`` asserts the port does too,
   bit-for-bit. A port that "fixed" this by folding the drift into ``c1`` would
   look more correct and would fail here.
3. **That ``L`` and ``N`` are honoured.** The reference sweeps five ``(L, N)``
   pairs at 1y/market A, including ``N = 25`` — far too few terms to converge.
   An engine that accepts the knobs and prices at the defaults returns one
   number five times; ``test_truncation_width_and_term_count_are_honoured``
   counts distinct prices.
4. **The truncation-bound early return.** When ``x >= b/2`` or ``x <= a/2`` the
   cosine series is abandoned and the engine returns the discounted
   no-arbitrage bound — a bound, not a price. The ``cos_bound_*`` cases assert
   both the value and that the branch predicate genuinely fired, so landing
   near intrinsic by accident is not enough.
5. **That no Greeks are invented.** C++ writes ``results_.value`` and nothing
   else, so all six accessors must raise.

Tolerance
---------
TIGHT (1e-14 abs / 1e-12 rel) everywhere, with two derived relaxations. Both
are selected by a property of the *case* — not by whether the case passed.

**(a) NPV: absolute floor raised from 1e-14 to 1e-13, relative criterion left
at 1e-12.** The call branch is ``spot*qf - K*df*(1 - s)``
(``cos_heston_engine.py:372``; C++ ``coshestonengine.cpp:116-120``) — a
subtraction of two O(100) quantities. One ULP of the larger operand is
``160 * 2**-53 = 1.78e-14``, and ``s`` is itself an ``N``-term cosine sum, so
the result carries a few ULP of *absolute* rounding however little the option
is worth. At ``A/fellerbad/1w/Call/K=160`` C++ prints ``-2.35e-14`` and this
port ``-5.68e-14``: both are zero. Worst absolute deviation over the 364 pinned
prices is 5.02e-14 (``cos_npv_A_lewis_3m_Call_k160``, C++
``6.394230111432808e-3``); 1e-13 is that floor rounded up to the next decade.
Anything worth more than 1e-1 is still governed by the 1e-12 relative
criterion, which is 355 of the 364 cases.

**(b) c3 / c4 / skew / kurtosis when ``kappa*t < 0.1``: 1e-8 relative with NO
absolute escape hatch.** This is *stricter* than the LOOSE tier, which would
allow 1e-8 absolute on quantities of size 1e-6. ``c4`` is an O(t^4) quantity
assembled from O(1) terms over ``64 exp(4 kappa t) kappa^7``, so the relative
condition number of its double evaluation grows like ``(kappa*t)^-3``.
Re-evaluating the *same* closed form at 60 significant digits (mpmath) shows
the pinned C++ ``c4`` at ``t = 0.02`` is itself wrong by 7.5e-11 (lewis),
4.3e-10 (forde), 8.6e-11 (fellerbad) and 1.07e-9 (poscorr) relative, while this
port's ``c4`` is wrong by 7.4e-11 / 4.0e-10 / 9.8e-11 / 6.3e-11 — closer to the
truth than C++ in three of the four. Demanding TIGHT there would demand
reproduction of C++'s rounding noise, not of its mathematics. The largest
observable disagreement is 1.003e-9 (poscorr kurtosis) and 1e-8 is one decade
above it. Only the ``t = 0.02`` row of the cumulant grid satisfies
``kappa*t < 0.1``; ``t >= 0.5`` and ``c1``/``c2``/``mu``/``var`` at every ``t``
are asserted TIGHT.

Everything else — the 24 characteristic-function values, the four
truncation-bound cases, the guards — is TIGHT or bit-exact.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
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
from pquantlib.pricingengines.vanilla.cos_heston_engine import COSHestonEngine
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

# probe.cpp:242 — const Market kMarketA{"A", 100.0, 0.05, 0.02};
# The cumulant and chF cases carry no market at all (the quantities do not
# depend on one), but the C++ engine still needs a model to exist and the probe
# built it on market A. Rebuilt explicitly here rather than defaulted silently.
MARKET_A_SPOT: float = 100.0
MARKET_A_R: float = 0.05
MARKET_A_Q: float = 0.02

# See "Tolerance (a)" in the module docstring.
_NPV_ABS_FLOOR: float = 1e-13
# See "Tolerance (b)". kappa*t below this is the cancellation-limited regime.
_CANCELLATION_KAPPA_T: float = 0.1
_CANCELLATION_REL_TOL: float = 1e-8
_ILL_CONDITIONED_CUMULANTS: frozenset[str] = frozenset({"c3", "c4", "skew", "kurtosis"})

CUMULANT_CASES: list[str] = sorted(n for n in CPP if n.startswith("cos_cumulants_"))
CHF_CASES: list[str] = sorted(n for n in CPP if n.startswith("cos_chf_"))
NPV_CASES: list[str] = sorted(n for n in CPP if n.startswith("cos_npv_"))
BOUND_CASES: list[str] = sorted(n for n in CPP if n.startswith("cos_bound_"))
GUARD_CASES: list[str] = [
    "cos_no_greeks",
    "cos_rejects_non_plain_vanilla_payoff",
    "cos_rejects_non_european_exercise",
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


def _heston_model(inputs: dict[str, Any]) -> HestonModel:
    return HestonModel(_heston_process(inputs))


def _option_type(name: str) -> OptionType:
    return OptionType.Call if name == "Call" else OptionType.Put


def _maturity_date(inputs: dict[str, Any]) -> Date:
    return TODAY + int(inputs["maturity_days"])


def _priced_option(inputs: dict[str, Any]) -> VanillaOption:
    """Rebuild one priced ``cos_npv_*`` / ``cos_bound_*`` case."""
    option = VanillaOption(
        PlainVanillaPayoff(_option_type(inputs["type"]), inputs["strike"]),
        EuropeanExercise(_maturity_date(inputs)),
    )
    option.set_pricing_engine(
        COSHestonEngine(_heston_model(inputs), inputs["L"], int(inputs["N"]))
    )
    return option


def _cumulants(engine: COSHestonEngine, t: float) -> dict[str, float]:
    return {
        "c1": engine.c1(t),
        "c2": engine.c2(t),
        "c3": engine.c3(t),
        "c4": engine.c4(t),
        "mu": engine.mu(t),
        "var": engine.var(t),
        "skew": engine.skew(t),
        "kurtosis": engine.kurtosis(t),
    }


def _has_any_greek(option: VanillaOption) -> bool:
    """probe.cpp:325-337 — ``hasAnyGreek``: does *any* accessor return?"""
    for accessor in GREEK_ACCESSORS:
        try:
            getattr(option, accessor)()
        except LibraryException:
            continue
        return True
    return False


def test_every_cos_case_is_covered() -> None:
    """No ``cos_*`` case may be silently skipped.

    The four parametrized families plus the three named guards must exhaust the
    reference. If the probe grows a case, this fails rather than ignoring it.
    """
    covered = set(CUMULANT_CASES) | set(CHF_CASES) | set(NPV_CASES) | set(BOUND_CASES)
    covered |= set(GUARD_CASES)
    assert covered == {n for n in CPP if n.startswith("cos_")}
    assert len(covered) == 408


@pytest.mark.parametrize("case_name", CUMULANT_CASES)
def test_cumulants_match_cpp(case_name: str) -> None:
    """Every cumulant/moment the probe pins is reproduced.

    The cumulants take no market, no spot and no strike: they are pure
    functions of ``(v0, kappa, theta, sigma, rho, t)``.
    """
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    got = _cumulants(COSHestonEngine(_heston_model(inputs)), inputs["t"])
    assert set(expected) <= set(got), f"{case_name} pins a field this test does not read"

    cancellation_limited = inputs["kappa"] * inputs["t"] < _CANCELLATION_KAPPA_T
    for field, want in expected.items():
        if cancellation_limited and field in _ILL_CONDITIONED_CUMULANTS:
            # See "Tolerance (b)": at kappa*t < 0.1 the pinned C++ digits are
            # themselves only good to ~1e-9 relative, measured against a
            # 60-digit evaluation of the same closed form. Pure relative bound,
            # no absolute escape hatch (this is stricter than the LOOSE tier).
            tolerance.custom(
                got[field],
                want,
                abs_tol=0.0,
                rel_tol=_CANCELLATION_REL_TOL,
                reason=(
                    f"{case_name}.{field}: kappa*t="
                    f"{inputs['kappa'] * inputs['t']:.4g} — the Mathematica "
                    "closed form is cancellation-limited here and the pinned "
                    "C++ value carries up to 1.07e-9 relative rounding of its "
                    "own (verified at 60 digits)"
                ),
            )
        else:
            tolerance.tight(got[field], want, reason=f"{case_name}.{field}")


@pytest.mark.parametrize("case_name", CUMULANT_CASES)
def test_mu_and_var_are_c1_and_c2(case_name: str) -> None:
    """# C++ parity: ``mu`` / ``var`` forward verbatim (.cpp:325-330)."""
    inputs = CPP[case_name]["inputs"]
    engine = COSHestonEngine(_heston_model(inputs))
    t = inputs["t"]
    tolerance.exact(engine.mu(t), engine.c1(t))
    tolerance.exact(engine.var(t), engine.c2(t))


@pytest.mark.parametrize("case_name", CUMULANT_CASES)
def test_skew_and_kurtosis_are_the_normalised_cumulants(case_name: str) -> None:
    """# C++ parity: ``c3/pow(c2, 1.5)`` and ``c4/(c2*c2)`` (.cpp:331-336)."""
    inputs = CPP[case_name]["inputs"]
    engine = COSHestonEngine(_heston_model(inputs))
    t = inputs["t"]
    c2 = engine.c2(t)
    tolerance.exact(engine.skew(t), engine.c3(t) / math.pow(c2, 1.5))
    tolerance.exact(engine.kurtosis(t), engine.c4(t) / (c2 * c2))


def test_cumulants_ignore_spot_and_rates() -> None:
    """``muT()`` is private and never called: the cumulants are driftless.

    The probe emits the same four numbers for market A (r = 5%, q = 2%) and
    market B (r = 35%, q = 17%) at identical model parameters — its own pair is
    bit-identical, and so must the port's be. A port that folded the drift
    ``log(qDiscount/rDiscount)`` into ``c1`` would pass every other test in this
    module at ``r == q`` and fail here.
    """
    market_a = CPP["cos_cumulants_fellerbad_t1"]
    market_b = CPP["cos_cumulants_marketB_fellerbad_t1"]
    assert market_b["inputs"]["market"] == "B"
    assert market_b["inputs"]["r"] != MARKET_A_R
    assert market_b["inputs"]["q"] != MARKET_A_Q

    for field in ("c1", "c2", "c3", "c4"):
        tolerance.exact(
            market_a["expected"][field],
            market_b["expected"][field],
            reason=f"the probe's own two markets should agree on {field}",
        )

    engine_a = COSHestonEngine(_heston_model(market_a["inputs"]))
    engine_b = COSHestonEngine(_heston_model(market_b["inputs"]))
    for field, want in market_b["expected"].items():
        tolerance.exact(_cumulants(engine_a, 1.0)[field], _cumulants(engine_b, 1.0)[field])
        tolerance.tight(_cumulants(engine_b, 1.0)[field], want, reason=field)


@pytest.mark.parametrize("case_name", CHF_CASES)
def test_chf_matches_cpp(case_name: str) -> None:
    """The normalised (driftless) characteristic function at real ``u``.

    # C++ parity: ``COSHestonEngine::chF(Real u, Real t)`` (.cpp:126-142) —
    # note the argument is real, unlike ``AnalyticHestonEngine::chF``.
    """
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    value = COSHestonEngine(_heston_model(inputs)).ch_f(inputs["u"], inputs["t"])
    tolerance.tight(value.real, expected["chf_real"], reason=f"{case_name}.real")
    tolerance.tight(value.imag, expected["chf_imag"], reason=f"{case_name}.imag")


@pytest.mark.parametrize("case_name", NPV_CASES)
def test_npv_matches_cpp(case_name: str) -> None:
    """Price across both markets, three parameter sets, four maturities, five
    strikes, both option types and five ``(L, N)`` settings."""
    case = CPP[case_name]
    tolerance.custom(
        _priced_option(case["inputs"]).npv(),
        case["expected"]["npv"],
        # See "Tolerance (a)": TIGHT relative, absolute floor at the
        # cancellation limit of `spot*qf - K*df*(1 - s)`.
        abs_tol=_NPV_ABS_FLOOR,
        rel_tol=1e-12,
        reason=case_name,
    )


def _distinct_price_count(prices: list[float]) -> int:
    """Number of prices that differ by more than 1e-13 relative.

    The knob sweep and floating-point noise are cleanly separated: the smallest
    knob-induced separation anywhere in the reference is 2.13e-10 relative
    (``(12,75)`` versus ``(16,200)``), while two settings that C++ prices
    *identically* differ here by at most 5e-16 relative (one ULP, from
    summation order in the cosine series). 1e-13 sits three orders of magnitude
    from each, so this counts genuinely different prices and not ULP ties.
    """
    kept: list[float] = []
    for price in sorted(prices):
        if not any(math.isclose(price, seen, rel_tol=1e-13, abs_tol=0.0) for seen in kept):
            kept.append(price)
    return len(kept)


def test_truncation_width_and_term_count_are_honoured() -> None:
    """``L`` and ``N`` must change the answer.

    The reference sweeps ``(16,200) (12,75) (20,400) (25,600) (16,25)`` at
    1y/market A. ``N = 25`` is far too few cosine terms to converge, so C++
    prices it visibly differently — by at least 7.0e-4 relative in every one of
    the 30 swept groups. An engine that accepted the knobs and priced at the
    defaults would return one number five times.
    """
    groups: dict[tuple[str, str, str, str, float], dict[tuple[float, int], str]] = {}
    for name in NPV_CASES:
        inputs = CPP[name]["inputs"]
        key = (
            inputs["market"],
            inputs["params"],
            inputs["maturity"],
            inputs["type"],
            inputs["strike"],
        )
        groups.setdefault(key, {})[(inputs["L"], int(inputs["N"]))] = name

    swept = {key: names for key, names in groups.items() if len(names) > 1}
    assert len(swept) == 30
    assert {frozenset(names) for names in swept.values()} == {
        frozenset({(16.0, 200), (12.0, 75), (20.0, 400), (25.0, 600), (16.0, 25)})
    }

    for key, names in swept.items():
        port = {knob: _priced_option(CPP[name]["inputs"]).npv() for knob, name in names.items()}
        cpp = {knob: CPP[name]["expected"]["npv"] for knob, name in names.items()}

        assert _distinct_price_count(list(port.values())) == _distinct_price_count(
            list(cpp.values())
        ), f"{key}: the (L, N) sweep does not move the port's price the way it moves C++'s"

        # The sharpest single statement: 25 cosine terms cannot reproduce 200.
        coarse, default = port[(16.0, 25)], port[(16.0, 200)]
        separation = abs(coarse - default) / max(abs(coarse), abs(default))
        assert separation > 1e-4, f"{key}: N is being discarded (separation {separation:.3e})"


@pytest.mark.parametrize("case_name", BOUND_CASES)
def test_truncation_bound_early_return(case_name: str) -> None:
    """Outside ``[a/2, b/2]`` the series is abandoned for the no-arbitrage bound.

    # C++ parity: coshestonengine.cpp:87-98. The returned number is the
    # discounted intrinsic — ``max(spot*qf - K*df, 0)`` for a call,
    # ``max(K*df - spot*qf, 0)`` for a put — and is NOT a price. Asserting only
    # that the answer is near intrinsic would not distinguish the branch from a
    # deep-wing series that happened to converge to it, so the branch predicate
    # itself is recomputed from the engine's own public cumulants.
    """
    case = CPP[case_name]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["hit_truncation_branch"] is True

    process = _heston_process(inputs)
    maturity_date = _maturity_date(inputs)
    maturity = process.time(maturity_date)
    engine = COSHestonEngine(_heston_model(inputs), inputs["L"], int(inputs["N"]))

    discount = process.risk_free_rate().discount(maturity_date)
    dividend_discount = process.dividend_yield().discount(maturity_date)
    spot = process.s0().value()
    strike = inputs["strike"]
    x = math.log(spot * dividend_discount / discount / strike)
    width = inputs["L"] * math.sqrt(abs(engine.c2(maturity)))
    lower = x + engine.c1(maturity) - width
    upper = x + engine.c1(maturity) + width
    assert x >= upper / 2 or x <= lower / 2, "the truncation branch did not fire"

    if _option_type(inputs["type"]) == OptionType.Call:
        bound = max(spot * dividend_discount - strike * discount, 0.0)
    else:
        bound = max(strike * discount - spot * dividend_discount, 0.0)
    tolerance.tight(bound, expected["no_arbitrage_bound"], reason="recomputed bound")
    tolerance.tight(
        expected["npv"],
        expected["no_arbitrage_bound"],
        reason="the probe's own value and bound must coincide",
    )
    tolerance.tight(_priced_option(inputs).npv(), expected["npv"], reason=case_name)


def test_no_greeks_are_provided() -> None:
    """C++ writes ``results_.value`` and nothing else.

    # C++ parity: coshestonengine.cpp:52-123 never touches the Greeks block,
    # so ``option.delta()`` throws "delta not provided". A port that filled
    # them would be inventing results C++ does not produce.
    """
    case = CPP["cos_no_greeks"]
    inputs, expected = case["inputs"], case["expected"]
    assert expected["has_any_greek"] is False

    option = VanillaOption(
        PlainVanillaPayoff(_option_type(inputs["type"]), inputs["strike"]),
        EuropeanExercise(_maturity_date(inputs)),
    )
    option.set_pricing_engine(COSHestonEngine(_heston_model(inputs)))
    assert option.npv() > 0.0
    assert _has_any_greek(option) is False
    for accessor in GREEK_ACCESSORS:
        with pytest.raises(LibraryException):
            getattr(option, accessor)()


def test_rejects_non_plain_vanilla_payoff() -> None:
    """# C++ parity: ``QL_REQUIRE(payoff, "non plain vanilla payoff given")``."""
    case = CPP["cos_rejects_non_plain_vanilla_payoff"]
    assert case["inputs"]["payoff"] == "CashOrNothingPayoff"
    assert case["expected"]["throws"] is True

    option = VanillaOption(
        CashOrNothingPayoff(OptionType.Call, 100.0, 1.0),
        EuropeanExercise(TODAY + 365),
    )
    option.set_pricing_engine(COSHestonEngine(_heston_model(CPP["cos_no_greeks"]["inputs"])))
    with pytest.raises(LibraryException, match="non plain vanilla payoff given"):
        option.npv()


def test_rejects_non_european_exercise() -> None:
    """# C++ parity: ``QL_REQUIRE(exercise->type() == Exercise::European)``."""
    case = CPP["cos_rejects_non_european_exercise"]
    assert case["inputs"]["exercise"] == "AmericanExercise"
    assert case["expected"]["throws"] is True

    option = VanillaOption(
        PlainVanillaPayoff(OptionType.Call, 100.0),
        AmericanExercise(TODAY, TODAY + 365),
    )
    option.set_pricing_engine(COSHestonEngine(_heston_model(CPP["cos_no_greeks"]["inputs"])))
    with pytest.raises(LibraryException, match="not an European option"):
        option.npv()
