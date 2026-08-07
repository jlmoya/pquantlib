"""Cross-validate IntegralNtdEngine against C++ QuantLib v1.43.

Probe source: migration-harness/cpp/probes/v143_experimental_cdsntd/probe.cpp
              (sections B and C)
Reference:    migration-harness/references/v143/experimental/cdsntd.json

C++ class: ``IntegralNtdEngine`` ql/experimental/credit/integralntdengine.hpp:31

Why two sections
----------------
Section B installs an *analytic* loss model —
``probAtLeastNEvents(n, d) = 1 - exp(-n * lambda * t)`` with ``t`` the
Actual365Fixed year fraction from the basket reference date — on a real
``Basket``. In C++ it is a probe-local ``DefaultLossModel`` subclass (the
hooks are protected virtuals and ``Basket`` is a friend); here it is the same
closed form on a small stub. That makes every reported number a function of
the engine's integration grid and nothing else, which is the point: the grid
is the part of this engine a port gets wrong.

The grid is NOT clamped to the accrual end. C++ (integralntdengine.cpp:78-152)
steps AFTER the loop body and shrinks the step to one day exactly once, the
first time ``d0 + step`` would overshoot; the walk then lands on
``accrualEndDate`` exactly. Writing ``d = min(d0 + step, accrual_end)``
instead agrees whenever the step divides the accrual period — which is why
the port shipped that way and every existing test passed — and collapses the
entire tail into one lump when the step EXCEEDS it. Cases ``ntd_m6`` and
``ntd_y1`` are chosen so the step exceeds the accrual period and the two
grids cannot agree.

Section C runs the same engine over a real
``ConstantLossModel`` + Gaussian copula, so "it prices" means it prices
something a user would actually build, not only a stub.

Evaluation date
---------------
The probe sets ``Settings::instance().evaluationDate()`` at the top of each
section (probe.cpp ``sectionNtdAnalytic`` / ``sectionNtdCopula``). The
fixtures pin the same dates from the reference and restore in teardown.

Tolerance
---------
Section B: LOOSE. The engine sums hundreds of discount-factor times
probability-increment products; the products themselves are exponentials, so
the accumulated rounding is well above TIGHT but nowhere near 1e-8 relative.
Section C: LOOSE, and additionally bounded by the Gauss-Hermite quadrature
inside the copula, which both sides run at the same order.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.basket import Basket
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossModel,
)
from pquantlib.experimental.credit.default_probability_key import (
    DefaultProbKey,
    NorthAmericaCorpDefaultKey,
)
from pquantlib.experimental.credit.default_probability_latent_model import (
    LatentModelIntegrationType,
)
from pquantlib.experimental.credit.default_type import Seniority
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.experimental.credit.nth_to_default import (
    NthToDefault,
    NthToDefaultResults,
)
from pquantlib.experimental.credit.pool import Pool
from pquantlib.experimental.math.gaussian_copula_policy import GaussianCopulaPolicy
from pquantlib.instruments.credit_default_swap import ProtectionSide
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.pricingengines.credit.integral_ntd_engine import IntegralNtdEngine
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

_ANALYTIC_CASES = [
    "ntd_w1",
    "ntd_m1",
    "ntd_m3",
    "ntd_m6",
    "ntd_y1",
    "ntd_buyer",
    "ntd_noaccr",
    "ntd_upfront",
    "ntd_first",
    "ntd_last",
]


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/cdsntd")


class _AnalyticNtdLossModel(Observable):
    """Mirror of the probe's ``AnalyticNtdLossModel``.

    probe.cpp, ``class AnalyticNtdLossModel : public DefaultLossModel``.
    Only the two hooks ``Basket`` forwards to are implemented; the rest of
    the C++ ``DefaultLossModel`` surface is ``QL_FAIL`` there and absent here.
    """

    def __init__(self, t_origin: Date, lambda_: float, recovery: float) -> None:
        super().__init__()
        self._t_origin = t_origin
        self._lambda = lambda_
        self._recovery = recovery
        self._dc = Actual365Fixed()

    def set_basket(self, basket: object) -> None:
        del basket  # C++ resetModel() is a no-op: nothing is cached.

    def expected_tranche_loss(self, d: Date) -> float:
        del d
        raise LibraryException("expectedTrancheLoss Not implemented for this model.")

    def prob_at_least_n_events(self, n: int, d: Date) -> float:
        t = self._dc.year_fraction(self._t_origin, d)
        if t <= 0.0:
            return 0.0
        return 1.0 - math.exp(-float(n) * self._lambda * t)

    def expected_recovery(self, d: Date, i_name: int, key: DefaultProbKey) -> float:
        del d, i_name, key
        return self._recovery


def _default_key() -> DefaultProbKey:
    """Probe: ``NorthAmericaCorpDefaultKey(EURCurrency(), SeniorSec, Period(), 1.0)``.

    ``SeniorSec`` is the Markit synonym C++ defines as ``SecDom``
    (defaulttype.hpp:38 + :46); it is spelled ``SecDom`` here because the
    Python aliases are attached dynamically.
    """
    return NorthAmericaCorpDefaultKey(
        EURCurrency(), Seniority.SecDom, Period(0, TimeUnit.Days), 1.0
    )


def _make_basket(
    ref_date: Date, names: int, notional_per_name: float, hazard: float
) -> Basket:
    key = _default_key()
    curve = FlatHazardRate(ref_date, SimpleQuote(hazard), Actual365Fixed())
    issuer = Issuer(probabilities=[(key, curve)])
    pool = Pool()
    ids = [f"Name{i}" for i in range(names)]
    for nm in ids:
        pool.add(nm, issuer, key)
    return Basket(
        ref_date, ids, [notional_per_name] * names, pool, 0.0, 1.0
    )


# ---------------------------------------------------------------------------
# Section B — analytic loss model, engine grid isolated
# ---------------------------------------------------------------------------


@pytest.fixture
def _analytic_eval_date(cpp_ref: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Probe: ``sectionNtdAnalytic`` sets the evaluation date to B_TODAY."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Date(int(cpp_ref["b_evaluationDate_serial"]))
    yield
    settings.evaluation_date = previous


def _build_analytic(cpp_ref: dict[str, Any], tag: str) -> NthToDefault:
    b_today = Date(int(cpp_ref["b_evaluationDate_serial"]))
    basket_ref = Date(int(cpp_ref[f"{tag}_basketRefDate_serial"]))
    names = int(cpp_ref[f"{tag}_names"])
    notional = cpp_ref[f"{tag}_notional"]

    basket = _make_basket(
        basket_ref, names, notional / names, cpp_ref["b_lambda"]
    )
    basket.set_loss_model(
        _AnalyticNtdLossModel(b_today, cpp_ref["b_lambda"], cpp_ref["b_recovery"])
    )

    sched = (
        MakeSchedule()
        .from_date(Date(int(cpp_ref[f"{tag}_schedStart_serial"])))
        .to(Date(int(cpp_ref[f"{tag}_schedEnd_serial"])))
        .with_tenor(
            Period(
                int(cpp_ref[f"{tag}_tenor_length"]),
                TimeUnit(int(cpp_ref[f"{tag}_tenor_units"])),
            )
        )
        .with_calendar(TARGET())
        .build()
    )
    ntd = NthToDefault(
        basket=basket,
        n=int(cpp_ref[f"{tag}_ntdOrder"]),
        side=ProtectionSide(int(cpp_ref[f"{tag}_side"])),
        premium_schedule=sched,
        upfront_rate=cpp_ref[f"{tag}_upfrontRate"],
        premium_rate=cpp_ref[f"{tag}_premiumRate"],
        day_counter=Actual360(),
        nominal=notional,
        settle_premium_accrual=bool(cpp_ref[f"{tag}_settleAccrual"]),
    )
    yts = FlatForward(
        b_today,
        SimpleQuote(0.035),
        Actual365Fixed(),
        Compounding.Continuous,
        Frequency.Annual,
    )
    step = Period(
        int(cpp_ref[f"{tag}_step_length"]),
        TimeUnit(int(cpp_ref[f"{tag}_step_units"])),
    )
    ntd.set_pricing_engine(IntegralNtdEngine(step, yts))
    return ntd


@pytest.mark.usefixtures("_analytic_eval_date")
@pytest.mark.parametrize("tag", _ANALYTIC_CASES)
def test_analytic_model_probabilities_match_cpp(
    cpp_ref: dict[str, Any], tag: str
) -> None:
    """The stub reproduces the probe's closed form before anything else.

    If these disagree, every NPV below is meaningless.
    """
    model = _AnalyticNtdLossModel(
        Date(int(cpp_ref["b_evaluationDate_serial"])),
        cpp_ref["b_lambda"],
        cpp_ref["b_recovery"],
    )
    sched = (
        MakeSchedule()
        .from_date(Date(int(cpp_ref[f"{tag}_schedStart_serial"])))
        .to(Date(int(cpp_ref[f"{tag}_schedEnd_serial"])))
        .with_tenor(
            Period(
                int(cpp_ref[f"{tag}_tenor_length"]),
                TimeUnit(int(cpp_ref[f"{tag}_tenor_units"])),
            )
        )
        .with_calendar(TARGET())
        .build()
    )
    order = int(cpp_ref[f"{tag}_ntdOrder"])
    expected = cpp_ref[f"{tag}_model_probs"]
    assert len(sched.dates) == len(expected)
    for d, want in zip(sched.dates, expected, strict=True):
        tolerance.tight(model.prob_at_least_n_events(order, d), want)


@pytest.mark.usefixtures("_analytic_eval_date")
@pytest.mark.parametrize("tag", _ANALYTIC_CASES)
def test_analytic_npv_matches_cpp(cpp_ref: dict[str, Any], tag: str) -> None:
    """``IntegralNtdEngine::calculate`` (integralntdengine.cpp:28-184).

    ``ntd_m6`` and ``ntd_y1`` use an integration step LARGER than the accrual
    period; they are the cases a clamped grid cannot reproduce.
    """
    assert cpp_ref[f"{tag}_error"] == "", "probe recorded a throw for this case"
    ntd = _build_analytic(cpp_ref, tag)
    tolerance.loose(ntd.npv(), cpp_ref[f"{tag}_NPV"])
    tolerance.loose(ntd.premium_leg_npv(), cpp_ref[f"{tag}_premiumLegNPV"])
    tolerance.loose(ntd.protection_leg_npv(), cpp_ref[f"{tag}_protectionLegNPV"])
    tolerance.loose(ntd.fair_premium(), cpp_ref[f"{tag}_fairPremium"])


@pytest.mark.usefixtures("_analytic_eval_date")
def test_coarse_step_differs_from_a_clamped_grid(cpp_ref: dict[str, Any]) -> None:
    """The regression guard for the grid, stated as a property.

    ``ntd_m3`` steps exactly one accrual period, so C++'s grid and a grid
    clamped to ``accrual_end`` coincide. ``ntd_m6`` steps two accrual periods,
    where they cannot: C++ walks daily to the accrual end while a clamped
    grid takes a single lump. Requiring the two C++ NPVs to differ by more
    than the LOOSE tier proves the reference actually discriminates, so a
    future reintroduction of the clamp cannot pass
    ``test_analytic_npv_matches_cpp`` by accident.
    """
    npv_m3 = cpp_ref["ntd_m3_NPV"]
    npv_m6 = cpp_ref["ntd_m6_NPV"]
    assert abs(npv_m6 - npv_m3) / abs(npv_m3) > 1e-4, (
        "the coarse-step case no longer discriminates the integration grid"
    )


@pytest.mark.usefixtures("_analytic_eval_date")
def test_first_accrual_before_curve_reference_raises(cpp_ref: dict[str, Any]) -> None:
    """C++ THROWS here, and the port must too.

    ``ntd_started``'s schedule starts before the discount curve's reference
    date, so the upfront branch (integralntdengine.cpp:157-163) discounts the
    first coupon's accrual start — a date the curve cannot reach — and
    QuantLib raises "negative time". A port that silently extrapolates the
    curve instead would return a plausible number for an ill-posed trade.
    """
    assert cpp_ref["ntd_started_error"].startswith("negative time")
    with pytest.raises(LibraryException, match="negative time"):
        _build_analytic(cpp_ref, "ntd_started").npv()


@pytest.mark.usefixtures("_analytic_eval_date")
def test_buyer_and_seller_are_exact_sign_flips(cpp_ref: dict[str, Any]) -> None:
    """``Protection::Buyer`` negates premium, accrual, claim and upfront.

    integralntdengine.cpp:164-169. ``ntd_m1`` and ``ntd_buyer`` differ only in
    the side, so their NPVs must be exact negatives of each other.
    """
    seller = _build_analytic(cpp_ref, "ntd_m1")
    buyer = _build_analytic(cpp_ref, "ntd_buyer")
    tolerance.loose(buyer.npv(), -seller.npv())
    tolerance.loose(cpp_ref["ntd_buyer_NPV"], -cpp_ref["ntd_m1_NPV"])


@pytest.mark.usefixtures("_analytic_eval_date")
def test_additional_results_use_the_cpp_keys(cpp_ref: dict[str, Any]) -> None:
    """C++ publishes camelCase keys (integralntdengine.cpp:179-183)."""
    ntd = _build_analytic(cpp_ref, "ntd_m1")
    ntd.npv()
    engine = ntd.pricing_engine()
    assert engine is not None
    results = cast("NthToDefaultResults", engine.get_results())
    extra: dict[str, Any] = results.additional_results
    assert set(extra) == {"fairPremium", "premiumLegNPV", "protectionLegNPV"}
    tolerance.loose(extra["fairPremium"], cpp_ref["ntd_m1_fairPremium"])
    tolerance.loose(extra["premiumLegNPV"], cpp_ref["ntd_m1_premiumLegNPV"])
    tolerance.loose(extra["protectionLegNPV"], cpp_ref["ntd_m1_protectionLegNPV"])


# ---------------------------------------------------------------------------
# Section C — the real copula stack
# ---------------------------------------------------------------------------


@pytest.fixture
def _copula_eval_date(cpp_ref: dict[str, Any]) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Probe: ``sectionNtdCopula`` sets the evaluation date."""
    settings = ObservableSettings()
    previous = settings.evaluation_date
    settings.evaluation_date = Date(int(cpp_ref["c_evaluationDate_serial"]))
    yield
    settings.evaluation_date = previous


def _build_copula_basket(cpp_ref: dict[str, Any]) -> Basket:
    today = Date(int(cpp_ref["c_evaluationDate_serial"]))
    names = int(cpp_ref["c_names"])
    basket = _make_basket(
        today, names, cpp_ref["c_notional"] / names, cpp_ref["c_hazard"]
    )
    loading = cpp_ref["c_factor_loading"]
    weights: Sequence[Sequence[float]] = [[loading] for _ in range(names)]
    basket.set_loss_model(
        ConstantLossModel(
            weights,
            [cpp_ref["c_recovery"]] * names,
            GaussianCopulaPolicy(weights),
            LatentModelIntegrationType.GaussianQuadrature,
        )
    )
    return basket


@pytest.mark.usefixtures("_copula_eval_date")
def test_copula_basket_probabilities_match_cpp(cpp_ref: dict[str, Any]) -> None:
    """``Basket::probAtLeastNEvents`` / ``Basket::recoveryRate``.

    basket.cpp:365-375 — the two forwards added so a real Basket satisfies
    the engine's protocol. Pinned before the NPVs that consume them.
    """
    basket = _build_copula_basket(cpp_ref)
    names = int(cpp_ref["c_names"])
    for k in range(4):
        d = Date(int(cpp_ref[f"c_probeDate_{k}_serial"]))
        for n in range(1, names + 1):
            tolerance.loose(
                basket.prob_at_least_n_events(n, d),
                cpp_ref[f"c_probAtLeast_{n}_{k}"],
            )
    tolerance.loose(
        basket.recovery_rate(Date(int(cpp_ref["c_probeDate_0_serial"])), 0),
        cpp_ref["c_recoveryRate_0"],
    )


@pytest.mark.usefixtures("_copula_eval_date")
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_copula_ntd_prices_match_cpp(cpp_ref: dict[str, Any], order: int) -> None:
    """The engine over a real ConstantLossModel + Gaussian copula.

    Section B proves the integration grid; this proves the grid composes with
    the loss-model stack an actual user would build.
    """
    today = Date(int(cpp_ref["c_evaluationDate_serial"]))
    basket = _build_copula_basket(cpp_ref)
    sched = (
        MakeSchedule()
        .from_date(Date.from_ymd(20, Month.March, 2024))
        .to(Date.from_ymd(20, Month.March, 2025))
        .with_tenor(Period(3, TimeUnit.Months))
        .with_calendar(TARGET())
        .build()
    )
    ntd = NthToDefault(
        basket=basket,
        n=order,
        side=ProtectionSide.Seller,
        premium_schedule=sched,
        upfront_rate=0.0,
        premium_rate=0.02,
        day_counter=Actual360(),
        nominal=cpp_ref["c_notional"],
        settle_premium_accrual=True,
    )
    yts = FlatForward(
        today,
        SimpleQuote(0.035),
        Actual365Fixed(),
        Compounding.Continuous,
        Frequency.Annual,
    )
    ntd.set_pricing_engine(IntegralNtdEngine(Period(1, TimeUnit.Months), yts))

    tag = f"c_ntd{order}"
    tolerance.loose(ntd.npv(), cpp_ref[f"{tag}_NPV"])
    tolerance.loose(ntd.premium_leg_npv(), cpp_ref[f"{tag}_premiumLegNPV"])
    tolerance.loose(ntd.protection_leg_npv(), cpp_ref[f"{tag}_protectionLegNPV"])
    tolerance.loose(ntd.fair_premium(), cpp_ref[f"{tag}_fairPremium"])
