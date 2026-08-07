"""BasketGeneratingEngine + MatchHelper vs C++ v1.43.

Reached through ``Gaussian1dNonstandardSwaptionEngine`` — the only public
route, since ``calibrationBasket`` is a member of the protected base and
``MatchHelper`` is a private nested class in C++.

Reference: ``migration-harness/references/v143/pe/swaption.json``, produced by
``migration-harness/cpp/probes/v143_pe_swaption/probe.cpp``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.thirty_360 import Convention, Thirty360
from pquantlib.exercise import EuropeanExercise
from pquantlib.indexes.swap_index import SwapIndex
from pquantlib.instruments.nonstandard_swap import NonstandardSwap
from pquantlib.instruments.nonstandard_swaption import NonstandardSwaption
from pquantlib.instruments.swap import SwapType
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.swaption.basket_generating_engine import (
    CalibrationBasketType,
    MatchHelper,
)
from pquantlib.pricingengines.swaption.gaussian1d_nonstandard_swaption_engine import (
    Gaussian1dNonstandardSwaptionEngine,
)
from pquantlib.termstructures.volatility.swaption.swaption_constant_vol import (
    ConstantSwaptionVolatility,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

from . import _v143_swaption_market as mkt

_BASKET_TYPES = {
    "Naive": CalibrationBasketType.Naive,
    "MaturityStrikeByDeltaGamma": CalibrationBasketType.MaturityStrikeByDeltaGamma,
}


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/pe/swaption")


@pytest.fixture(autouse=True)
def _eval_date() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    settings = ObservableSettings()
    saved = settings.evaluation_date
    # probe.cpp:293 — Settings::instance().evaluationDate() = Date(3, March, 2025)
    settings.evaluation_date = mkt.TODAY
    yield
    settings.evaluation_date = saved


def _build() -> tuple[NonstandardSwaption, Gaussian1dNonstandardSwaptionEngine]:
    """The probe's standard 5y-into-5y payer swaption, as a NonstandardSwaption."""
    curve = mkt.forwarding_curve()
    fixed_sched = mkt.fixed_schedule()
    float_sched = mkt.float_schedule()
    n_fix = len(fixed_sched) - 1
    n_flt = len(float_sched) - 1
    swap = NonstandardSwap(
        type_=SwapType.Payer,
        fixed_nominal=[1.0] * n_fix,
        floating_nominal=[1.0] * n_flt,
        fixed_schedule=fixed_sched,
        fixed_rate=[0.030] * n_fix,
        fixed_day_count=Thirty360(Convention.BondBasis),
        floating_schedule=float_sched,
        ibor_index=mkt.euribor6m(curve),
        gearing=1.0,
        spread=0.0,
        floating_day_count=Actual360(),
    )
    nonstd = NonstandardSwaption(swap, EuropeanExercise(mkt.exercise_date()))
    engine = Gaussian1dNonstandardSwaptionEngine(mkt.make_gsr(curve))
    nonstd.set_pricing_engine(engine)
    return nonstd, engine


def _swap_index() -> SwapIndex:
    # probe.cpp:1393-1396 — SwapIndex("EurSwap5Y", 5Y, 2, EUR, TARGET, 1Y,
    # ModifiedFollowing, Thirty360(BondBasis), Euribor6M).
    return SwapIndex(
        "EurSwap5Y",
        Period(5, TimeUnit.Years),
        2,
        EURCurrency(),
        mkt.calendar(),
        Period(1, TimeUnit.Years),
        BusinessDayConvention.ModifiedFollowing,
        Thirty360(Convention.BondBasis),
        mkt.euribor6m(mkt.forwarding_curve()),
    )


def _vol_structure() -> ConstantSwaptionVolatility:
    # probe.cpp:1397-1398.
    return ConstantSwaptionVolatility(
        settlement_days=0,
        calendar=mkt.calendar(),
        business_day_convention=BusinessDayConvention.Following,
        volatility=0.20,
        day_counter=Actual365Fixed(),
        volatility_type=VolatilityType.ShiftedLognormal,
        shift=0.0,
    )


def test_naive_basket(cpp: dict[str, Any]) -> None:
    """Naive: one ATM helper per alive exercise date, out to underlyingLastDate."""
    expected = cpp["basket_naive"]["expected"]
    nonstd, engine = _build()
    tolerance.loose(nonstd.npv(), expected["exotic_npv"])

    basket = engine.calibration_basket(
        nonstd.exercise(), _swap_index(), _vol_structure(), CalibrationBasketType.Naive
    )
    assert len(basket) == expected["basket_size"]
    for i, helper in enumerate(basket):
        swap = helper.underlying_swap()
        assert str(helper.swaption().exercise().date(0)) == expected["expiries"][i]
        assert str(swap.maturity_date()) == expected["maturities"][i]
        # The ATM strike comes straight out of the helper's own fair-rate
        # computation, so it is bit-reproducible.
        tolerance.tight(swap.fixed_rate(), expected["strikes"][i])
        tolerance.tight(swap.nominal(), expected["nominals"][i])
        tolerance.tight(helper.volatility.value(), expected["volatilities"][i])


def test_delta_gamma_basket(cpp: dict[str, Any]) -> None:
    """MaturityStrikeByDeltaGamma: a Levenberg-Marquardt fit through MatchHelper."""
    expected = cpp["basket_delta_gamma"]["expected"]
    nonstd, engine = _build()
    tolerance.loose(nonstd.npv(), expected["exotic_npv"])

    basket = engine.calibration_basket(
        nonstd.exercise(),
        _swap_index(),
        _vol_structure(),
        CalibrationBasketType.MaturityStrikeByDeltaGamma,
    )
    assert len(basket) == expected["basket_size"]
    for i, helper in enumerate(basket):
        swap = helper.underlying_swap()
        assert str(helper.swaption().exercise().date(0)) == expected["expiries"][i]
        assert str(swap.maturity_date()) == expected["maturities"][i]
        # The calibrated (nominal, rate) pair is the output of a
        # Levenberg-Marquardt search, so it is only reproducible to the
        # optimiser's own stopping tolerance, not to machine precision.
        #
        # Derivation of the bound: EndCriteria uses xtol = gtol = ftol = 1e-8
        # (basketgeneratingengine.cpp:173) and the residuals themselves are
        # built from a central finite difference in the model state with
        # h = 1e-4 (cpp:111), so the residual carries ~eps/h^2 = 2e-8 of
        # noise on the gamma component. Along the (nominal, rate) trade-off
        # direction the objective is nearly flat — the C++ optimum and the
        # Python optimum have residual norms 2.4e-7 and 2.4e-7 respectively
        # — so the position along that direction is determined only to
        # ~sqrt(noise / curvature). Empirically the two MINPACK
        # implementations (QuantLib's own lmdif vs SciPy's _lmder) land
        # 1.7e-7 apart in nominal and 2.8e-9 apart in rate. Bound the rate
        # at 1e-7 absolute and the nominal at 1e-5 relative: both are two
        # decades tighter than the divergence that the (now fixed)
        # diff_step/epsfcn mix-up used to produce (5.5e-7 / 2.8e-4), so
        # this would still catch a regression there.
        tolerance.custom(
            swap.fixed_rate(),
            expected["strikes"][i],
            abs_tol=1e-7,
            rel_tol=0.0,
            reason=(
                "Levenberg-Marquardt optimum along a near-flat "
                "(nominal, rate) direction; EndCriteria tolerances are 1e-8 "
                "and the residual itself is a finite difference with h=1e-4"
            ),
        )
        tolerance.custom(
            swap.nominal(),
            expected["nominals"][i],
            abs_tol=0.0,
            rel_tol=1e-5,
            reason="same LM flat-direction argument as the strike above",
        )
        tolerance.tight(helper.volatility.value(), expected["volatilities"][i])


def test_naive_and_delta_gamma_differ(cpp: dict[str, Any]) -> None:
    """The two basket types must not produce the same helper."""
    naive = cpp["basket_naive"]["expected"]
    dg = cpp["basket_delta_gamma"]["expected"]
    assert naive["strikes"] != dg["strikes"]
    assert naive["nominals"] != dg["nominals"]


def test_match_helper_residuals_vanish_at_the_solution(cpp: dict[str, Any]) -> None:
    """MatchHelper is a real cost function: exercise it directly.

    Feeding it the exotic's own (npv, delta, gamma) and a parameter vector
    equal to the exotic's own shape must give residuals close to zero,
    because the exotic here IS a standard 5y swap.
    """
    nonstd, engine = _build()
    nonstd.npv()  # force the arguments to be populated
    expiry = mkt.exercise_date()
    h = 0.0001
    npv = engine.underlying_npv(expiry, 0.0)
    npvm = engine.underlying_npv(expiry, -h)
    npvp = engine.underlying_npv(expiry, h)
    delta = (npvp - npvm) / (2.0 * h)
    gamma = (npvp - 2.0 * npv + npvm) / (h * h)

    helper = MatchHelper(
        engine.underlying_type(),
        npv,
        delta,
        gamma,
        mkt.make_gsr(mkt.forwarding_curve()),
        _swap_index(),
        expiry,
        50.0,
        h,
    )
    guess = engine.initial_guess(expiry)
    assert guess.size == 3
    residuals = helper.values(guess)
    assert residuals.size == 3
    # The initial guess is (nominal 1.0, ~5y, rate 3%) which is exactly the
    # underlying, so the residual should already be small; the LM fit only
    # polishes it. 1e-2 is a structural assertion (right order of magnitude,
    # right sign convention), not a cross-validated number.
    assert float(np.max(np.abs(residuals))) < 1e-2
    # value() is the RMS of values() — C++ CostFunction's default.
    tolerance.tight(
        helper.value(guess),
        float(np.sqrt(float(np.sum(residuals * residuals)) / residuals.size)),
    )


def test_underlying_hooks(cpp: dict[str, Any]) -> None:
    """The four BasketGeneratingEngine hooks the C++ base declares pure-virtual."""
    expected = cpp["basket_naive"]["expected"]
    nonstd, engine = _build()
    nonstd.npv()
    assert engine.underlying_type() == SwapType.Payer
    assert str(engine.underlying_last_date()) == expected["maturities"][0]
    # underlying_npv at y = 0, discounted back to t = 0 with the model
    # numeraire, is the European swaption's intrinsic-at-the-mode; it must
    # at minimum be finite and of the right scale.
    npv0 = engine.underlying_npv(mkt.exercise_date(), 0.0)
    assert np.isfinite(npv0)
    guess = engine.initial_guess(mkt.exercise_date())
    tolerance.tight(float(guess[0]), 1.0)  # constant nominal
    tolerance.tight(float(guess[2]), 0.03)  # constant fixed rate
