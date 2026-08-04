"""Python equivalents of the two short-rate tests C++ QuantLib added in v1.43.

C++ source: ``test-suite/shortratemodels.cpp`` @ v1.43 —
``testHullWhiteUpdatesR0WhenTermStructureRelinks`` and
``testVasicekDiscountFactorForSmallMeanReversion``.

Neither pins *new* v1.43 behaviour: the Hull-White one pins behaviour C++ has
had since the class existed, and the Vasicek one pins the analytic limit that
v1.43 substituted for a hard-coded ``0.0``. Both caught a real divergence in
this port, which is why they are worth having rather than assuming.

No probe reference is needed: the Hull-White expectation is read back off the
curve under test, and the Vasicek expectation is the closed-form limit written
out in ``vasicek.cpp`` itself.
"""

from __future__ import annotations

import math

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.models.shortrate.onefactor.vasicek import Vasicek
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month


def test_hull_white_updates_r0_when_the_curve_moves() -> None:
    """``r0`` must follow the term structure the model observes.

    C++ ``HullWhite::generateArguments`` refreshes ``r0_`` from the curve; this
    port only set it in the constructor, so every quantity derived from ``r0``
    silently kept using the rate the curve had when the model was built.

    C++ moves the curve by relinking a ``RelinkableHandle``. This port holds
    the term structure directly (no ``Handle`` indirection — see
    ``TermStructureConsistentModel``), so the equivalent move is to change the
    quote the curve is built on: either way the model is notified through the
    observer chain and ``generate_arguments`` runs.
    """
    today = Date.from_ymd(19, Month.May, 2026)
    forward = SimpleQuote(0.02)
    term_structure = FlatForward(today, forward, Actual365Fixed())

    def curve_short_rate() -> float:
        return term_structure.forward_rate(
            0.0, 0.0, Compounding.Continuous, Frequency.NoFrequency
        ).rate()

    # C++ compares ``r0`` (built by generateArguments from ``zeroRate(0.0, ...)``)
    # against ``forwardRate(0.0, 0.0, ...)`` at abs tolerance 1e-12 — the two are
    # different curve queries, and both take the ``t == 0 -> t = 1e-4`` shortcut,
    # so an exp/log round trip over dt = 1e-4 amplifies a 1-ulp discount-factor
    # error (~2.2e-16) to ~2.2e-12 in the rate. That is why the upstream test
    # uses an absolute rather than a relative bound; we reuse its number.
    r0_tolerance = 1.0e-12

    model = HullWhite(term_structure)
    custom(
        model.r0(),
        curve_short_rate(),
        abs_tol=r0_tolerance,
        rel_tol=0.0,
        reason="C++ testHullWhiteUpdatesR0WhenTermStructureRelinks tolerance",
    )

    forward.set_value(0.05)

    custom(
        model.r0(),
        curve_short_rate(),
        abs_tol=r0_tolerance,
        rel_tol=0.0,
        reason="C++ testHullWhiteUpdatesR0WhenTermStructureRelinks tolerance",
    )
    # And it really moved — the pre-move value would have been ~0.02.
    assert model.r0() > 0.04


def test_vasicek_discount_factor_for_small_mean_reversion() -> None:
    """The ``a -> 0`` branch must return the analytic limit, not zero.

    C++ parity: ``testVasicekDiscountFactorForSmallMeanReversion``. With
    ``a = 1e-12 < sqrt(QL_EPSILON)`` the model takes its degenerate branch;
    this port returned ``A = 0`` there, pricing *every* zero bond at exactly
    zero — a wrong answer that looks like a number rather than an error.
    """
    r0 = 0.05
    a = 1.0e-12
    b = 0.05
    sigma = 0.01
    lambda_ = 0.0
    now = 0.0
    maturity = 1.0

    model = Vasicek(r0=r0, a=a, b=b, sigma=sigma, lambda_=lambda_)

    expected = math.exp(
        -r0 * maturity + sigma * sigma * maturity * maturity * maturity / 6.0
    )
    tight(model.discount_bond_scalar(now, maturity, r0), expected)


def test_vasicek_ordinary_branch_is_unchanged() -> None:
    """The non-degenerate branch must be untouched by the limit fix."""
    model = Vasicek(r0=0.05, a=0.1, b=0.05, sigma=0.01, lambda_=0.0)
    p = model.discount_bond_scalar(0.0, 1.0, 0.05)
    # A one-year zero on a 5% short rate sits just under exp(-0.05); the exact
    # value is pinned by test_vasicek.py against the C++ probe, so this only
    # asserts the ordinary branch still produces a sane, non-degenerate number.
    assert abs(p - math.exp(-0.05)) < 1.0e-3
