"""Tests for GeneralizedHullWhite (analytic-fitting surface).

Cross-validates against ``migration-harness/references/cluster/w8d.json``.

C++ parity: ql/experimental/shortrate/generalizedhullwhite.{hpp,cpp}
@ v1.42.1.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.shortrate.generalized_hull_white import GeneralizedHullWhite
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.payoffs import OptionType
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import custom, tight
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# The Gurrieri B/V/A integrals use composite-Simpson quadrature; the analytic
# Hull-White A/B use closed forms. They agree to ~1e-4 relative — looser than
# TIGHT but far better than LOOSE. Per-test justification documented inline.
_GHW_VS_HW_REL = 5e-4


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w8d")


def _curve() -> FlatForward:
    return FlatForward(Date.from_ymd(15, Month.January, 2024), SimpleQuote(0.04), Actual365Fixed())


def _ghw() -> GeneralizedHullWhite:
    return GeneralizedHullWhite(_curve(), 0.1, 0.012)


def _hw() -> HullWhite:
    return HullWhite(_curve(), a=0.1, sigma=0.012)


def test_inspectors() -> None:
    ghw = _ghw()
    tight(ghw.a(), 0.1)
    tight(ghw.sigma(), 0.012)


def test_discount_bond_matches_probe(ref: dict[str, Any]) -> None:
    """``discountBond(t,T,r)`` reproduces the C++ probe (TIGHT)."""
    ghw = _ghw()
    tight(ghw.discount_bond_scalar(2.0, 5.0, 0.045), ref["ghw_discountbond_2_5_r"])


def test_discount_bond_option_matches_probe(ref: dict[str, Any]) -> None:
    """``discountBondOption`` reproduces the C++ probe (TIGHT)."""
    ghw = _ghw()
    tight(
        ghw.discount_bond_option(OptionType.Call, 0.85, 2.0, 5.0),
        ref["ghw_dbo_call"],
    )
    tight(
        ghw.discount_bond_option(OptionType.Put, 0.85, 2.0, 5.0),
        ref["ghw_dbo_put"],
    )


def test_reduces_to_classical_hull_white() -> None:
    """At constant reversion + vol the analytic GHW matches classical HW.

    The match is at ``5e-4`` relative (not TIGHT): the GHW Gurrieri
    formulae use composite-Simpson numerical integration for B / V while
    classical Hull-White uses closed forms. Both anchor to the same C++
    probe values at TIGHT individually; this cross-check confirms the
    *model equivalence* to engineering precision.
    """
    ghw = _ghw()
    hw = _hw()
    custom(
        ghw.discount_bond_scalar(2.0, 5.0, 0.045),
        hw.discount_bond_scalar(2.0, 5.0, 0.045),
        rel_tol=_GHW_VS_HW_REL,
        abs_tol=0.0,
        reason="Gurrieri Simpson-integral B/V vs HW closed form",
    )
    custom(
        ghw.discount_bond_option(OptionType.Call, 0.85, 2.0, 5.0),
        hw.discount_bond_option(OptionType.Call, 0.85, 2.0, 5.0),
        rel_tol=_GHW_VS_HW_REL,
        abs_tol=0.0,
        reason="Gurrieri Simpson-integral V vs HW closed-form bond vol",
    )


def test_piecewise_matches_probe(ref: dict[str, Any]) -> None:
    """Piecewise (time-varying) GHW reproduces the C++ probe (TIGHT)."""
    speed_dates = [Date.from_ymd(15, Month.January, 2024), Date.from_ymd(15, Month.January, 2026)]
    vol_dates = [Date.from_ymd(15, Month.January, 2024), Date.from_ymd(15, Month.January, 2026)]
    ghw = GeneralizedHullWhite.piecewise(
        _curve(), speed_dates, vol_dates, [0.1, 0.2], [0.010, 0.015]
    )
    tight(ghw.discount_bond_scalar(2.0, 5.0, 0.045), ref["ghwpw_discountbond_2_5_r"])
    tight(
        ghw.discount_bond_option(OptionType.Call, 0.85, 2.0, 5.0),
        ref["ghwpw_dbo_call"],
    )


def test_dynamics_raises_use_hw_dynamics() -> None:
    """``dynamics()`` deliberately fails — use ``hw_dynamics()``."""
    ghw = _ghw()
    with pytest.raises(LibraryException):
        ghw.dynamics()
    # hw_dynamics() works and round-trips the short-rate change of variable.
    dyn = ghw.hw_dynamics()
    r = 0.05
    t = 1.0
    x = dyn.variable(t, r)
    tight(dyn.short_rate(t, x), r)


def test_fixed_reversion_mask() -> None:
    """``fixed_reversion`` masks the reversion slots True, vol slots False."""
    speed_dates = [Date.from_ymd(15, Month.January, 2024), Date.from_ymd(15, Month.January, 2026)]
    vol_dates = [Date.from_ymd(15, Month.January, 2024)]
    ghw = GeneralizedHullWhite.piecewise(
        _curve(), speed_dates, vol_dates, [0.1, 0.2], [0.012]
    )
    assert ghw.fixed_reversion() == [True, True, False]
