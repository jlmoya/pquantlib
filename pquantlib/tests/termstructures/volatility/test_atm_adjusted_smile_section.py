"""Tests for AtmAdjustedSmileSection (recentered smile)."""

from __future__ import annotations

from pquantlib.termstructures.volatility.atm_adjusted_smile_section import (
    AtmAdjustedSmileSection,
)
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.sabr_smile_section import SabrSmileSection
from pquantlib.testing import reference_reader, tolerance

_REF = reference_reader.load("cluster/l10a")
_ADJ = _REF["atm_adjusted_smile_section"]


def _sabr_base() -> SabrSmileSection:
    return SabrSmileSection(
        forward=0.05,
        sabr_params=(0.04, 0.5, 0.4, -0.1),
        exercise_time=_ADJ["exercise_time"],
    )


def test_atm_level_overrides_base() -> None:
    sec = AtmAdjustedSmileSection(
        base=_sabr_base(),
        atm=_ADJ["target_atm"],
        recenter_smile=True,
    )
    tolerance.exact(sec.atm_level(), _ADJ["target_atm"])


def test_adjustment_value() -> None:
    sec = AtmAdjustedSmileSection(
        base=_sabr_base(),
        atm=_ADJ["target_atm"],
        recenter_smile=True,
    )
    tolerance.tight(sec.adjustment(), _ADJ["adjustment"])


def test_vol_at_new_atm_matches_base_at_old_atm() -> None:
    """When recentered, vol at the NEW atm shifts to base vol at OLD atm."""
    base = _sabr_base()
    sec = AtmAdjustedSmileSection(
        base=base, atm=_ADJ["target_atm"], recenter_smile=True,
    )
    # The C++ probe reports vol_at_new_atm == base_vol_at_5pct (where
    # 5pct is the base ATM and the recentered strike of 6pct + (-1pct
    # adjustment) = 5pct).
    tolerance.tight(sec.volatility(_ADJ["target_atm"]), _ADJ["vol_at_new_atm"])


def test_vol_at_strike_6pct_matches_probe() -> None:
    sec = AtmAdjustedSmileSection(
        base=_sabr_base(), atm=_ADJ["target_atm"], recenter_smile=True,
    )
    tolerance.tight(sec.volatility(0.06), _ADJ["vol_at_strike_6pct"])


def test_no_recenter_means_zero_adjustment() -> None:
    """recenter_smile=False is the C++ default and gives 0 shift."""
    sec = AtmAdjustedSmileSection(
        base=_sabr_base(), atm=_ADJ["target_atm"], recenter_smile=False,
    )
    tolerance.exact(sec.adjustment(), 0.0)


def test_atm_falls_back_to_base() -> None:
    base = _sabr_base()
    sec = AtmAdjustedSmileSection(base=base)
    tolerance.exact(sec.atm_level(), base.atm_level())


def test_adjustment_is_zero_when_no_atm_provided() -> None:
    base = _sabr_base()
    sec = AtmAdjustedSmileSection(base=base, recenter_smile=True)
    # adjustment = base_atm - atm = 0 (since atm = base_atm).
    tolerance.exact(sec.adjustment(), 0.0)


def test_strike_bounds_delegate_to_base() -> None:
    base = _sabr_base()
    sec = AtmAdjustedSmileSection(base=base, atm=0.07)
    assert sec.min_strike() == base.min_strike()
    assert sec.max_strike() == base.max_strike()


def test_volatility_type_delegates() -> None:
    base = _sabr_base()
    sec = AtmAdjustedSmileSection(base=base, atm=0.07)
    assert sec.volatility_type() == base.volatility_type()
    assert sec.shift() == base.shift()


def test_recenter_with_flat_base_volatility_unchanged() -> None:
    """A flat smile produces constant vol regardless of shift."""
    base = FlatSmileSection(volatility=0.18, exercise_time=2.0, atm_level=0.05)
    sec = AtmAdjustedSmileSection(base=base, atm=0.07, recenter_smile=True)
    # Flat vol stays flat after shift.
    tolerance.exact(sec.volatility(0.05), 0.18)
    tolerance.exact(sec.volatility(0.07), 0.18)
