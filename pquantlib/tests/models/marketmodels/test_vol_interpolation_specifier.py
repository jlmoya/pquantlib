"""Tests for the W10-A volatility interpolation specifiers.

Cross-validates against ``migration-harness/references/cluster/w10a.json``.

C++ parity:
  ql/models/marketmodels/models/volatilityinterpolationspecifier.hpp
  ql/models/marketmodels/models/volatilityinterpolationspecifierabcd.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.models.piecewise_constant_abcd_variance import (
    PiecewiseConstantAbcdVariance,
)
from pquantlib.models.marketmodels.models.volatility_interpolation_specifier_abcd import (
    VolatilityInterpolationSpecifierabcd,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w10a")


def _times_small() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]  # 0.5..3.0 -> 5 small rates


def _make_spec(last_caplet_vol: float = 0.0) -> VolatilityInterpolationSpecifierabcd:
    # period=2, offset=1, noBigRates=2 -> noSmallRates=5. Big-rate times are
    # small[offset + j*period] = small[1], small[3], small[5] = 1.0, 2.0, 3.0.
    times_small = _times_small()
    big_rt = [times_small[1], times_small[3], times_small[5]]
    big_vars = [
        PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 0, big_rt),
        PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 1, big_rt),
    ]
    return VolatilityInterpolationSpecifierabcd(
        2, 1, big_vars, times_small, last_caplet_vol
    )


def test_specifier_dimensions(ref: dict[str, Any]) -> None:
    spec = _make_spec()
    exact(float(spec.get_period()), ref["vis_period"])
    exact(float(spec.get_offset()), ref["vis_offset"])
    exact(float(spec.get_no_big_rates()), ref["vis_no_big"])
    exact(float(spec.get_no_small_rates()), ref["vis_no_small"])
    assert len(spec.interpolated_variances()) == spec.get_no_small_rates()
    assert len(spec.original_variances()) == spec.get_no_big_rates()


def test_specifier_interpolated_total_volatilities(ref: dict[str, Any]) -> None:
    spec = _make_spec()
    iv = spec.interpolated_variances()
    for k in range(spec.get_no_small_rates()):
        tight(iv[k].total_volatility(k), ref[f"vis_small_totvol_{k}"])


def test_specifier_interpolated_variances(ref: dict[str, Any]) -> None:
    spec = _make_spec()
    iv = spec.interpolated_variances()
    tight(iv[0].variance(0), ref["vis_small0_var0"])
    tight(iv[4].variance(0), ref["vis_small4_var0"])


def test_specifier_terminal_matches_auto_caplet_vol() -> None:
    # With last_caplet_vol=0.0 the terminal small rate's total vol is rescaled
    # to the last big rate's total vol (auto-detected).
    times_small = _times_small()
    big_rt = [times_small[1], times_small[3], times_small[5]]
    pv1 = PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 1, big_rt)
    spec = _make_spec()
    iv = spec.interpolated_variances()
    last = spec.get_no_small_rates() - 1
    tight(iv[last].total_volatility(last), pv1.total_volatility(1))


def test_specifier_set_scaling_factors(ref: dict[str, Any]) -> None:
    spec = _make_spec()
    spec.set_scaling_factors([1.1, 0.9])
    iv = spec.interpolated_variances()
    tight(iv[0].total_volatility(0), ref["vis_scaled_small0_totvol"])
    tight(iv[2].total_volatility(2), ref["vis_scaled_small2_totvol"])


def test_specifier_set_scaling_factors_wrong_size() -> None:
    spec = _make_spec()
    with pytest.raises(LibraryException):
        spec.set_scaling_factors([1.1, 0.9, 1.0])  # noBigRates is 2


def test_specifier_set_last_caplet_vol(ref: dict[str, Any]) -> None:
    spec = _make_spec()
    spec.set_last_caplet_vol(0.25)
    iv = spec.interpolated_variances()
    last = spec.get_no_small_rates() - 1
    tight(iv[last].total_volatility(last), ref["vis_lastvol_small4_totvol"])


def test_specifier_size_mismatch_rejected() -> None:
    # noSmallRates=5, but pass big rates that don't satisfy
    # (noSmall - offset)/period == noBig.
    times_small = _times_small()
    big_rt = [times_small[1], times_small[3], times_small[5]]
    big_vars = [PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 0, big_rt)]
    with pytest.raises(LibraryException):
        VolatilityInterpolationSpecifierabcd(2, 1, big_vars, times_small, 0.0)


def test_specifier_rate_time_mismatch_rejected() -> None:
    # Big-rate times that don't line up with small[offset + j*period].
    times_small = _times_small()
    bad_big_rt = [times_small[0], times_small[2], times_small[4]]  # offset 0, not 1
    big_vars = [
        PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 0, bad_big_rt),
        PiecewiseConstantAbcdVariance(-0.02, 0.5, 1.0, 0.14, 1, bad_big_rt),
    ]
    with pytest.raises(LibraryException):
        VolatilityInterpolationSpecifierabcd(2, 1, big_vars, times_small, 0.0)
