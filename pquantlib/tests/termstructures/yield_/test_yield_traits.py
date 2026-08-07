"""Cross-validation of the yield-curve bootstrap traits against C++ v1.43.

# C++ parity: ql/termstructures/yield/bootstraptraits.hpp.

Reference: ``migration-harness/references/v143/ts/traits.json``, produced by
``migration-harness/cpp/probes/v143_ts_traits/probe.cpp``.

Every trait member here is closed form (the extrapolating ``guess`` branches
run the curve's own interpolation, which is itself pinned elsewhere), so the
whole file is TIGHT tier.

The probe builds the traits against fully-constructed interpolated curves,
never a mid-bootstrap curve, so ``times()[i]`` / ``data()[i-1]`` are always
in range. The node data deliberately contains a negative rate so the
``r < 0 ? r*2 : r/2`` sign branch of ``min/max_value_after`` is exercised.
"""

from __future__ import annotations

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.interpolations.log_linear import LogLinearInterpolation
from pquantlib.termstructures.yield_.interpolated_discount_curve import (
    InterpolatedDiscountCurve,
)
from pquantlib.termstructures.yield_.interpolated_forward_curve import (
    InterpolatedForwardCurve,
)
from pquantlib.termstructures.yield_.interpolated_zero_curve import (
    InterpolatedZeroCurve,
)
from pquantlib.termstructures.yield_.yield_traits import (
    Discount,
    ForwardRate,
    ZeroYield,
)
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import exact, tight
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

CPP = reference_reader.load("v143/ts/traits")

# probe.cpp:57 — ``const Date kRef(15, January, 2024)``.
REF_DATE = Date.from_ymd(15, Month.January, 2024)

# probe.cpp:59-67 — nodeDates().
NODE_DATES = [
    REF_DATE,
    REF_DATE + Period(6, TimeUnit.Months),
    REF_DATE + Period(1, TimeUnit.Years),
    REF_DATE + Period(2, TimeUnit.Years),
    REF_DATE + Period(5, TimeUnit.Years),
    REF_DATE + Period(10, TimeUnit.Years),
]
# probe.cpp:71-73 — nodeDiscounts().
NODE_DISCOUNTS = [1.0, 0.985, 0.968, 0.930, 0.815, 0.640]
# probe.cpp:77-79 — nodeRates(); note the negative third entry.
NODE_RATES = [0.030, 0.031, -0.005, 0.035, 0.0405, 0.0446]
# probe.cpp:83 — kPillars.
PILLARS = [1, 2, 3, 5]
# probe.cpp:87-88.
TRANSFORM_DIRECT_X = -0.35
TRANSFORM_INVERSE_X = 0.87


def _discount_curve() -> InterpolatedDiscountCurve:
    c = InterpolatedDiscountCurve(
        dates=NODE_DATES,
        dfs=NODE_DISCOUNTS,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        interpolator=LogLinearInterpolation,
    )
    c.enable_extrapolation()
    return c


def _zero_curve() -> InterpolatedZeroCurve:
    c = InterpolatedZeroCurve(
        dates=NODE_DATES,
        yields=NODE_RATES,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        interpolator=LinearInterpolation,
    )
    c.enable_extrapolation()
    return c


def _forward_curve() -> InterpolatedForwardCurve:
    c = InterpolatedForwardCurve(
        dates=NODE_DATES,
        forwards=NODE_RATES,
        day_counter=Actual365Fixed(),
        calendar=TARGET(),
        interpolator=LinearInterpolation,
    )
    c.enable_extrapolation()
    return c


def test_node_dates_match_the_probe() -> None:
    """Guard: the whole file is meaningless if the grids differ."""
    assert [d.serial_number() for d in NODE_DATES] == CPP["discount_dates"]
    assert REF_DATE.serial_number() == CPP["ref_date"]


# --- Discount ---------------------------------------------------------------


def test_discount_grid_and_scalars() -> None:
    c = _discount_curve()
    exact(Discount().initial_value(c), CPP["discount_initial_value"])
    assert Discount().initial_date(c).serial_number() == CPP["discount_initial_date"]
    assert Discount().max_iterations() == int(CPP["discount_max_iterations"])
    for got, want in zip(c.times(), CPP["discount_times"], strict=True):
        tight(got, want)
    for got, want in zip(c.data(), CPP["discount_data"], strict=True):
        exact(got, want)


@pytest.mark.parametrize("i", PILLARS)
def test_discount_guess(i: int) -> None:
    """``i == 1`` uses ``times()[1]``; ``i > 1`` flat-rate extrapolates.

    # C++ parity: bootstraptraits.hpp:61-75.
    """
    c = _discount_curve()
    tight(Discount().guess(i, c, True, 0), CPP[f"discount_i{i}_guess_valid"])
    tight(Discount().guess(i, c, False, 0), CPP[f"discount_i{i}_guess_invalid"])


@pytest.mark.parametrize("i", PILLARS)
def test_discount_bracket(i: int) -> None:
    """``minValueAfter`` uses the real ``dt``; ``maxValueAfter`` ignores validData.

    # C++ parity: bootstraptraits.hpp:78-101.
    """
    c = _discount_curve()
    tight(Discount().min_value_after(i, c, True, 0), CPP[f"discount_i{i}_min_valid"])
    tight(
        Discount().min_value_after(i, c, False, 0), CPP[f"discount_i{i}_min_invalid"]
    )
    tight(Discount().max_value_after(i, c, True, 0), CPP[f"discount_i{i}_max_valid"])
    tight(
        Discount().max_value_after(i, c, False, 0), CPP[f"discount_i{i}_max_invalid"]
    )


@pytest.mark.parametrize("i", PILLARS)
def test_discount_transforms(i: int) -> None:
    """# C++ parity: bootstraptraits.hpp:104-113 — ``exp`` / ``log``."""
    c = _discount_curve()
    tight(
        Discount().transform_direct(TRANSFORM_DIRECT_X, i, c),
        CPP[f"discount_i{i}_transform_direct"],
    )
    tight(
        Discount().transform_inverse(TRANSFORM_INVERSE_X, i, c),
        CPP[f"discount_i{i}_transform_inverse"],
    )


@pytest.mark.parametrize("i", [1, 2])
def test_discount_update_guess(i: int) -> None:
    """Discount writes only ``data[i]`` — no mirror into ``data[0]``."""
    data = list(NODE_DISCOUNTS)
    Discount().update_guess(data, 0.123456789, i)
    for got, want in zip(data, CPP[f"discount_i{i}_update_guess"], strict=True):
        exact(got, want)


# --- ZeroYield --------------------------------------------------------------


def test_zero_yield_scalars() -> None:
    c = _zero_curve()
    exact(ZeroYield().initial_value(c), CPP["zeroyield_initial_value"])
    assert ZeroYield().initial_date(c).serial_number() == CPP["zeroyield_initial_date"]
    assert ZeroYield().max_iterations() == int(CPP["zeroyield_max_iterations"])


@pytest.mark.parametrize("i", PILLARS)
def test_zero_yield_guess(i: int) -> None:
    """``i > 1`` extrapolates via ``zeroRate(d, dc, Continuous, Annual, true)``.

    # C++ parity: bootstraptraits.hpp:157-160. Returning ``data[i-1]``
    # instead is the classic port shortcut; the reference values are not
    # ``data[i-1]``.
    """
    c = _zero_curve()
    tight(ZeroYield().guess(i, c, True, 0), CPP[f"zeroyield_i{i}_guess_valid"])
    tight(ZeroYield().guess(i, c, False, 0), CPP[f"zeroyield_i{i}_guess_invalid"])


@pytest.mark.parametrize("i", PILLARS)
def test_zero_yield_bracket(i: int) -> None:
    """# C++ parity: bootstraptraits.hpp:164-191."""
    c = _zero_curve()
    tight(ZeroYield().min_value_after(i, c, True, 0), CPP[f"zeroyield_i{i}_min_valid"])
    tight(
        ZeroYield().min_value_after(i, c, False, 0), CPP[f"zeroyield_i{i}_min_invalid"]
    )
    tight(ZeroYield().max_value_after(i, c, True, 0), CPP[f"zeroyield_i{i}_max_valid"])
    tight(
        ZeroYield().max_value_after(i, c, False, 0), CPP[f"zeroyield_i{i}_max_invalid"]
    )


@pytest.mark.parametrize("i", PILLARS)
def test_zero_yield_transforms_are_identity(i: int) -> None:
    """# C++ parity: bootstraptraits.hpp:194-203."""
    c = _zero_curve()
    exact(
        ZeroYield().transform_direct(TRANSFORM_DIRECT_X, i, c),
        CPP[f"zeroyield_i{i}_transform_direct"],
    )
    exact(
        ZeroYield().transform_inverse(TRANSFORM_INVERSE_X, i, c),
        CPP[f"zeroyield_i{i}_transform_inverse"],
    )


@pytest.mark.parametrize("i", [1, 2])
def test_zero_yield_update_guess(i: int) -> None:
    """Pillar 1 is mirrored into pillar 0. # C++ parity: bootstraptraits.hpp:206-212."""
    data = list(NODE_RATES)
    ZeroYield().update_guess(data, 0.123456789, i)
    for got, want in zip(data, CPP[f"zeroyield_i{i}_update_guess"], strict=True):
        exact(got, want)


# --- ForwardRate ------------------------------------------------------------


def test_forward_rate_scalars() -> None:
    c = _forward_curve()
    exact(ForwardRate().initial_value(c), CPP["forwardrate_initial_value"])
    assert (
        ForwardRate().initial_date(c).serial_number() == CPP["forwardrate_initial_date"]
    )
    assert ForwardRate().max_iterations() == int(CPP["forwardrate_max_iterations"])


@pytest.mark.parametrize("i", PILLARS)
def test_forward_rate_guess(i: int) -> None:
    """``i > 1`` extrapolates via ``forwardRate(d, d, ...)`` — the instantaneous
    forward, computed by finite difference, so it is NOT exactly ``data[i]``.

    # C++ parity: bootstraptraits.hpp:250-253.
    """
    c = _forward_curve()
    tight(ForwardRate().guess(i, c, True, 0), CPP[f"forwardrate_i{i}_guess_valid"])
    tight(ForwardRate().guess(i, c, False, 0), CPP[f"forwardrate_i{i}_guess_invalid"])


@pytest.mark.parametrize("i", PILLARS)
def test_forward_rate_bracket(i: int) -> None:
    """# C++ parity: bootstraptraits.hpp:257-284."""
    c = _forward_curve()
    tight(
        ForwardRate().min_value_after(i, c, True, 0),
        CPP[f"forwardrate_i{i}_min_valid"],
    )
    tight(
        ForwardRate().min_value_after(i, c, False, 0),
        CPP[f"forwardrate_i{i}_min_invalid"],
    )
    tight(
        ForwardRate().max_value_after(i, c, True, 0),
        CPP[f"forwardrate_i{i}_max_valid"],
    )
    tight(
        ForwardRate().max_value_after(i, c, False, 0),
        CPP[f"forwardrate_i{i}_max_invalid"],
    )


@pytest.mark.parametrize("i", PILLARS)
def test_forward_rate_transforms_are_identity(i: int) -> None:
    """# C++ parity: bootstraptraits.hpp:287-296."""
    c = _forward_curve()
    exact(
        ForwardRate().transform_direct(TRANSFORM_DIRECT_X, i, c),
        CPP[f"forwardrate_i{i}_transform_direct"],
    )
    exact(
        ForwardRate().transform_inverse(TRANSFORM_INVERSE_X, i, c),
        CPP[f"forwardrate_i{i}_transform_inverse"],
    )


@pytest.mark.parametrize("i", [1, 2])
def test_forward_rate_update_guess(i: int) -> None:
    """# C++ parity: bootstraptraits.hpp:299-305."""
    data = list(NODE_RATES)
    ForwardRate().update_guess(data, 0.123456789, i)
    for got, want in zip(data, CPP[f"forwardrate_i{i}_update_guess"], strict=True):
        exact(got, want)
