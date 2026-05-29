"""CPI — namespace for inflation-fixing utilities.

# C++ parity: ``struct CPI`` in ql/indexes/inflationindex.hpp (v1.42.1).

The C++ ``CPI`` struct holds:

* ``InterpolationType`` enum — ``AsIndex / Flat / Linear``.
* ``laggedFixing(index, date, observationLag, interpolationType)`` —
  ``ZeroInflationIndex`` lookup at ``date - observationLag``, interpolated
  if asked.
* ``laggedYoYRate(index, date, observationLag, interpolationType)`` — same
  but for ``YoYInflationIndex``.

We port the struct as a Python module-level enum + two free functions
(``lagged_fixing`` / ``lagged_yoy_rate``). This avoids the awkwardness of
wrapping a static-method-only struct in a Python class.

# C++ parity divergence: the C++ Linear-mode branch has a fast-path when
# the requested date sits exactly on an inflation-period start (no
# interpolation needed → returns ``I0``). We preserve that fast-path. The
# branch that interpolates between ``I0`` and ``I1`` may need a forecast
# fixing on the period-end side; we surface the same error path as the
# index itself (qassert.fail through ZeroInflationIndex.fixing).
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.indexes.inflation.inflation_index import (
        YoYInflationIndex,
        ZeroInflationIndex,
    )
    from pquantlib.time.date import Date


class InterpolationType(IntEnum):
    """How to interpolate inflation fixings between period anchors.

    # C++ parity: ``CPI::InterpolationType`` in inflationindex.hpp:42-46.
    """

    AsIndex = 0  # same interpolation as index (==Flat for our zero/yoy ports)
    Flat = 1  # flat from previous fixing
    Linear = 2  # linear between bracketing fixings


_ONE_DAY: Period = Period(1, TimeUnit.Days)


def is_interpolated(interp: InterpolationType) -> bool:
    """True iff the interpolation is ``Linear``.

    # C++ parity: ``detail::CPI::isInterpolated``. ``AsIndex`` resolves to
    # the index's own ``interpolated()`` flag, which in v1.42.1 is always
    # ``False`` for ``ZeroInflationIndex``; we therefore report ``False``
    # for ``AsIndex`` here.
    """
    return interp == InterpolationType.Linear


def effective_interpolation_type(interp: InterpolationType) -> InterpolationType:
    """Collapse ``AsIndex`` to ``Flat``.

    # C++ parity: ``detail::CPI::effectiveInterpolationType`` in
    # ``inflationindex.cpp``. v1.42.1 treats ``AsIndex`` as ``Flat`` because
    # the index-side ``interpolated`` flag is fixed false.
    """
    if interp == InterpolationType.AsIndex:
        return InterpolationType.Flat
    return interp


def lagged_fixing(
    index: ZeroInflationIndex,
    d: Date,
    observation_lag: Period,
    interpolation_type: InterpolationType,
) -> float:
    """Inflation fixing on ``d - observation_lag``, interpolated if requested.

    # C++ parity: ``CPI::laggedFixing`` in ql/indexes/inflationindex.cpp:28-62.
    """
    if interpolation_type in (InterpolationType.AsIndex, InterpolationType.Flat):
        fixing_start, _ = inflation_period(d - observation_lag, index.frequency())
        return index.fixing(fixing_start)
    if interpolation_type == InterpolationType.Linear:
        fixing_start, fixing_end = inflation_period(d - observation_lag, index.frequency())
        interp_start, interp_end = inflation_period(d, index.frequency())
        i0 = index.fixing(fixing_start)
        if d == interp_start:
            # Fast path: avoid asking for the period-end fixing which might
            # require a forecast curve.
            return i0
        i1 = index.fixing(fixing_end + _ONE_DAY)
        # Linear interpolation: (d - interp_start) / (interp_end + 1day - interp_start).
        # Date - Date returns int day count (see Date.__sub__ overload).
        num = float(d - interp_start)
        denom = float((interp_end + _ONE_DAY) - interp_start)
        return i0 + (i1 - i0) * num / denom
    qassert.fail(f"unknown CPI interpolation type: {interpolation_type}")


def lagged_yoy_rate(
    index: YoYInflationIndex,
    d: Date,
    observation_lag: Period,
    interpolation_type: InterpolationType,
) -> float:
    """YoY inflation rate on ``d - observation_lag``, interpolated if requested.

    # C++ parity: ``CPI::laggedYoYRate`` in ql/indexes/inflationindex.cpp:65-116.
    # The ratio-mode-Linear branch interpolates underlying-index fixings
    # first then takes the ratio; we mirror that.
    """
    if interpolation_type == InterpolationType.AsIndex:
        return index.fixing(d - observation_lag)
    if interpolation_type == InterpolationType.Flat:
        fixing_start, _ = inflation_period(d - observation_lag, index.frequency())
        return index.fixing(fixing_start)
    if interpolation_type == InterpolationType.Linear:
        if index.ratio():
            # Underlying-index interpolation then ratio — only valid when no
            # forecast curve is needed (all fixings are historical). We
            # approximate the C++ ``needsForecast(date)`` predicate as
            # ``always-historical`` here since our port does not yet wire
            # forecasting paths through the YoYInflationTermStructure.
            underlying = index.underlying_index()
            qassert.require(
                underlying is not None,
                "ratio-mode YoYInflationIndex must have an underlying ZeroInflationIndex",
            )
            assert underlying is not None
            one_year = Period(1, TimeUnit.Years)
            z1 = lagged_fixing(underlying, d, observation_lag, interpolation_type)
            z0 = lagged_fixing(underlying, d - one_year, observation_lag, interpolation_type)
            return z1 / z0 - 1.0
        fixing_start, fixing_end = inflation_period(d - observation_lag, index.frequency())
        interp_start, interp_end = inflation_period(d, index.frequency())
        y0 = index.fixing(fixing_start)
        if d == interp_start:
            return y0
        y1 = index.fixing(fixing_end + _ONE_DAY)
        num = float(d - interp_start)
        denom = float((interp_end + _ONE_DAY) - interp_start)
        return y0 + (y1 - y0) * num / denom
    qassert.fail(f"unknown CPI interpolation type: {interpolation_type}")
