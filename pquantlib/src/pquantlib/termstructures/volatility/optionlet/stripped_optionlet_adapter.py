"""StrippedOptionletAdapter — wraps a StrippedOptionletBase as a vol surface.

# C++ parity: ql/termstructures/volatility/optionlet/strippedoptionletadapter.{hpp,cpp}
# (v1.42.1).

Builds per-fixing-time linear strike interpolants, then linearly
interpolates over fixing times to produce ``volatility(t, strike)``.
The C++ class also exposes smile sections — those require cubic
interpolation across strikes which lands in the cubic-spline
carve-out; PQuantLib defers them.

Inputs:
    A non-empty ``StrippedOptionletBase``. The adapter forwards
    settlement_days / calendar / convention / day counter /
    volatility_type / displacement from the base.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_base import (
    StrippedOptionletBase,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date


class StrippedOptionletAdapter(OptionletVolatilityStructure):
    """Adapter from a ``StrippedOptionletBase`` to a vol surface."""

    def __init__(self, stripped: StrippedOptionletBase) -> None:
        super().__init__(
            business_day_convention=stripped.business_day_convention(),
            calendar=stripped.calendar(),
            day_counter=stripped.day_counter(),
            settlement_days=stripped.settlement_days(),
        )
        self._stripped: StrippedOptionletBase = stripped
        self._n: int = stripped.optionlet_maturities()
        qassert.require(self._n > 0, "no optionlet maturities")

        # Build per-time linear-in-strike interpolants.
        self._strike_interps: list[LinearInterpolation] = []
        for i in range(self._n):
            strikes = np.asarray(stripped.optionlet_strikes(i), dtype=np.float64)
            vols = np.asarray(stripped.optionlet_volatilities(i), dtype=np.float64)
            qassert.require(
                len(strikes) >= 2,
                f"need >=2 strikes for the strike-axis interpolation; "
                f"got {len(strikes)} at row {i}",
            )
            self._strike_interps.append(LinearInterpolation(strikes, vols))

    # --- TermStructure interface -----------------------------------------

    def max_date(self) -> Date:
        # C++ parity: returns the last optionlet fixing date.
        return self._stripped.optionlet_fixing_dates()[-1]

    def min_strike(self) -> float:
        return float(self._stripped.optionlet_strikes(0)[0])

    def max_strike(self) -> float:
        return float(self._stripped.optionlet_strikes(0)[-1])

    def volatility_type(self) -> VolatilityType:
        return self._stripped.volatility_type()

    def displacement(self) -> float:
        return self._stripped.displacement()

    # --- subclass hook ---------------------------------------------------

    def _volatility_impl(self, t: float, strike: float) -> float:
        # # C++ parity: StrippedOptionletAdapter::volatilityImpl:
        # for each optionlet date i, evaluate the strike interp at
        # ``strike``; then linearly interpolate the resulting vector
        # over the optionlet fixing times at ``t``.
        vol_at_strike = np.asarray(
            [
                self._strike_interps[i](strike, allow_extrapolation=True)
                for i in range(self._n)
            ],
            dtype=np.float64,
        )
        times = np.asarray(self._stripped.optionlet_fixing_times(), dtype=np.float64)
        # Need >=2 fixing times for the time interp.
        if len(times) < 2:
            return float(vol_at_strike[0])
        return LinearInterpolation(times, vol_at_strike)(t, allow_extrapolation=True)
