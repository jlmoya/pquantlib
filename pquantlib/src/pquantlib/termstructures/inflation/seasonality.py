"""Seasonality + MultiplicativePriceSeasonality for inflation curves.

# C++ parity: ql/termstructures/inflation/seasonality.{hpp,cpp} (v1.42.1).

C++ exposes an abstract ``Seasonality`` interface with
``correctZeroRate``/``correctYoYRate``/``isConsistent`` and the concrete
``MultiplicativePriceSeasonality`` (stationary or multi-year cycle,
factor-based correction to a CPI/RPI/HICP price curve). ``KerkhofSeasonality``
is a tail subclass with a different correction kernel and is deferred to
Phase 8+ (zero current callers among the L7-A/B/C/D scope).

Math, C++ verbatim (MultiplicativePriceSeasonality::seasonalityCorrection):

    factorAt          = factor(atDate)
    if isZeroRate:
        factorBase    = factor(curveBaseDate)
        seasonalityAt = factorAt / factorBase
        period        = inflationPeriod(atDate, frequency())
        timeFromCurveBase = dayCounter.yearFraction(curveBaseDate, period.first)
        f             = (seasonalityAt) ** (1 / timeFromCurveBase)
    else (YoY):
        factor1YBefore = factor(atDate - 1 year)
        f             = factorAt / factor1YBefore
    return (rate + 1) * f - 1

Divergences inline-noted in code where the Python idiom differs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

from pquantlib import qassert
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.termstructures.inflation.inflation_term_structure import (
        InflationTermStructure,
    )


_ONE_YEAR: Final[Period] = Period(1, TimeUnit.Years)


class Seasonality(ABC):
    """Abstract base for seasonality corrections.

    # C++ parity: ``Seasonality`` in seasonality.hpp.
    """

    @abstractmethod
    def correct_zero_rate(
        self, d: Date, r: float, ts: InflationTermStructure
    ) -> float:
        """Apply seasonality to a zero-coupon inflation rate."""

    @abstractmethod
    def correct_yoy_rate(
        self, d: Date, r: float, ts: InflationTermStructure
    ) -> float:
        """Apply seasonality to a year-on-year inflation rate."""

    def is_consistent(self, ts: InflationTermStructure) -> bool:
        """Whether the seasonality is consistent with ``ts``.

        # C++ parity: ``Seasonality::isConsistent`` defaults to True.
        """
        del ts
        return True


class MultiplicativePriceSeasonality(Seasonality):
    """Stationary or multi-year multiplicative seasonality on a price index.

    # C++ parity: ``MultiplicativePriceSeasonality`` in seasonality.{hpp,cpp}.
    # Factors are reused cyclically (wrap around), so a 24-factor list with
    # Monthly frequency is a 2-year non-stationary cycle.
    """

    def __init__(
        self,
        seasonality_base_date: Date,
        frequency: Frequency,
        seasonality_factors: list[float],
    ) -> None:
        self._frequency: Frequency = frequency
        self._seasonality_factors: list[float] = list(seasonality_factors)
        self._seasonality_base_date: Date = seasonality_base_date
        self._validate()

    # ---- inspectors --------------------------------------------------

    def seasonality_base_date(self) -> Date:
        return self._seasonality_base_date

    def frequency(self) -> Frequency:
        return self._frequency

    def seasonality_factors(self) -> list[float]:
        return list(self._seasonality_factors)

    def seasonality_factor(self, to: Date) -> float:
        """Return the raw multiplicative factor at ``to`` (no normalization).

        # C++ parity: ``MultiplicativePriceSeasonality::seasonalityFactor``.
        # The C++ algorithm reverse-engineers the index in the factor cycle
        # from days-between, depending on the units of the factor period.
        """
        from_date = self._seasonality_base_date
        factor_period = Period.from_frequency(self._frequency)
        n_factors = len(self._seasonality_factors)
        if from_date == to:
            which = 0
        else:
            diff_days = abs(to - from_date)
            direction = 1 if from_date <= to else -1
            unit = factor_period.units
            if unit == TimeUnit.Days:
                diff = direction * diff_days
            elif unit == TimeUnit.Weeks:
                diff = direction * (diff_days // 7)
            elif unit == TimeUnit.Months:
                lim_first, lim_second = inflation_period(to, self._frequency)
                # diff is an upper-bound seed (31 days per factor-length month);
                # C++ then walks forward/back until the target lands inside the
                # bracketing inflation period.
                diff = diff_days // (31 * factor_period.length)
                go = from_date + (direction * diff * factor_period)
                while not (lim_first <= go <= lim_second):
                    go = go + (direction * factor_period)
                    diff += 1
                diff = direction * diff
            elif unit == TimeUnit.Years:
                qassert.fail(
                    f"seasonality period time unit is not allowed to be : {unit}"
                )
            else:
                qassert.fail(f"unknown time unit: {unit}")
            # Wrap into the cyclic factor index, direction-aware.
            which = (
                diff % n_factors
                if direction == 1
                else (n_factors - (-diff % n_factors)) % n_factors
            )
        return self._seasonality_factors[which]

    # ---- corrections -------------------------------------------------

    def correct_zero_rate(
        self, d: Date, r: float, ts: InflationTermStructure
    ) -> float:
        """Apply seasonality to the zero-coupon inflation rate.

        # C++ parity: ``MultiplicativePriceSeasonality::correctZeroRate``.
        """
        curve_base_date = ts.base_date()
        effective_fixing_date, _ = inflation_period(d, ts.frequency())
        return self._seasonality_correction(
            r, effective_fixing_date, ts, curve_base_date, is_zero_rate=True
        )

    def correct_yoy_rate(
        self, d: Date, r: float, ts: InflationTermStructure
    ) -> float:
        """Apply seasonality to the year-on-year inflation rate.

        # C++ parity: ``MultiplicativePriceSeasonality::correctYoYRate``.
        """
        _, curve_base_date = inflation_period(ts.base_date(), ts.frequency())
        return self._seasonality_correction(
            r, d, ts, curve_base_date, is_zero_rate=False
        )

    def is_consistent(self, ts: InflationTermStructure) -> bool:
        """Whether this seasonality is consistent with ``ts``.

        # C++ parity: ``MultiplicativePriceSeasonality::isConsistent``.
        # Multi-year (non-stationary) cycles must agree on the curve-base
        # anchor across each yearly slice; stationary cycles always pass.
        # Daily seasonality (frequency = Daily) is exempt per C++.
        """
        if self._frequency == Frequency.Daily:
            return True
        if int(self._frequency) == len(self._seasonality_factors):
            return True
        n_test = len(self._seasonality_factors) // int(self._frequency)
        _, curve_base_date = inflation_period(ts.base_date(), ts.frequency())
        factor_base = self.seasonality_factor(curve_base_date)
        eps = 0.00001
        for i in range(1, n_test):
            factor_at = self.seasonality_factor(curve_base_date + Period(i, TimeUnit.Years))
            qassert.require(
                abs(factor_at - factor_base) < eps,
                f"seasonality is inconsistent with inflation term structure, factors "
                f"{factor_base} and later factor {factor_at}, {i} years later from "
                f"inflation curve with base date at {curve_base_date}",
            )
        return True

    # ---- guts --------------------------------------------------------

    def _validate(self) -> None:
        """Mirror C++ validate(): allowed frequencies + factor-count rule."""
        f = self._frequency
        if f not in (
            Frequency.Semiannual,
            Frequency.EveryFourthMonth,
            Frequency.Quarterly,
            Frequency.Bimonthly,
            Frequency.Monthly,
            Frequency.Biweekly,
            Frequency.Weekly,
            Frequency.Daily,
        ):
            qassert.fail(
                f"bad frequency specified: {f}, "
                "only semi-annual through daily permitted."
            )
        qassert.require(
            len(self._seasonality_factors) > 0, "no seasonality factors given"
        )
        qassert.require(
            len(self._seasonality_factors) % int(f) == 0,
            f"For frequency {f} require multiple of {int(f)} factors "
            f"{len(self._seasonality_factors)} were given.",
        )

    def _seasonality_correction(
        self,
        rate: float,
        at_date: Date,
        ts: InflationTermStructure,
        curve_base_date: Date,
        *,
        is_zero_rate: bool,
    ) -> float:
        """Compute the corrected rate. See module docstring for the formula."""
        factor_at = self.seasonality_factor(at_date)
        if is_zero_rate:
            factor_base = self.seasonality_factor(curve_base_date)
            seasonality_at = factor_at / factor_base
            period_start, _ = inflation_period(at_date, self._frequency)
            time_from_curve_base = ts.day_counter().year_fraction(
                curve_base_date, period_start
            )
            f = seasonality_at ** (1.0 / time_from_curve_base)
        else:
            factor_1y_before = self.seasonality_factor(at_date - _ONE_YEAR)
            f = factor_at / factor_1y_before
        return (rate + 1.0) * f - 1.0
