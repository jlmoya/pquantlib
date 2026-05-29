"""BaseCorrelationStructure — 2-D base-correlation surface.

# C++ parity: ql/experimental/credit/basecorrelationstructure.hpp (v1.42.1).

Base-correlation surfaces map (tranche-tenor, loss-level) -> correlation
quote. The C++ class is templated on a 2-D interpolator (bilinear or
bicubic-spline); the Python port keeps the same surface but takes the
interpolator as a constructor argument (delegated to
``pquantlib.math.interpolations.bilinear.BilinearInterpolation`` by default).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.credit.correlation_structure import (
    CorrelationTermStructure,
)
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.quotes.quote import Quote
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period

# An interpolation factory takes (xs, ys, z_matrix) -> callable(x, y) -> float.
# # C++ parity: corresponds to the ``Interpolator2D_T`` template parameter.
Interpolator2DFactory = Callable[[Array, Array, Matrix], object]


def _default_interpolator(xs: Array, ys: Array, z: Matrix) -> BilinearInterpolation:
    """Default 2-D interpolator — bilinear (matches C++ default)."""
    return BilinearInterpolation(xs, ys, z)


class BaseCorrelationStructure(CorrelationTermStructure):
    """Matrix-based base-correlation term structure.

    Tranche tenors and loss levels are passed at construction; the correlation
    values are themselves market quotes (so the surface can re-interpolate
    when one of them moves). The default interpolator is bilinear; pass a
    custom factory for bicubic / monotone-cubic etc.

    # C++ parity divergence: the C++ ``BaseCorrelationTermStructure`` is
    # templated on the 2-D interpolator type. Python uses a factory closure
    # so the runtime can swap interpolators without re-instantiating the
    # class — this matches downstream uses (`bilinear` for arbitrage-safe,
    # `bicubic_spline` for smoother surfaces).
    """

    __slots__ = (
        "_corr_quotes",
        "_correlations",
        "_interpolation",
        "_interpolator_factory",
        "_loss_levels",
        "_n_losses",
        "_n_tranche_tenors",
        "_tenors",
        "_tranche_dates",
        "_tranche_times",
    )

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        bdc: BusinessDayConvention,
        tenors: Sequence[Period],  # sorted
        loss_levels: Sequence[float],  # sorted in (0, 1]
        correlation_quotes: Sequence[Sequence[Quote]],
        day_counter: DayCounter,
        interpolator_factory: Interpolator2DFactory | None = None,
    ) -> None:
        super().__init__(
            bdc=bdc,
            calendar=calendar,
            day_counter=day_counter,
            settlement_days=settlement_days,
        )
        self._tenors = list(tenors)
        self._loss_levels = list(loss_levels)
        self._corr_quotes = [list(row) for row in correlation_quotes]
        self._n_tranche_tenors = len(tenors)
        self._n_losses = len(loss_levels)
        self._interpolator_factory = (
            interpolator_factory
            if interpolator_factory is not None
            else _default_interpolator
        )

        # Validate inputs.
        # # C++ parity: basecorrelationstructure.hpp checkTrancheTenors/checkLosses/checkInputs.
        self._check_tranche_tenors()
        self._check_losses()
        qassert.require(
            len(self._corr_quotes) == self._n_tranche_tenors,
            f"correl_quotes row count {len(self._corr_quotes)} != n_tranche_tenors {self._n_tranche_tenors}",
        )
        for row in self._corr_quotes:
            qassert.require(
                len(row) == self._n_losses,
                f"correl_quotes row width {len(row)} != n_losses {self._n_losses}",
            )

        # Compute tranche dates + times.
        self._tranche_dates = [
            calendar.advance_period(self.reference_date(), t, bdc) for t in tenors
        ]
        self._tranche_times = [
            self.time_from_reference(d) for d in self._tranche_dates
        ]
        self._correlations: Matrix = np.zeros(
            (self._n_losses, self._n_tranche_tenors), dtype=np.float64
        )
        self._update_matrix()

        # Register as observer of every quote so a quote update invalidates us.
        for row in self._corr_quotes:
            for q in row:
                q.register_with(self)

        # Build the interpolator.
        # # C++ parity divergence: matrix is in (loss, tenor) layout in C++.
        # The Python ``BilinearInterpolation`` expects shape (len(ys), len(xs))
        # which in our case is (n_losses, n_tranche_tenors); ys=loss_levels
        # x-axis=tranche_times. We pass them in that order.
        self._interpolation = self._interpolator_factory(
            np.asarray(self._tranche_times, dtype=np.float64),
            np.asarray(self._loss_levels, dtype=np.float64),
            self._correlations,
        )

    def _check_tranche_tenors(self) -> None:
        # # C++ parity: basecorrelationstructure.hpp:131-140.
        qassert.require(
            self._tenors[0].length > 0,
            f"first tranche tenor is non-positive ({self._tenors[0]})",
        )
        for i in range(1, self._n_tranche_tenors):
            qassert.require(
                self._tenors[i] > self._tenors[i - 1],
                f"non-increasing tranche tenor at index {i}",
            )

    def _check_losses(self) -> None:
        # # C++ parity: basecorrelationstructure.hpp:143-157.
        qassert.require(
            self._loss_levels[0] > 0.0,
            f"first loss level is non-positive ({self._loss_levels[0]})",
        )
        qassert.require(
            self._loss_levels[0] <= 1.0,
            f"first loss level > 100%: {self._loss_levels[0]}",
        )
        for i in range(1, self._n_losses):
            qassert.require(
                self._loss_levels[i] > self._loss_levels[i - 1],
                f"non-increasing loss level at index {i}",
            )
            qassert.require(
                self._loss_levels[i] <= 1.0,
                f"loss level {i} > 100%: {self._loss_levels[i]}",
            )

    def _update_matrix(self) -> None:
        # # C++ parity: basecorrelationstructure.hpp:193-198.
        for i in range(self._n_tranche_tenors):
            for j in range(self._n_losses):
                self._correlations[j, i] = self._corr_quotes[i][j].value()

    def update(self) -> None:
        """Refresh quote-driven matrix and forward to TermStructure observers."""
        # # C++ parity: basecorrelationstructure.hpp:187-190 — updateMatrix +
        # TermStructure::update.
        self._update_matrix()
        super().update()

    def correlation_size(self) -> int:
        # # C++ parity: basecorrelationstructure.hpp:93.
        return 1

    def max_date(self) -> Date:
        # # C++ parity: basecorrelationstructure.hpp:107.
        return self._tranche_dates[-1]

    def correlation(self, d: Date, loss_level: float) -> float:
        """Return the correlation at (date, loss-level) via interpolation."""
        return self.correlation_at_time(self.time_from_reference(d), loss_level)

    def correlation_at_time(self, t: float, loss_level: float) -> float:
        """Same as ``correlation`` but skips the date->time conversion."""
        # We always allow_extrapolation=True to mirror C++ which passes
        # ``extrapolate=true`` to the interpolator (basecorrelationstructure.hpp:114).
        return float(self._interpolation(t, loss_level, allow_extrapolation=True))  # pyright: ignore[reportCallIssue, reportUnknownArgumentType]

    def tenors(self) -> list[Period]:
        return list(self._tenors)

    def loss_levels(self) -> list[float]:
        return list(self._loss_levels)

    def tranche_dates(self) -> list[Date]:
        return list(self._tranche_dates)

    def tranche_times(self) -> list[float]:
        return list(self._tranche_times)

    def correlations_matrix(self) -> Matrix:
        """Defensive copy of the live correlation matrix."""
        return self._correlations.copy()
