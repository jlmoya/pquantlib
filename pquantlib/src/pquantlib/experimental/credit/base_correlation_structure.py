"""BaseCorrelationTermStructure — 2-D base-correlation surface.

# C++ parity: ql/experimental/credit/basecorrelationstructure.hpp:50-198
# + basecorrelationstructure.cpp:28-47 (v1.43).

Base-correlation surfaces map (tranche-tenor, loss-level) -> correlation
quote. The C++ class is templated on a 2-D interpolator (bilinear or
bicubic-spline); the Python port keeps the same surface but takes the
interpolator as a constructor argument (delegated to
``pquantlib.math.interpolations.bilinear.BilinearInterpolation`` by default).

Two C++ defects are reproduced verbatim; both are pinned by the
``bcts_*`` block of ``migration-harness/references/v143/experimental/creditloss.json``.
See :class:`BaseCorrelationTermStructure` for the derivations.
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


class BaseCorrelationTermStructure(CorrelationTermStructure):
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

    # C++ parity note (DEFECT 1, reproduced verbatim): the constructor calls
    # ``checkInputs(correlations_.rows(), correlations_.columns())``
    # (basecorrelationstructure.hpp:84) — that is ``(nTenors, nLosses)``,
    # since ``correlations_`` is built as
    # ``Matrix(correls.size(), correls.front().size())`` (hpp:71). But
    # ``checkInputs`` asserts
    #     QL_REQUIRE(nLosses_==volRows, ...)
    #     QL_REQUIRE(nTrancheTenors_==volsColumns, ...)
    # (hpp:168-175), i.e. ``nLosses == nTenors`` AND ``nTenors == nLosses``.
    # A perfectly well-formed non-square surface therefore always throws.
    # Pinned by ``bcts_defect_non_square_throws`` (true).

    # C++ parity note (DEFECT 2, reproduced verbatim): the documented layout
    # is ``correls[iYear][iLoss]`` (hpp:54) and ``updateMatrix`` fills
    # ``correlations_[i][j] = correlHandles_[i][j]`` with i = tenor,
    # j = loss (hpp:193-198). But ``setupInterpolation`` hands that matrix to
    #     BilinearInterpolation(trancheTimes_.begin(), trancheTimes_.end(),
    #                           lossLevel_.begin(), lossLevel_.end(),
    #                           correlations_)
    # (basecorrelationstructure.cpp:31-34), and ``Interpolation2D`` indexes
    # its z as ``zData_[y_index][x_index]`` — so row i is read as the LOSS
    # index and column j as the TENOR index. The surface therefore comes out
    # TRANSPOSED: ``correlation(t_i, l_j) == correls[j][i]``. Only the
    # square-shape accident of defect 1 keeps this from being a size error.
    # Pinned by ``bcts_on_node``: quotes {{.1,.2,.3},{.4,.5,.6},{.7,.8,.9}}
    # read back, walking (tenor-major, loss-minor), as
    # [.1,.4,.7, .2,.5,.8, .3,.6,.9] rather than [.1,.2,.3, .4,.5,.6, ...].

    # C++ parity note: ``checkLosses`` (hpp:143-157) is public but the C++
    # constructor never calls it (hpp:77-88 runs checkTrancheTenors,
    # initializeTrancheTimes, checkInputs, updateMatrix,
    # registerWithMarketData, setupInterpolation — and nothing else). So an
    # unsorted / out-of-range loss-level vector is accepted at construction.
    # Reproduced: :meth:`check_losses` exists and is callable, but the
    # constructor does not invoke it.
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

        # The C++ constructor body, in order (basecorrelationstructure.hpp:77-88).
        self.check_tranche_tenors()

        self._tranche_dates = [
            calendar.advance_period(self.reference_date(), t, bdc) for t in tenors
        ]
        self._tranche_times = [
            self.time_from_reference(d) for d in self._tranche_dates
        ]

        # ``correlations_`` is Matrix(correls.size(), correls.front().size())
        # = (rows = n_tranche_tenors, cols = n_losses) — hpp:71.
        self._correlations: Matrix = np.zeros(
            (len(self._corr_quotes), len(self._corr_quotes[0])), dtype=np.float64
        )
        # DEFECT 1: the arguments are (rows, columns) = (nTenors, nLosses),
        # but check_inputs compares them the other way round.
        self.check_inputs(
            self._correlations.shape[0], self._correlations.shape[1]
        )
        self.update_matrix()

        # Register as observer of every quote so a quote update invalidates us.
        # # C++ parity: registerWithMarketData (hpp:178-184).
        for row in self._corr_quotes:
            for q in row:
                q.register_with(self)

        # DEFECT 2: x = tranche times, y = loss levels, z = correlations_ as
        # filled — i.e. z is read as z[i_loss][i_tenor] but was written as
        # [i_tenor][i_loss]. # C++ parity: basecorrelationstructure.cpp:31-34.
        self._interpolation = self._interpolator_factory(
            np.asarray(self._tranche_times, dtype=np.float64),
            np.asarray(self._loss_levels, dtype=np.float64),
            self._correlations,
        )

    def check_inputs(self, vol_rows: int, vols_columns: int) -> None:
        """Validate the correlation-matrix shape.

        # C++ parity: basecorrelationstructure.hpp:165-176. See the DEFECT 1
        # note on the class: the comparison is transposed relative to the
        # arguments the constructor supplies, so only square surfaces pass.
        """
        qassert.require(
            self._n_losses == vol_rows,
            f"mismatch between number of loss levels ({self._n_losses}) and "
            f"number of rows ({vol_rows}) in the correl matrix",
        )
        qassert.require(
            self._n_tranche_tenors == vols_columns,
            f"mismatch between number of tranche tenors ({self._n_tranche_tenors}) "
            f"and number of columns ({vols_columns}) in the correl matrix",
        )

    def check_tranche_tenors(self) -> None:
        # # C++ parity: basecorrelationstructure.hpp:130-140.
        qassert.require(
            self._tenors[0].length > 0,
            f"first tranche tenor is non-positive ({self._tenors[0]})",
        )
        for i in range(1, self._n_tranche_tenors):
            qassert.require(
                self._tenors[i] > self._tenors[i - 1],
                f"non-increasing tranche tenor at index {i}",
            )

    def check_losses(self) -> None:
        # # C++ parity: basecorrelationstructure.hpp:142-157. Public in C++
        # and never called by the constructor; see the class note.
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

    def update_matrix(self) -> None:
        """Re-read every quote into the correlation matrix.

        # C++ parity: basecorrelationstructure.hpp:192-198 —
        # ``correlations_[i][j] = correlHandles_[i][j]->value()`` with
        # i = tenor index and j = loss index. Copied index-for-index; the
        # transposition against the interpolator (DEFECT 2) happens at
        # lookup, not here.
        """
        for i in range(len(self._corr_quotes)):
            for j in range(len(self._corr_quotes[0])):
                self._correlations[i, j] = self._corr_quotes[i][j].value()

    def update(self) -> None:
        """Refresh quote-driven matrix and forward to TermStructure observers."""
        # # C++ parity: basecorrelationstructure.hpp:186-190 — updateMatrix +
        # TermStructure::update.
        self.update_matrix()
        super().update()

    def correlation_size(self) -> int:
        # # C++ parity: basecorrelationstructure.hpp:93.
        return 1

    def max_date(self) -> Date:
        # # C++ parity: basecorrelationstructure.hpp:107.
        return self._tranche_dates[-1]

    def correlation(
        self, d: Date, loss_level: float, extrapolate: bool = False
    ) -> float:
        """Return the correlation at (date, loss-level) via interpolation.

        # C++ parity: basecorrelationstructure.hpp:108-110.
        """
        return self.correlation_at_time(
            self.time_from_reference(d), loss_level, extrapolate
        )

    def correlation_at_time(
        self, t: float, loss_level: float, extrapolate: bool = False
    ) -> float:
        """Same as ``correlation`` but skips the date->time conversion.

        # C++ parity: basecorrelationstructure.hpp:111-115. The
        # ``extrapolate`` argument is accepted and DISCARDED in C++ too —
        # the body passes a hard-coded ``true`` to the interpolator. Kept
        # for signature parity.
        """
        del extrapolate
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
