"""GridModelLocalVolSurface — a calibratable local-vol grid.

# C++ parity: ql/termstructures/volatility/equityfx/gridmodellocalvolsurface.hpp +
#             gridmodellocalvolsurface.cpp (v1.43).

A :class:`FixedLocalVolSurface` whose every grid value is a free model
parameter, so an optimizer can calibrate the whole surface at once. The class
is simultaneously a ``LocalVolTermStructure`` (it answers ``local_vol``
queries) and a ``CalibratedModel`` (it exposes ``params`` / ``set_params`` /
``calibrate``); ``generate_arguments`` rebuilds the underlying fixed surface
from the current parameter vector after every optimizer step.

Parameter layout — the C++ ``std::transform`` writes the arguments into the
matrix through ``Matrix::begin()``, which walks ROW-major over a
``(n_strikes, n_times)`` matrix. So argument ``i * n_times + j`` is the local
vol at strike ``i``, time ``j``. Getting that transposed would still calibrate
(the optimizer does not care) but would produce a completely different
surface, so it is pinned by the probe.

Every argument starts at 1.0 under a ``PositiveConstraint``; the constraint is
what keeps the optimizer from proposing a negative volatility.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.optimization.constraint import PositiveConstraint
from pquantlib.models.model import CalibratedModel, Model
from pquantlib.models.parameter import ConstantParameter
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.termstructures.volatility.equity_fx.fixed_local_vol_surface import (
    FixedLocalVolSurface,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date


class GridModelLocalVolSurface(LocalVolTermStructure, CalibratedModel):
    """Parameterized local-vol surface, calibratable node by node."""

    # C++ parity: ``typedef FixedLocalVolSurface::Extrapolation Extrapolation``.
    Extrapolation = FixedLocalVolSurface.Extrapolation

    def __init__(
        self,
        *,
        reference_date: Date,
        dates: Sequence[Date],
        strikes: Sequence[Sequence[float]],
        day_counter: DayCounter,
        lower_extrapolation: FixedLocalVolSurface.Extrapolation = (
            FixedLocalVolSurface.Extrapolation.ConstantExtrapolation
        ),
        upper_extrapolation: FixedLocalVolSurface.Extrapolation = (
            FixedLocalVolSurface.Extrapolation.ConstantExtrapolation
        ),
    ) -> None:
        LocalVolTermStructure.__init__(
            self,
            reference_date=reference_date,
            calendar=NullCalendar(),
            day_counter=day_counter,
        )
        qassert.require(len(dates) > 0, "empty date grid")
        qassert.require(len(strikes) > 0, "empty strike grid")
        # CalibratedModel resets the observer set, so it must run before
        # anything registers with this object.
        CalibratedModel.__init__(self, len(dates) * len(strikes[0]))

        for i in range(1, len(strikes)):
            qassert.require(
                len(strikes[i]) == len(strikes[0]),
                "strike vectors must have the same dimension",
            )

        # C++ ``std::fill(arguments_.begin(), arguments_.end(),
        # ConstantParameter(1.0, PositiveConstraint()))`` copies the value into
        # every slot; Python needs a fresh object per slot or they would all
        # alias one parameter array.
        for i in range(len(self.arguments)):
            self.arguments[i] = ConstantParameter(1.0, PositiveConstraint())

        self._reference_date_value: Date = reference_date
        self._times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in dates
        ]
        self._strikes: list[list[float]] = [[float(k) for k in row] for row in strikes]
        self._day_counter_value: DayCounter = day_counter
        self._lower_extrapolation: FixedLocalVolSurface.Extrapolation = (
            lower_extrapolation
        )
        self._upper_extrapolation: FixedLocalVolSurface.Extrapolation = (
            upper_extrapolation
        )

        self._local_vol: FixedLocalVolSurface | None = None
        self.generate_arguments()

    # --- Observer ----------------------------------------------------------

    def update(self) -> None:
        # C++ parity: LocalVolTermStructure::update() then CalibratedModel::update().
        TermStructure.update(self)
        Model.update(self)

    # --- delegated accessors -----------------------------------------------

    @property
    def _surface(self) -> FixedLocalVolSurface:
        assert self._local_vol is not None
        return self._local_vol

    def max_date(self) -> Date:
        return self._surface.max_date()

    def max_time(self) -> float:
        return self._surface.max_time()

    def min_strike(self) -> float:
        return self._surface.min_strike()

    def max_strike(self) -> float:
        return self._surface.max_strike()

    # --- CalibratedModel hook ----------------------------------------------

    def generate_arguments(self) -> None:
        """Rebuild the fixed surface from the current parameter vector.

        # C++ parity: ``GridModelLocalVolSurface::generateArguments``.
        """
        n_strikes = len(self._strikes[0])
        n_times = len(self._times)
        values: npt.NDArray[np.float64] = np.asarray(
            [p(0.0) for p in self.arguments], dtype=np.float64
        )
        # Row-major, matching C++ Matrix::begin().
        local_vol_matrix = values.reshape(n_strikes, n_times)
        self._local_vol = FixedLocalVolSurface(
            reference_date=self._reference_date_value,
            times=self._times,
            strikes=self._strikes,
            local_vol_matrix=local_vol_matrix,
            day_counter=self._day_counter_value,
            lower_extrapolation=self._lower_extrapolation,
            upper_extrapolation=self._upper_extrapolation,
        )

    # --- local-vol impl ----------------------------------------------------

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        # C++ parity: ``localVol_->localVol(t, strike, true)`` — the inner
        # surface is always queried with extrapolation enabled.
        return self._surface.local_vol_at_time(t, underlying_level, extrapolate=True)


__all__ = ["GridModelLocalVolSurface"]
