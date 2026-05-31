"""Catastrophe-risk event models for cat bonds.

# C++ parity: ql/experimental/catbonds/catrisk.{hpp,cpp} (v1.42.1).

A ``CatRisk`` is a factory that produces a ``CatSimulation`` over a
[start, end] window.  Each simulation's ``next_path`` yields one scenario
of (event_date, loss) pairs per call, returning ``False`` when no further
scenarios are available.

Two concrete risks:

- ``EventSet`` — replays a historical event catalogue, tiling consecutive
  windows of the requested length across the catalogue period
  (deterministic; this is the cross-validated path).
- ``BetaRisk`` — synthesises Poisson-arrival events with Beta-distributed
  severities (Monte Carlo; statistical-property validation only).

# C++ parity divergence — RNG:
# The C++ ``BetaRiskSimulation`` draws from ``std::mt19937`` +
# ``std::exponential_distribution`` + ``std::gamma_distribution``.  The
# Python port uses ``numpy.random.Generator`` (PCG64) with the equivalent
# exponential / gamma draws and the same Beta = G(a)/(G(a)+G(b)) ratio
# construction.  Exact draw-by-draw reproduction of the C++ stream is not
# possible across RNG engines; BetaRisk is therefore validated only on its
# distributional moments, while the deterministic EventSet path is
# cross-validated against C++ exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

# An (event_date, loss) pair.
EventPath = list[tuple[Date, float]]


def _years(n: int) -> Period:
    return Period(n, TimeUnit.Years)


class CatSimulation(ABC):
    """Per-window scenario generator.

    # C++ parity: ``class CatSimulation`` (catrisk.hpp:35-47).
    """

    def __init__(self, start: Date, end: Date) -> None:
        self._start: Date = start
        self._end: Date = end

    @abstractmethod
    def next_path(self, path: EventPath) -> bool:
        """Fill ``path`` with the next scenario; return False when exhausted.

        # C++ parity: ``CatSimulation::nextPath`` (pure virtual).

        ``path`` is cleared in place and repopulated (mirrors the C++
        ``std::vector`` out-parameter).
        """


class CatRisk(ABC):
    """Factory for cat simulations.

    # C++ parity: ``class CatRisk`` (catrisk.hpp:49-53).
    """

    @abstractmethod
    def new_simulation(self, start: Date, end: Date) -> CatSimulation:
        """Create a simulation over [start, end].

        # C++ parity: ``CatRisk::newSimulation`` (pure virtual).
        """


class EventSetSimulation(CatSimulation):
    """Replays a historical event catalogue window-by-window.

    # C++ parity: ``class EventSetSimulation`` (catrisk.{hpp,cpp}).
    """

    def __init__(
        self,
        events: Sequence[tuple[Date, float]],
        events_start: Date,
        events_end: Date,
        start: Date,
        end: Date,
    ) -> None:
        super().__init__(start, end)
        self._events: list[tuple[Date, float]] = list(events)
        self._events_start: Date = events_start
        self._events_end: Date = events_end

        self._years: int = end.year() - start.year()
        # Anchor the first replay period onto the catalogue's start year.
        if events_start.month() < start.month() or (
            events_start.month() == start.month()
            and events_start.day_of_month() <= start.day_of_month()
        ):
            self._period_start: Date = Date.from_ymd(
                start.day_of_month(), start.month(), events_start.year()
            )
        else:
            self._period_start = Date.from_ymd(
                start.day_of_month(), start.month(), events_start.year() + 1
            )
        self._period_end: Date = Date.from_ymd(
            end.day_of_month(), end.month(), self._period_start.year() + self._years
        )
        self._i: int = 0
        # Advance i to the first event at/after period_start.
        while self._i < len(self._events) and self._events[self._i][0] < self._period_start:
            self._i += 1

    def next_path(self, path: EventPath) -> bool:
        # C++ parity: catrisk.cpp:48-69.
        path.clear()
        if self._period_end > self._events_end:  # ran out of event data
            return False

        while self._i < len(self._events) and self._events[self._i][0] < self._period_start:
            self._i += 1
        while self._i < len(self._events) and self._events[self._i][0] <= self._period_end:
            ev_date, ev_loss = self._events[self._i]
            shifted_date = ev_date + _years(self._start.year() - self._period_start.year())
            path.append((shifted_date, ev_loss))
            self._i += 1

        if self._start + _years(self._years) < self._end:
            self._period_start += _years(self._years + 1)
            self._period_end += _years(self._years + 1)
        else:
            self._period_start += _years(self._years)
            self._period_end += _years(self._years)
        return True


class EventSet(CatRisk):
    """Historical event catalogue replayed as cat risk.

    # C++ parity: ``class EventSet`` (catrisk.{hpp,cpp}).
    """

    def __init__(
        self,
        events: Sequence[tuple[Date, float]],
        events_start: Date,
        events_end: Date,
    ) -> None:
        self._events: list[tuple[Date, float]] = list(events)
        self._events_start: Date = events_start
        self._events_end: Date = events_end

    def new_simulation(self, start: Date, end: Date) -> CatSimulation:
        # C++ parity: catrisk.cpp:76-78.
        return EventSetSimulation(
            self._events, self._events_start, self._events_end, start, end
        )


class BetaRiskSimulation(CatSimulation):
    """Poisson arrivals with Beta-distributed severities.

    # C++ parity: ``class BetaRiskSimulation`` (catrisk.{hpp,cpp}).
    """

    def __init__(
        self,
        start: Date,
        end: Date,
        max_loss: float,
        lambda_: float,
        alpha: float,
        beta: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(start, end)
        self._max_loss: float = max_loss
        self._lambda: float = lambda_
        self._alpha: float = alpha
        self._beta: float = beta
        dc = ActualActual(ActualActualConvention.ISDA)
        self._day_count: int = dc.day_count(start, end)
        self._year_fraction: float = dc.year_fraction(start, end)
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    def _generate_beta(self) -> float:
        # C++ parity: catrisk.cpp:92-97 — X / (X+Y) scaled by maxLoss.
        x = self._rng.gamma(self._alpha, 1.0)
        y = self._rng.gamma(self._beta, 1.0)
        return x * self._max_loss / (x + y)

    def next_path(self, path: EventPath) -> bool:
        # C++ parity: catrisk.cpp:99-116.
        path.clear()
        event_fraction = self._rng.exponential(1.0 / self._lambda)
        while event_fraction <= self._year_fraction:
            days = round(event_fraction * self._day_count / self._year_fraction)
            event_date = self._start + int(days)
            if event_date <= self._end:
                path.append((event_date, self._generate_beta()))
            else:
                break
            event_fraction = self._rng.exponential(1.0 / self._lambda)
        return True


class BetaRisk(CatRisk):
    """Beta-severity Poisson cat risk.

    # C++ parity: ``class BetaRisk`` (catrisk.{hpp,cpp}).

    ``years`` is the mean inter-arrival time (so ``lambda = 1/years``);
    ``mean`` / ``std_dev`` are the target severity moments, mapped onto the
    Beta(alpha, beta) shape parameters via the method of moments.
    """

    def __init__(self, max_loss: float, years: float, mean: float, std_dev: float) -> None:
        self._max_loss: float = max_loss
        self._lambda: float = 1.0 / years
        # C++ parity: catrisk.cpp:118-130 (method-of-moments mapping).
        qassert.require(
            mean < max_loss,
            f"Mean {mean} of the loss distribution must be less than the maximum loss {max_loss}",
        )
        normalized_mean = mean / max_loss
        normalized_var = std_dev * std_dev / (max_loss * max_loss)
        qassert.require(
            normalized_var < normalized_mean * (1.0 - normalized_mean),
            f"Standard deviation of {std_dev} is impossible to achieve in gamma "
            f"distribution with mean {mean}",
        )
        nu = normalized_mean * (1.0 - normalized_mean) / normalized_var - 1.0
        self._alpha: float = normalized_mean * nu
        self._beta: float = (1.0 - normalized_mean) * nu

    def new_simulation(self, start: Date, end: Date) -> CatSimulation:
        # C++ parity: catrisk.cpp:132-134.
        return BetaRiskSimulation(
            start, end, self._max_loss, self._lambda, self._alpha, self._beta
        )

    # Inspectors (Python convenience — used by tests to verify moments).
    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def intensity(self) -> float:
        """Poisson intensity ``lambda = 1/years``."""
        return self._lambda


__all__ = [
    "BetaRisk",
    "BetaRiskSimulation",
    "CatRisk",
    "CatSimulation",
    "EventPath",
    "EventSet",
    "EventSetSimulation",
]
