"""Statistical analysis of historical forward rates.

# C++ parity: ql/models/marketmodels/historicalforwardratesanalysis.hpp
# (v1.43) — the free ``historicalForwardRatesAnalysis<Traits, Interpolator>``
# template, the abstract ``HistoricalForwardRatesAnalysis`` interface and the
# concrete ``HistoricalForwardRatesAnalysisImpl<Traits, Interpolator>``.

Walks a historical date range, and on each date bootstraps a yield curve from
the supplied ibor + swap index fixings, then reads the time-to-go forward
rates at a fixed grid of fixing periods (``initial_gap`` to ``horizon``,
stepping by the forward index tenor). The relative day-over-day differences of
those forward rates are accumulated into a ``SequenceStatistics`` (so the
caller can extract the historical forward-rate covariance / correlation).

Divergences from C++:

- C++ sets ``Settings::instance().enforcesTodaysHistoricFixings() = true`` and
  guards it with a ``SavedSettings`` RAII restore. PQuantLib's
  ``ObservableSettings`` exposes neither the ``enforcesTodaysHistoricFixings``
  flag nor ``SavedSettings``; instead this function saves/restores
  ``evaluation_date`` manually (the only Settings field it actually mutates),
  and relies on ``InterestRateIndex.fixing(date, False)`` already reading the
  stored historic fixing for a past date.
- C++ re-drives one relinkable-quote-backed ``PiecewiseYieldCurve`` per
  iteration via ``SimpleQuote::setValue``. PQuantLib rebuilds a fresh
  ``PiecewiseYieldCurve`` (reference date = current date) each iteration.
  This is NOT cosmetic: ``termstructures/yield_/piecewise_yield_curve.py``
  documents that the port bootstraps once on first access and silently
  ignores later ``SimpleQuote`` mutations, so re-driving would read a stale
  curve. Rebuilding at ``reference_date = current_date`` reproduces the C++
  curve, whose ``settlementDays = 0`` reference date is
  ``cal.advance(currentDate, 0*Days)`` — and ``currentDate`` is always a
  business day, having been produced by a ``Following`` advance.
- C++ takes ``Traits`` and ``Interpolator`` as template parameters; Python is
  duck-typed, so both are ordinary arguments (``traits`` defaulting to
  ``Discount``, ``interpolator`` to the curve default). They are NOT
  cosmetic either — the bootstrap state variable and the interpolation
  together determine the between-pillar forward rates this analysis reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pquantlib.exceptions import LibraryException
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.deposit_rate_helper import DepositRateHelper
from pquantlib.termstructures.yield_.piecewise_yield_curve import PiecewiseYieldCurve
from pquantlib.termstructures.yield_.swap_rate_helper import SwapRateHelper
from pquantlib.termstructures.yield_.yield_traits import Discount
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.indexes.interest_rate_index import InterestRateIndex
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.math.statistics.sequence_statistics import SequenceStatistics
    from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period


def historical_forward_rates_analysis(  # noqa: PLR0915
    statistics: SequenceStatistics,
    skipped_dates: list[Date],
    skipped_dates_error_message: list[str],
    failed_dates: list[Date],
    failed_dates_error_message: list[str],
    fixing_periods: list[Period],
    start_date: Date,
    end_date: Date,
    step: Period,
    fwd_index: InterestRateIndex,
    initial_gap: Period,
    horizon: Period,
    ibor_indexes: list[IborIndex],
    swap_indexes: list[SwapIndex],
    yield_curve_day_counter: DayCounter,
    yield_curve_accuracy: float = 1.0e-12,
    interpolator: Any = None,
    traits: Any = Discount,
) -> None:
    """Accumulate relative day-over-day forward-rate moves into ``statistics``.

    # C++ parity: historicalforwardratesanalysis.hpp
    historicalForwardRatesAnalysis (the free template function).

    Mutates ``statistics`` and appends to the skipped/failed out-lists +
    ``fixing_periods``, mirroring the C++ out-parameter contract.
    ``interpolator`` selects the curve interpolation (C++ template param
    ``Interpolator``); ``None`` uses the curve default. ``traits`` selects the
    bootstrap state variable (C++ template param ``Traits``:
    ``Discount`` / ``ZeroYield`` / ``ForwardRate``).
    """
    skipped_dates.clear()
    skipped_dates_error_message.clear()
    failed_dates.clear()
    failed_dates_error_message.clear()
    fixing_periods.clear()

    settings = ObservableSettings()
    saved_evaluation_date = settings.evaluation_date

    try:
        # Build SimpleQuotes that drive the rate helpers' values.
        ibor_quotes = [SimpleQuote(0.0) for _ in ibor_indexes]
        swap_quotes = [SimpleQuote(0.0) for _ in swap_indexes]

        # Set up the forward-rates time grid (initial_gap .. horizon, by
        # the forward index tenor).
        index_tenor = fwd_index.tenor()
        fixing_period = initial_gap
        while fixing_period <= horizon:
            fixing_periods.append(fixing_period)
            fixing_period = fixing_period + index_tenor

        n_rates = len(fixing_periods)
        statistics.reset(n_rates)
        fwd_rates = [0.0] * n_rates
        prev_fwd_rates = [0.0] * n_rates
        index_day_counter = fwd_index.day_counter()
        cal = fwd_index.fixing_calendar()

        # start with a valid business date
        current_date = cal.advance(
            start_date, 1, TimeUnit.Days, BusinessDayConvention.Following
        )
        is_first = True
        while current_date <= end_date:
            # move the evaluation date to current_date
            settings.evaluation_date = current_date

            try:
                for i, ibor in enumerate(ibor_indexes):
                    ibor_quotes[i].set_value(ibor.fixing(current_date, False))
                for i, swap in enumerate(swap_indexes):
                    swap_quotes[i].set_value(swap.fixing(current_date, False))
            except (LibraryException, RuntimeError) as e:
                skipped_dates.append(current_date)
                skipped_dates_error_message.append(str(e))
                current_date = cal.advance_period(
                    current_date, step, BusinessDayConvention.Following
                )
                continue

            try:
                # Rebuild the bootstrapped yield curve at current_date.
                rate_helpers = _build_rate_helpers(
                    ibor_indexes, swap_indexes, ibor_quotes, swap_quotes, current_date
                )
                yc = PiecewiseYieldCurve(
                    traits,
                    current_date,
                    rate_helpers,
                    yield_curve_day_counter,
                    None,
                    interpolator,
                    yield_curve_accuracy,
                )
                for i in range(n_rates):
                    # Time-to-go forwards.
                    d = current_date + fixing_periods[i]
                    d2 = d + index_tenor
                    fwd_rates[i] = yc.forward_rate(
                        d,
                        d2,
                        Compounding.Simple,
                        Frequency.Annual,
                        False,
                        index_day_counter,
                    ).rate()
            except (LibraryException, RuntimeError) as e:
                failed_dates.append(current_date)
                failed_dates_error_message.append(str(e))
                current_date = cal.advance_period(
                    current_date, step, BusinessDayConvention.Following
                )
                continue

            # From 2nd step onwards, relative forward-rate differences.
            if not is_first:
                fwd_rates_diff = [fwd_rates[i] / prev_fwd_rates[i] - 1.0 for i in range(n_rates)]
                statistics.add(fwd_rates_diff)
            else:
                is_first = False

            # Store last calculated forward rates (swap).
            prev_fwd_rates, fwd_rates = fwd_rates, prev_fwd_rates
            current_date = cal.advance_period(
                current_date, step, BusinessDayConvention.Following
            )
    finally:
        # SavedSettings substitute: restore the evaluation date.
        settings.evaluation_date = saved_evaluation_date


def _build_rate_helpers(
    ibor_indexes: list[IborIndex],
    swap_indexes: list[SwapIndex],
    ibor_quotes: list[SimpleQuote],
    swap_quotes: list[SimpleQuote],
    evaluation_date: Date,
) -> list[BootstrapHelper[Any]]:
    """Build deposit + swap rate helpers from the indexes and live quotes.

    # C++ parity: the DepositRateHelper / SwapRateHelper construction loop in
    historicalForwardRatesAnalysis.
    """
    rate_helpers: list[BootstrapHelper[Any]] = []
    for ibor, quote in zip(ibor_indexes, ibor_quotes, strict=True):
        rate_helpers.append(
            DepositRateHelper(
                quote,
                ibor.tenor(),
                ibor.fixing_days(),
                ibor.fixing_calendar(),
                ibor.business_day_convention(),
                ibor.end_of_month(),
                ibor.day_counter(),
                evaluation_date=evaluation_date,
            )
        )
    for swap, quote in zip(swap_indexes, swap_quotes, strict=True):
        rate_helpers.append(
            SwapRateHelper(
                quote,
                swap_index=swap,
                evaluation_date=evaluation_date,
            )
        )
    return rate_helpers


class HistoricalForwardRatesAnalysis(ABC):
    """Read-only interface onto a completed historical forward-rate analysis.

    # C++ parity: historicalforwardratesanalysis.hpp
    HistoricalForwardRatesAnalysis (the pure-virtual base).

    The concrete driver is :class:`HistoricalForwardRatesAnalysisImpl`.
    """

    @abstractmethod
    def skipped_dates(self) -> list[Date]:
        """Dates whose index fixings could not be read."""

    @abstractmethod
    def skipped_dates_error_message(self) -> list[str]:
        """Error message for each skipped date."""

    @abstractmethod
    def failed_dates(self) -> list[Date]:
        """Dates whose curve bootstrap / forward read failed."""

    @abstractmethod
    def failed_dates_error_message(self) -> list[str]:
        """Error message for each failed date."""

    @abstractmethod
    def fixing_periods(self) -> list[Period]:
        """The forward-rate fixing-period grid."""


class HistoricalForwardRatesAnalysisImpl(HistoricalForwardRatesAnalysis):
    """Historical forward-rate analysis driver.

    # C++ parity: historicalforwardratesanalysis.hpp
    HistoricalForwardRatesAnalysisImpl<Traits, Interpolator>.

    Runs :func:`historical_forward_rates_analysis` at construction and exposes
    the populated statistics + skipped/failed-date diagnostics + the fixing
    periods grid.

    The C++ class template parameters ``Traits`` and ``Interpolator`` become
    the trailing ``traits`` / ``interpolator`` constructor arguments (the
    convention the free function above already established: the type is passed
    as a value, class or instance, because Python is duck-typed).
    """

    def __init__(
        self,
        stats: SequenceStatistics,
        start_date: Date,
        end_date: Date,
        step: Period,
        fwd_index: InterestRateIndex,
        initial_gap: Period,
        horizon: Period,
        ibor_indexes: list[IborIndex],
        swap_indexes: list[SwapIndex],
        yield_curve_day_counter: DayCounter,
        yield_curve_accuracy: float = 1.0e-12,
        interpolator: Any = None,
        traits: Any = Discount,
    ) -> None:
        self._stats = stats
        self._skipped_dates: list[Date] = []
        self._skipped_dates_error_message: list[str] = []
        self._failed_dates: list[Date] = []
        self._failed_dates_error_message: list[str] = []
        self._fixing_periods: list[Period] = []
        historical_forward_rates_analysis(
            self._stats,
            self._skipped_dates,
            self._skipped_dates_error_message,
            self._failed_dates,
            self._failed_dates_error_message,
            self._fixing_periods,
            start_date,
            end_date,
            step,
            fwd_index,
            initial_gap,
            horizon,
            ibor_indexes,
            swap_indexes,
            yield_curve_day_counter,
            yield_curve_accuracy,
            interpolator,
            traits,
        )

    def skipped_dates(self) -> list[Date]:
        """Dates whose index fixings could not be read.

        # C++ parity: HistoricalForwardRatesAnalysisImpl::skippedDates.
        """
        return self._skipped_dates

    def skipped_dates_error_message(self) -> list[str]:
        """Error message for each skipped date.

        # C++ parity: skippedDatesErrorMessage.
        """
        return self._skipped_dates_error_message

    def failed_dates(self) -> list[Date]:
        """Dates whose curve bootstrap / forward read failed.

        # C++ parity: HistoricalForwardRatesAnalysisImpl::failedDates.
        """
        return self._failed_dates

    def failed_dates_error_message(self) -> list[str]:
        """Error message for each failed date.

        # C++ parity: failedDatesErrorMessage.
        """
        return self._failed_dates_error_message

    def fixing_periods(self) -> list[Period]:
        """The forward-rate fixing-period grid.

        # C++ parity: HistoricalForwardRatesAnalysisImpl::fixingPeriods.
        """
        return self._fixing_periods

    def stats(self) -> SequenceStatistics:
        """The populated sequence statistics."""
        return self._stats
