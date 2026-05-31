"""historical_rates_analysis — historical-rate covariance/correlation extraction.

# C++ parity: ql/models/marketmodels/historicalratesanalysis.{hpp,cpp}
# (v1.42.1).

Walks a historical date range (stepping by ``step``, snapping to business
days with ``Following``), reads each index's fixing on each date, forms the
relative day-over-day differences ``r_t / r_{t-1} - 1``, and accumulates them
into a ``SequenceStatistics`` (so the caller can extract the historical
covariance / correlation of the rate moves). Dates whose fixings can't be
read are skipped (recorded with their error message).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.exceptions import LibraryException
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.indexes.interest_rate_index import InterestRateIndex
    from pquantlib.math.statistics.sequence_statistics import SequenceStatistics
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period


def historical_rates_analysis(
    statistics: SequenceStatistics,
    skipped_dates: list[Date],
    skipped_dates_error_message: list[str],
    start_date: Date,
    end_date: Date,
    step: Period,
    indexes: list[InterestRateIndex],
) -> None:
    """Accumulate relative day-over-day rate moves into ``statistics``.

    # C++ parity: historicalratesanalysis.cpp historicalRatesAnalysis.

    Mutates ``statistics`` (reset to ``len(indexes)`` dimensions then fed the
    relative-difference observations) and appends to the ``skipped_dates`` /
    ``skipped_dates_error_message`` out-lists, mirroring the C++
    out-parameter contract.
    """
    skipped_dates.clear()
    skipped_dates_error_message.clear()

    n_rates = len(indexes)
    statistics.reset(n_rates)

    sample = [0.0] * n_rates
    prev_sample = [0.0] * n_rates

    cal = indexes[0].fixing_calendar()
    # start with a valid business date
    current_date = cal.advance(start_date, 1, TimeUnit.Days, BusinessDayConvention.Following)
    is_first = True
    while current_date <= end_date:
        try:
            for i in range(n_rates):
                sample[i] = indexes[i].fixing(current_date, False)
        except (LibraryException, RuntimeError) as e:
            skipped_dates.append(current_date)
            skipped_dates_error_message.append(str(e))
            current_date = cal.advance_period(
                current_date, step, BusinessDayConvention.Following
            )
            continue

        # From 2nd step onwards, calculate relative differences.
        if not is_first:
            sample_diff = [sample[i] / prev_sample[i] - 1.0 for i in range(n_rates)]
            statistics.add(sample_diff)
        else:
            is_first = False

        # Store last calculated rates (swap, as in C++ std::swap).
        prev_sample, sample = sample, prev_sample
        current_date = cal.advance_period(current_date, step, BusinessDayConvention.Following)


class HistoricalRatesAnalysis:
    """Historical rate analysis driver.

    # C++ parity: historicalratesanalysis.hpp/.cpp HistoricalRatesAnalysis.

    Runs :func:`historical_rates_analysis` at construction and exposes the
    populated statistics + skipped-date diagnostics.
    """

    def __init__(
        self,
        stats: SequenceStatistics,
        start_date: Date,
        end_date: Date,
        step: Period,
        indexes: list[InterestRateIndex],
    ) -> None:
        self._stats = stats
        self._skipped_dates: list[Date] = []
        self._skipped_dates_error_message: list[str] = []
        historical_rates_analysis(
            self._stats,
            self._skipped_dates,
            self._skipped_dates_error_message,
            start_date,
            end_date,
            step,
            indexes,
        )

    def skipped_dates(self) -> list[Date]:
        """Dates whose fixings could not be read.

        # C++ parity: HistoricalRatesAnalysis::skippedDates.
        """
        return self._skipped_dates

    def skipped_dates_error_message(self) -> list[str]:
        """Error message for each skipped date.

        # C++ parity: HistoricalRatesAnalysis::skippedDatesErrorMessage.
        """
        return self._skipped_dates_error_message

    def stats(self) -> SequenceStatistics:
        """The populated sequence statistics.

        # C++ parity: HistoricalRatesAnalysis::stats.
        """
        return self._stats
