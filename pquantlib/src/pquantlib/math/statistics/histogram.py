"""Histogram of a data set.

# C++ parity: ql/math/statistics/histogram.hpp +
#             ql/math/statistics/histogram.cpp (v1.43).

The caller supplies either an explicit bin count, an explicit list of break
points, or one of three bin-count rules (Sturges, Freedman-Diaconis,
Scott). The rules are transcribed exactly, including:

* the quantile used by the Freedman-Diaconis rule is Hyndman & Fan **type
  8** (median-unbiased), with its own two boundary short-circuits — not
  numpy's or scipy's default quantile;
* Scott's rule takes the variance from
  :class:`~pquantlib.math.statistics.incremental_statistics.IncrementalStatistics`,
  i.e. the unbiased ``N/(N-1)``-scaled one;
* the bin count is floored at 1 for the rule-driven constructors only;
* when explicit break points are given, ``bins`` is fixed at
  ``len(breaks) + 1`` *before* the near-duplicate break points are collapsed
  with ``close_enough``, so a de-duplicated input leaves permanently empty
  bins between the last break and the overflow bin. That is C++'s
  behaviour and it is reproduced, not corrected.

Binning is left-open: a sample lands in the first bin whose break point it
is *strictly* below, and everything at or above the last break lands in the
final bin.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from typing import Final

from pquantlib import qassert
from pquantlib.math.closeness import close_enough
from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics

# C++ parity: ql/utilities/null.hpp — ``Null<Size>()`` is
# ``std::numeric_limits<int>::max()``, the "bin count not yet decided"
# sentinel the algorithm constructors store.
_NULL_SIZE: Final[int] = 2147483647


def _quantile(samples: Sequence[float], prob: float) -> float:
    """Hyndman & Fan (1996) type-8 discontinuous quantile.

    # C++ parity: the anonymous-namespace ``quantile`` helper
    # (histogram.cpp:37-62). Median-unbiased regardless of the sample
    # distribution, which is why the FD rule uses it rather than a plain
    # order statistic.
    """
    nsample = len(samples)
    qassert.require(prob >= 0.0 and prob <= 1.0, "Probability has to be in [0,1].")
    qassert.require(nsample > 0, "The sample size has to be positive.")

    if nsample == 1:
        return samples[0]

    # two special cases: close to boundaries
    a = 1.0 / 3
    b = 2 * a / (nsample + a)
    if prob < b:
        return min(samples)
    if prob > 1 - b:
        return max(samples)

    # general situation: middle region and nsample >= 2
    index = math.floor((nsample + a) * prob + a)
    # C++ uses partial_sort_copy into a buffer of length index+1; the first
    # index+1 entries of the fully sorted sample are the same values.
    sorted_samples = sorted(samples)[: index + 1]

    # use "index & index+1"th elements to interpolate the quantile
    weight = nsample * prob + a - index
    return (1 - weight) * sorted_samples[index - 1] + weight * sorted_samples[index]


class Histogram:
    """Histogram of a data set.

    # C++ parity: ``class Histogram`` (histogram.hpp:38-89).

    Args:
        data: the samples; ``None`` default-constructs an empty histogram.
        breaks: either the number of break points (C++'s ``Size breaks``
            constructor, giving ``breaks + 1`` bins) or an explicit sequence
            of break points (C++'s iterator-pair constructor).
        algorithm: the bin-count rule (C++'s ``Algorithm`` constructor).

    Exactly one of ``breaks`` / ``algorithm`` must be given when ``data``
    is. C++ distinguishes the constructors by overload resolution; Python
    cannot, so the combination is checked.
    """

    class Algorithm(IntEnum):
        """Bin-count rule.

        # C++ parity: ``enum Algorithm { None, Sturges, FD, Scott }``
        # (histogram.hpp:40). ``None`` is a Python keyword, hence ``None_``.
        # ``Unset`` is the out-of-range ``Algorithm(-1)`` the C++ default
        # constructor stores (histogram.hpp:44).
        """

        Unset = -1
        None_ = 0
        Sturges = 1
        FD = 2
        Scott = 3

    __slots__ = ("_algorithm", "_bins", "_breaks", "_counts", "_data", "_frequency")

    def __init__(
        self,
        data: Sequence[float] | None = None,
        breaks: int | Sequence[float] | None = None,
        algorithm: Algorithm | None = None,
    ) -> None:
        self._data: list[float] = []
        self._bins: int = 0
        self._algorithm: Histogram.Algorithm = Histogram.Algorithm.None_
        self._breaks: list[float] = []
        self._counts: list[int] = []
        self._frequency: list[float] = []

        if data is None:
            # C++ parity: histogram.hpp:44 — ``Histogram() : algorithm_(Algorithm(-1))``.
            qassert.require(
                breaks is None and algorithm is None,
                "breaks/algorithm given without data",
            )
            self._algorithm = Histogram.Algorithm.Unset
            return

        qassert.require(
            (breaks is None) != (algorithm is None),
            "exactly one of breaks or algorithm must be given",
        )
        self._data = list(data)
        if algorithm is not None:
            # C++ parity: histogram.hpp:52-57.
            self._bins = _NULL_SIZE
            self._algorithm = algorithm
        elif isinstance(breaks, int):
            # C++ parity: histogram.hpp:46-50.
            self._bins = breaks + 1
        elif breaks is not None:
            # C++ parity: histogram.hpp:59-64.
            self._breaks = list(breaks)
            self._bins = len(self._breaks) + 1
        else:  # pragma: no cover - excluded by the check above
            qassert.fail("exactly one of breaks or algorithm must be given")
        self._calculate()

    # --- inspectors -----------------------------------------------------

    def bins(self) -> int:
        """Number of bins.

        # C++ parity: histogram.cpp:67-69.
        """
        return self._bins

    def breaks(self) -> list[float]:
        """The break points, sorted and de-duplicated.

        # C++ parity: histogram.cpp:71-73.
        """
        return self._breaks

    def algorithm(self) -> Algorithm:
        """The bin-count rule in force.

        # C++ parity: histogram.cpp:75-77.
        """
        return self._algorithm

    def empty(self) -> bool:
        """True when no bins were computed.

        # C++ parity: histogram.cpp:79-81.
        """
        return self._bins == 0

    # --- results --------------------------------------------------------

    def counts(self, i: int) -> int:
        """Number of samples in bin ``i``.

        # C++ parity: histogram.cpp:83-89.
        """
        return self._counts[i]

    def frequency(self, i: int) -> float:
        """Fraction of samples in bin ``i``.

        # C++ parity: histogram.cpp:91-97.
        """
        return self._frequency[i]

    # --- calculation ----------------------------------------------------

    def _calculate(self) -> None:
        """Compute breaks, counts and frequencies.

        # C++ parity: ``Histogram::calculate`` (histogram.cpp:99-178).
        """
        qassert.require(len(self._data) > 0, "no data given")

        data_min = min(self._data)
        data_max = max(self._data)

        # calculate number of bins if necessary
        if self._bins == _NULL_SIZE:
            if self._algorithm == Histogram.Algorithm.Sturges:
                self._bins = math.ceil(math.log(float(len(self._data))) / math.log(2.0) + 1)
            elif self._algorithm == Histogram.Algorithm.FD:
                r1 = _quantile(self._data, 0.25)
                r2 = _quantile(self._data, 0.75)
                h = 2.0 * (r2 - r1) * float(len(self._data)) ** (-1.0 / 3.0)
                self._bins = math.ceil((data_max - data_min) / h)
            elif self._algorithm == Histogram.Algorithm.Scott:
                summary = IncrementalStatistics()
                summary.add_sequence(self._data)
                variance = summary.variance()
                h = 3.5 * math.sqrt(variance) * float(len(self._data)) ** (-1.0 / 3.0)
                self._bins = math.ceil((data_max - data_min) / h)
            elif self._algorithm == Histogram.Algorithm.None_:
                qassert.fail("a bin-partition algorithm is required")
            else:
                qassert.fail("unknown bin-partition algorithm")
            self._bins = max(self._bins, 1)

        if not self._breaks:
            # set breaks if not provided; ensure they evenly span the data range
            h = (data_max - data_min) / self._bins
            self._breaks = [data_min + (i + 1) * h for i in range(self._bins - 1)]
        else:
            # or ensure they're sorted if given
            self._breaks.sort()
            unique: list[float] = []
            for b in self._breaks:
                if not unique or not close_enough(unique[-1], b):
                    unique.append(b)
            self._breaks = unique

        # finally, calculate counts and frequencies
        self._counts = [0] * self._bins

        for p in self._data:
            processed = False
            for i in range(len(self._breaks)):
                if p < self._breaks[i]:
                    self._counts[i] += 1
                    processed = True
                    break
            if not processed:
                self._counts[self._bins - 1] += 1

        total_counts = len(self._data)
        self._frequency = [float(self._counts[i]) / total_counts for i in range(self._bins)]
