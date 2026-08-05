"""Sequence statistics with an L2 discrepancy calculation.

# C++ parity: ql/math/statistics/discrepancystatistics.hpp +
#             ql/math/statistics/discrepancystatistics.cpp (v1.43).

The discrepancy is *not* recomputed from the point set on demand: ``add``
folds each new point into two running sums (``adiscr_``, ``cdiscr_``) and
``discrepancy()`` only assembles them. A batch recomputation would agree on
the final value and disagree on every intermediate one — and intermediate
values are exactly what the low-discrepancy test-suite reads, sampling the
discrepancy as the sequence grows. The incremental update below is
therefore transcribed literally, including the two separate inner loops
that differ only in which argument of ``max`` is the new point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.math.statistics.sequence_statistics import SequenceStatistics


class DiscrepancyStatistics(SequenceStatistics):
    """Sequence statistics plus the L2 discrepancy of the point set.

    # C++ parity: ``class DiscrepancyStatistics``
    # (discrepancystatistics.hpp:35-98).
    """

    _adiscr: float
    _bdiscr: float
    _cdiscr: float
    _ddiscr: float

    def __init__(self, dimension: int) -> None:
        """Build over ``dimension``-dimensional points.

        # C++ parity: discrepancystatistics.hpp:103-106 — the base
        # constructor already resets, and the body resets again.
        """
        super().__init__(dimension)
        self.reset(dimension)

    def reset(self, dimension: int = 0) -> None:
        """(Re-)initialize; ``dimension=0`` keeps the current one.

        # C++ parity: discrepancystatistics.hpp:108-120.
        """
        if dimension == 0:  # if no size given,
            dimension = self._dimension  # keep the current one
        qassert.require(dimension != 1, "dimension==1 not allowed")

        super().reset(dimension)

        self._adiscr = 0.0
        self._bdiscr = 1.0 / 2.0 ** (dimension - 1)
        self._cdiscr = 0.0
        self._ddiscr = 1.0 / 3.0**dimension

    def add(self, sample: Sequence[float], weight: float = 1.0) -> None:
        """Add a point and fold it into the running discrepancy sums.

        # C++ parity: discrepancystatistics.hpp:49-93.
        """
        super().add(sample, weight)

        n = self.samples()

        temp = 1.0
        for k in range(self._dimension):
            r_ik = sample[k]  # i=N
            temp *= 1.0 - r_ik * r_ik
        self._cdiscr += temp

        for m in range(n - 1):
            temp = 1.0
            for k in range(self._dimension):
                # running i=1..(N-1)
                r_ik = self._stats[k].data()[m][0]
                # fixed j=N
                r_jk = sample[k]
                temp *= 1.0 - max(r_ik, r_jk)
            self._adiscr += temp

            temp = 1.0
            for k in range(self._dimension):
                # fixed i=N
                r_ik = sample[k]
                # running j=1..(N-1)
                r_jk = self._stats[k].data()[m][0]
                temp *= 1.0 - max(r_ik, r_jk)
            self._adiscr += temp

        temp = 1.0
        for k in range(self._dimension):
            # fixed i=N, j=N
            r_ik = r_jk = sample[k]
            temp *= 1.0 - max(r_ik, r_jk)
        self._adiscr += temp

    def discrepancy(self) -> float:
        """The L2 discrepancy of the accumulated point set.

        # C++ parity: discrepancystatistics.cpp:24-52.
        """
        n = self.samples()
        return math.sqrt(self._adiscr / (n * n) - self._bdiscr / n * self._cdiscr + self._ddiscr)
