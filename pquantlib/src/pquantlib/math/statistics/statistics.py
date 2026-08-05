"""The default statistics tool.

# C++ parity: ql/math/statistics/statistics.hpp (v1.43) —
# ``typedef RiskStatistics Statistics;``

``Statistics`` is a pure alias, exactly as in C++: no behaviour of its own,
same type as :class:`~pquantlib.math.statistics.risk_statistics.RiskStatistics`,
so ``isinstance(x, Statistics)`` and ``isinstance(x, RiskStatistics)`` agree.
"""

from __future__ import annotations

from pquantlib.math.statistics.risk_statistics import RiskStatistics

# C++ parity: statistics.hpp:35.
Statistics = RiskStatistics

__all__ = ["Statistics"]
