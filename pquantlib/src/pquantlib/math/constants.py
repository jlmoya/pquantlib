"""Math constants.

# C++ parity: ql/mathconstants.hpp + ql/types.hpp QL_EPSILON, QL_MAX_REAL (v1.42.1).

The C++ source ``ql/mathconstants.hpp`` defines M_PI, M_E, M_SQRT2, etc. as
preprocessor macros (or pulls them from <cmath>). The Python port surfaces
them as module-level ``Final[float]`` constants matching Python's ``math``
module values (which are double-precision IEEE 754 like C++ doubles).

QL_EPSILON in C++ is ``std::numeric_limits<double>::epsilon()`` ≈ 2.22e-16.
QL_MAX_REAL is ``std::numeric_limits<double>::max()`` ≈ 1.7977e308.
"""

from __future__ import annotations

import math
import sys
from typing import Final

# Mathematical constants — values identical to C++ <cmath> M_* macros.
M_PI: Final[float] = math.pi
M_E: Final[float] = math.e
M_SQRT2: Final[float] = math.sqrt(2.0)
M_SQRT1_2: Final[float] = 1.0 / math.sqrt(2.0)
M_LN2: Final[float] = math.log(2.0)
M_LN10: Final[float] = math.log(10.0)
M_LOG2E: Final[float] = 1.0 / math.log(2.0)
M_LOG10E: Final[float] = 1.0 / math.log(10.0)
M_1_PI: Final[float] = 1.0 / math.pi
M_2_PI: Final[float] = 2.0 / math.pi
M_2_SQRTPI: Final[float] = 2.0 / math.sqrt(math.pi)
M_PI_2: Final[float] = math.pi / 2.0
M_PI_4: Final[float] = math.pi / 4.0

# QL-specific machine-precision constants.
QL_EPSILON: Final[float] = sys.float_info.epsilon
QL_MAX_REAL: Final[float] = sys.float_info.max
QL_MIN_POSITIVE_REAL: Final[float] = sys.float_info.min
QL_MAX_INTEGER: Final[int] = sys.maxsize
