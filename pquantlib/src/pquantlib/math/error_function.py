"""Error function (erf).

# C++ parity: ql/math/errorfunction.hpp + ql/math/errorfunction.cpp (v1.42.1).

The C++ class implements its own Sun-Microsystems-derived approximation
of erf via piecewise polynomial fits. The Python port delegates to
``math.erf`` from stdlib (C99-equivalent, IEEE 754 double precision).

For most inputs the two agree at TIGHT tolerance (~1e-14); a few extreme-
range inputs may diverge by a few ULPs because the C++ approximation has
known accuracy bounds. Documented divergence: cross-validation runs at
TIGHT and inline LOOSE override is added per-test if necessary.
"""

from __future__ import annotations

import math


class ErrorFunction:
    """Function object: ``ErrorFunction()(x)`` returns erf(x)."""

    def __call__(self, x: float) -> float:
        return math.erf(x)
