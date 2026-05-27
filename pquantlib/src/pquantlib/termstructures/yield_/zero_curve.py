"""ZeroCurve — type alias for ``InterpolatedZeroCurve`` (with Linear).

# C++ parity: ql/termstructures/yield/zerocurve.hpp (v1.42.1)
#   typedef InterpolatedZeroCurve<Linear> ZeroCurve;

C++ specializes the template at compile time. Python's ``type`` statement
(PEP 695) gives a name to an existing type without runtime cost; the
default interpolator on ``InterpolatedZeroCurve`` is already
``LinearInterpolation`` so a re-export suffices.
"""

from __future__ import annotations

from pquantlib.termstructures.yield_.interpolated_zero_curve import InterpolatedZeroCurve

type ZeroCurve = InterpolatedZeroCurve
