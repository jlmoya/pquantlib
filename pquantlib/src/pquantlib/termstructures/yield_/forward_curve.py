"""ForwardCurve — type alias for ``InterpolatedForwardCurve`` (BackwardFlat).

# C++ parity: ql/termstructures/yield/forwardcurve.hpp (v1.42.1)
#   typedef InterpolatedForwardCurve<BackwardFlat> ForwardCurve;
"""

from __future__ import annotations

from pquantlib.termstructures.yield_.interpolated_forward_curve import InterpolatedForwardCurve

type ForwardCurve = InterpolatedForwardCurve
