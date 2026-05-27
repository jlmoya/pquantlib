"""DiscountCurve — type alias for ``InterpolatedDiscountCurve`` (LogLinear).

# C++ parity: ql/termstructures/yield/discountcurve.hpp (v1.42.1)
#   typedef InterpolatedDiscountCurve<LogLinear> DiscountCurve;
"""

from __future__ import annotations

from pquantlib.termstructures.yield_.interpolated_discount_curve import InterpolatedDiscountCurve

type DiscountCurve = InterpolatedDiscountCurve
