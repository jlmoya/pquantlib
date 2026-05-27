"""DiscountSpreadedTermStructure — alias for ``InterpolatedSpreadDiscountCurve``.

# C++ parity: there is no class named ``DiscountSpreadedTermStructure`` in
# QuantLib v1.42.1. The L2-B cluster scope refers to that name; the C++
# class that "wraps a base curve and multiplies discount factors by a
# spread factor" is ``InterpolatedSpreadDiscountCurve`` (with the
# ``SpreadDiscountCurve = InterpolatedSpreadDiscountCurve<LogLinear>``
# typedef). This module re-exports both under the cluster-scope name
# for clients that prefer the symmetric ``*SpreadedTermStructure``
# naming with its siblings ``Forward*`` and ``Zero*``.

The spread is a sequence of (date, discount-factor) pairs interpolated
in log-linear space by default. This is *not* a Quote-driven spread:
unlike ForwardSpreadedTermStructure and ZeroSpreadedTermStructure,
C++ does not provide a Quote-handle scalar-spread variant for discount
factors. Documented divergence from the cluster-scope description
(which says "Quote-driven"): the C++ closest analogue is term-structure
of spreads, not a scalar Quote.
"""

from __future__ import annotations

from pquantlib.termstructures.yield_.interpolated_spread_discount_curve import (
    InterpolatedSpreadDiscountCurve,
)

type DiscountSpreadedTermStructure = InterpolatedSpreadDiscountCurve
type SpreadDiscountCurve = InterpolatedSpreadDiscountCurve
