"""RateAveraging enum.

# C++ parity: ql/cashflows/rateaveraging.hpp (v1.42.1).

C++ uses ``struct RateAveraging { enum Type { Simple, Compound }; };``.
The nested-enum-inside-struct idiom is Python-translated as a plain
``IntEnum`` named ``RateAveraging`` — callers write ``RateAveraging.Simple``
/ ``RateAveraging.Compound``, mirroring the ``Duration`` port.

The enum selects how interest accrues in a multi-fixing coupon: ``Simple``
applies each sub-rate to the principal alone and sums the results, while
``Compound`` applies it to the principal plus the accumulated unpaid
interest.

PQuantLib's ``OvernightIndexedCoupon`` implements ``Compound`` only; the
enum exists so instruments that take an averaging method in C++ (the
overnight-indexed swap, the v1.43 cross-currency swaps) can carry the
argument faithfully and reject ``Simple`` explicitly instead of silently
compounding.
"""

from __future__ import annotations

from enum import IntEnum


class RateAveraging(IntEnum):
    """Mirrors ``QuantLib::RateAveraging::Type`` (ql/cashflows/rateaveraging.hpp:35)."""

    Simple = 0
    Compound = 1


__all__ = ["RateAveraging"]
