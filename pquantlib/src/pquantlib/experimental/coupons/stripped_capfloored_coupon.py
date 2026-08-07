"""StrippedCappedFlooredCouponLeg — strip the optionality out of a whole leg.

# C++ parity: ql/experimental/coupons/strippedcapflooredcoupon.{hpp,cpp} (v1.43),
# class ``StrippedCappedFlooredCouponLeg`` (hpp:74-80, cpp:115-131).

Wraps an existing ``Leg``: every entry that is a
:class:`~pquantlib.cashflows.capped_floored_coupon.CappedFlooredCoupon` is
replaced by a
:class:`~pquantlib.cashflows.stripped_capped_floored_coupon.StrippedCappedFlooredCoupon`
(whose ``rate()`` is the embedded cap/floor value alone); every other entry is
passed through **as the same object**, not a copy.

The coupon class itself lives at
``pquantlib.cashflows.stripped_capped_floored_coupon`` — this module adds only
the leg wrapper, which is what the C++ header ships alongside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.cashflows.capped_floored_coupon import CappedFlooredCoupon
from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.stripped_capped_floored_coupon import StrippedCappedFlooredCoupon

if TYPE_CHECKING:
    from collections.abc import Sequence


class StrippedCappedFlooredCouponLeg:
    """Leg wrapper that strips every capped/floored coupon it contains.

    # C++ parity: strippedcapflooredcoupon.hpp:74-80 + .cpp:115-131.

    The C++ class is consumed through an implicit ``operator Leg()``; Python
    finishes it with :meth:`leg` (also available as ``__call__``)::

        stripped = StrippedCappedFlooredCouponLeg(underlying_leg).leg()
    """

    def __init__(self, underlying_leg: Sequence[CashFlow]) -> None:
        self._underlying_leg: list[CashFlow] = list(underlying_leg)

    def underlying_leg(self) -> list[CashFlow]:
        """The leg this wrapper was built from (a copy of the sequence)."""
        return list(self._underlying_leg)

    def __call__(self) -> list[CashFlow]:
        """Sugar for :meth:`leg` — the C++ ``operator Leg() const``."""
        return self.leg()

    def leg(self) -> list[CashFlow]:
        """# C++ parity: ``StrippedCappedFlooredCouponLeg::operator Leg()``.

        Non-``CappedFlooredCoupon`` entries are pushed back unchanged — the
        C++ loop pushes the same ``shared_ptr``, so identity is preserved and
        the two legs share those cash flows.
        """
        result: list[CashFlow] = []
        for cf in self._underlying_leg:
            if isinstance(cf, CappedFlooredCoupon):
                result.append(StrippedCappedFlooredCoupon(cf))
            else:
                result.append(cf)
        return result


__all__ = ["StrippedCappedFlooredCouponLeg"]
