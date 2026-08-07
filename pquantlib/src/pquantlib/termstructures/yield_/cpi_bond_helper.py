"""CPIBondHelper — bootstrap from an inflation-linked bond price quote.

# C++ parity: ql/termstructures/yield/bondhelpers.{hpp,cpp} class CPIBondHelper
# (v1.43).

C++ declares this class in ``bondhelpers.hpp`` next to ``BondHelper`` and
``FixedRateBondHelper``; PQuantLib keeps it in its own module purely so that
``bond_helper.py`` does not have to import the whole inflation stack at
module-load time.  It is the same class: a :class:`~pquantlib.termstructures.
yield_.bond_helper.BondHelper` whose wrapped bond is a
:class:`~pquantlib.instruments.bonds.cpi_bond.CPIBond`, so ``implied_quote``,
``set_term_structure`` and the date pair are all inherited unchanged.

Worth restating because it catches people out: the helper's dates are the
BOND's, not the schedule's (bondhelpers.cpp:35-38) —

    latest_date   = bond.cashflows()[-1].date()     # last CASHFLOW, so a
                                                    # rolled redemption lands
                                                    # after bond.maturity_date()
    earliest_date = bond.next_cash_flow_date()      # moves with the evaluation
                                                    # date, not with the issue

and they are captured once, at construction.  ``BondHelper`` is a plain
``RateHelper`` in C++, not a ``RelativeDateRateHelper``, so it does not
re-initialize when the evaluation date moves; rebuild the helper instead.

The coupons need the CPI index to carry a fixing history AND a zero-inflation
term structure — a missing fixing does not merely degrade the answer, it
raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib.termstructures.yield_.bond_helper import BondHelper, BondPriceType
from pquantlib.time.business_day_convention import BusinessDayConvention

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.inflation.cpi import InterpolationType
    from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
    from pquantlib.quotes.quote import Quote
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.date import Date
    from pquantlib.time.period import Period
    from pquantlib.time.schedule import Schedule


class CPIBondHelper(BondHelper):
    """CPI bond helper for curve bootstrap.

    # C++ parity: ``class CPIBondHelper : public BondHelper``
    # (bondhelpers.hpp:101-150, bondhelpers.cpp:110-165).

    Note the default ``payment_convention`` is ``Following`` here, matching
    ``CPIBondHelper``'s own default (bondhelpers.hpp:113) — which is NOT the
    ``ModifiedFollowing`` that :class:`~pquantlib.instruments.bonds.cpi_bond.CPIBond`
    itself defaults to.
    """

    def __init__(
        self,
        price: Quote | float,
        settlement_days: int,
        face_amount: float,
        base_cpi: float,
        observation_lag: Period,
        cpi_index: ZeroInflationIndex,
        observation_interpolation: InterpolationType,
        schedule: Schedule,
        fixed_rate: Sequence[float],
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        issue_date: Date | None = None,
        payment_calendar: Calendar | None = None,
        ex_coupon_period: Period | None = None,
        ex_coupon_calendar: Calendar | None = None,
        ex_coupon_convention: BusinessDayConvention = BusinessDayConvention.Unadjusted,
        ex_coupon_end_of_month: bool = False,
        price_type: BondPriceType = BondPriceType.Clean,
        growth_only: bool = False,
    ) -> None:
        # ``growth_only`` is the trailing keyword that
        # ``pquantlib.instruments.bonds.cpi_bond.CPIBond`` uses to spell C++'s
        # deprecated overload; C++ ``CPIBondHelper`` mirrors that pair of
        # constructors (bondhelpers.cpp:112-165) with the non-deprecated one
        # forwarding ``growthOnly = false``. Same default here.
        #
        # Local import: termstructures/ should not depend on instruments/ at
        # module-load time.
        from pquantlib.instruments.bonds.cpi_bond import CPIBond  # noqa: PLC0415

        super().__init__(
            price,
            CPIBond(
                settlement_days,
                face_amount,
                base_cpi,
                observation_lag,
                cpi_index,
                observation_interpolation,
                schedule,
                fixed_rate,
                accrual_day_counter,
                payment_convention,
                issue_date,
                payment_calendar,
                ex_coupon_period,
                ex_coupon_calendar,
                ex_coupon_convention,
                ex_coupon_end_of_month,
                growth_only,
            ),
            price_type,
        )


__all__ = ["CPIBondHelper"]
