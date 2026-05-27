"""SwapIndex — swap-rate index referencing an IborIndex.

# C++ parity: ql/indexes/swapindex.hpp + .cpp (v1.42.1)

C++ ``SwapIndex`` stores both inspectors (fixed_leg_tenor, fixed_leg_convention,
fixed_leg_day_counter, ibor_index, discount handle) and a swap-construction
helper (``underlyingSwap`` → ``MakeVanillaSwap``). The construction side
depends on L3's ``VanillaSwap`` instrument and ``MakeVanillaSwap`` builder.

PQuantLib L2-C carve-outs (port in L3):
- ``underlying_swap(fixing_date)`` — defers to L3 ``MakeVanillaSwap``.
- ``forecast_fixing`` — full par-swap calculation needs ``VanillaSwap.fair_rate()``.

What's ported in L2-C:
- All inspectors (family_name / tenor / fixing_days / currency / calendar /
  day_counter / fixed_leg_tenor / fixed_leg_convention / fixed_leg_day_counter /
  ibor_index / discounting_term_structure / exogenous_discount).
- ``maturity_date(value_date)`` — by C++ falls through to ``underlyingSwap``;
  we approximate as ``calendar.advance(value_date, tenor)``.
- The base class plumbing so L3 can subclass and add the full swap engine.
"""

from __future__ import annotations

from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class SwapIndex(InterestRateIndex):
    """Swap-rate index: fixed-vs-IBOR par-swap quote.

    Concrete forecast / underlying-swap functionality is deferred to L3
    (depends on ``VanillaSwap`` + ``MakeVanillaSwap``).
    """

    def __init__(
        self,
        family_name: str,
        tenor: Period,
        fixing_days: int,
        currency: Currency,
        fixing_calendar: Calendar,
        fixed_leg_tenor: Period,
        fixed_leg_convention: BusinessDayConvention,
        fixed_leg_day_counter: DayCounter,
        ibor_index: IborIndex,
        discounting_term_structure: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            family_name,
            tenor,
            fixing_days,
            currency,
            fixing_calendar,
            fixed_leg_day_counter,
        )
        self._fixed_leg_tenor: Period = fixed_leg_tenor
        self._fixed_leg_convention: BusinessDayConvention = fixed_leg_convention
        self._fixed_leg_day_counter: DayCounter = fixed_leg_day_counter
        self._ibor_index: IborIndex = ibor_index
        self._discount: YieldTermStructureProtocol | None = discounting_term_structure
        self._exogenous_discount: bool = discounting_term_structure is not None

    # --- InterestRateIndex interface ------------------------------------------

    def maturity_date(self, value_date: Date) -> Date:
        """C++ defers to ``underlyingSwap(...)->maturityDate()``; we approximate.

        Since L2-C doesn't yet have ``VanillaSwap``, we return
        ``calendar.advance(value_date, tenor, fixed_leg_convention)``. The
        full swap-driven calculation lands when L3 adds ``VanillaSwap``.
        """
        return self._fixing_calendar.advance(
            value_date,
            self._tenor.length,
            self._tenor.units,
            self._fixed_leg_convention,
            False,
        )

    def forecast_fixing(self, fixing_date: Date) -> float:
        """Par-swap rate at ``fixing_date`` — drives the underlying vanilla swap.

        # C++ parity: ql/indexes/swapindex.cpp ``SwapIndex::forecastFixing``
        # calls ``underlyingSwap(fixingDate)->fairRate()``.
        """
        return self.underlying_swap(fixing_date).fair_rate()

    def underlying_swap(self, fixing_date: Date):  # type: ignore[no-untyped-def]
        """Build the underlying VanillaSwap for a given fixing date.

        # C++ parity: ql/indexes/swapindex.cpp ``SwapIndex::underlyingSwap`` —
        # delegates to ``MakeVanillaSwap`` with effective date = value_date(fixing_date).
        """
        # Local import to keep the layering clean — indexes/ depend only on
        # cashflows + termstructures; instruments/make_vanilla_swap pulls in
        # the L3 swap stack which would invert the dependency order on
        # module import.
        from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap  # noqa: PLC0415
        from pquantlib.time.time_unit import TimeUnit  # noqa: PLC0415

        # Spot date from the fixing date (fixing_days business days forward).
        eff_date = self._fixing_calendar.advance(
            fixing_date, self._fixing_days, TimeUnit.Days,
        )
        return make_vanilla_swap(
            swap_tenor=self._tenor,
            ibor_index=self._ibor_index,
            fixed_rate=None,
            effective_date=eff_date,
            fixed_leg_tenor=self._fixed_leg_tenor,
            fixed_leg_convention=self._fixed_leg_convention,
            fixed_leg_termination_convention=self._fixed_leg_convention,
            fixed_leg_day_count=self._fixed_leg_day_counter,
            discount_curve=self._discount,
        )

    # --- inspectors ------------------------------------------------------------

    def fixed_leg_tenor(self) -> Period:
        return self._fixed_leg_tenor

    def fixed_leg_convention(self) -> BusinessDayConvention:
        return self._fixed_leg_convention

    def fixed_leg_day_counter(self) -> DayCounter:
        return self._fixed_leg_day_counter

    def ibor_index(self) -> IborIndex:
        return self._ibor_index

    def forwarding_term_structure(self) -> YieldTermStructureProtocol | None:
        return self._ibor_index.forecast_term_structure()

    def discounting_term_structure(self) -> YieldTermStructureProtocol | None:
        return self._discount

    def exogenous_discount(self) -> bool:
        return self._exogenous_discount

    # --- mutators --------------------------------------------------------------

    def clone(
        self,
        forwarding: YieldTermStructureProtocol | None = None,
        discounting: YieldTermStructureProtocol | None = None,
    ) -> SwapIndex:
        """Mirror C++ ``SwapIndex::clone`` (curve-overload variant)."""
        return SwapIndex(
            self._family_name,
            self._tenor,
            self._fixing_days,
            self._currency,
            self._fixing_calendar,
            self._fixed_leg_tenor,
            self._fixed_leg_convention,
            self._fixed_leg_day_counter,
            self._ibor_index.clone(forwarding) if forwarding is not None else self._ibor_index,
            discounting if discounting is not None else self._discount,
        )
