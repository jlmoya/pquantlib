"""OptionletStripper — abstract intermediate holding the shared stripper state.

# C++ parity: ql/termstructures/volatility/optionlet/optionletstripper.{hpp,cpp}
# (v1.43).

``StrippedOptionletBase`` specialization sitting between that interface and
the two concrete strippers. It owns every shared member and implements the
whole read interface; subclasses supply only ``_perform_calculations``
(C++ ``LazyObject::performCalculations``).

Split of responsibility, exactly as in v1.43 — read
``optionletstripper.cpp`` before changing anything here:

* The **constructor** (optionletstripper.cpp:30-85) computes
  ``optionletTenors_``, ``capFloorLengths_``, ``nStrikes_`` and
  ``nOptionletTenors_`` by walking from ``indexTenor`` up to the term-vol
  surface's longest option tenor. It only *sizes* ``optionletDates_``,
  ``optionletTimes_``, ``optionletPaymentDates_``,
  ``optionletAccrualPeriods_``, ``atmOptionletRate_``,
  ``optionletStrikes_`` and ``optionletVolatilities_``.
* The **date grid** (fixing dates, payment dates, accrual periods, fixing
  times, ATM rates) is NOT built here. ``OptionletStripper1`` derives it from
  the last floating coupon of a per-tenor cap
  (optionletstripper1.cpp:61-84); ``OptionletStripper2`` copies it wholesale
  from its stripper1 (optionletstripper2.cpp:60-68).
* Every accessor that reads a mutable vector calls ``calculate()`` first
  (optionletstripper.cpp:87-137). ``optionletFixingTenors()`` and
  ``optionletMaturities()`` do not — they are constructor-time state.

The tenor walk is driven by ``optionletFrequency_`` when supplied, otherwise
by ``iborIndex_->tenor()`` (optionletstripper.cpp:58). ``optionletFrequency_``
is an ``ext::optional<Period>``: ``nullopt`` is NOT a zero ``Period``, so the
Python port models it as ``Period | None`` and never coerces one to the other.

Cross-validated by ``migration-harness/references/v143/ts/optionletstripper.json``
(probe ``migration-harness/cpp/probes/v143_ts_optionletstripper/probe.cpp``).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.indexes.overnight_index import OvernightIndex
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_surface import (
    CapFloorTermVolSurface,
)
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_base import (
    StrippedOptionletBase,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


def _period_le(a: Period, b: Period) -> bool:
    """C++-style ``Period::operator<=`` modulo normalized equality.

    # C++ parity: ``operator<=(Period, Period)`` is ``!(rhs < lhs)``
    # (ql/time/period.hpp), and ``operator==`` is ``!(lhs<rhs) && !(rhs<lhs)``
    # — so C++ treats ``60M`` and ``5Y`` as equal.

    PQuantLib's :class:`Period` is a ``@dataclass(frozen=True)``, so its
    ``__eq__`` compares ``(length, units)`` field-by-field and reports
    ``60M != 5Y``. ``Period.__le__`` is ``self < other or self == other``, so
    it inherits that gap. Normalizing both operands first restores the C++
    answer for the multiple-of-12 case the tenor walk actually produces.

    DIVERGENCE (pre-existing, not introduced here): the real fix belongs in
    ``pquantlib/src/pquantlib/time/period.py`` — ``Period.__eq__`` should be
    ``not (self < other) and not (other < self)``. Flagged for a separate
    ``align(time)`` commit; kept local so this refactor moves no numbers.
    """
    return a.normalized() < b.normalized() or a.normalized() == b.normalized()


class OptionletStripper(StrippedOptionletBase):
    """Abstract ``StrippedOptionletBase`` holding the shared stripper state."""

    def __init__(
        self,
        term_vol_surface: CapFloorTermVolSurface,
        ibor_index: IborIndex,
        discount_curve: YieldTermStructureProtocol | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        displacement: float = 0.0,
        optionlet_frequency: Period | None = None,
    ) -> None:
        # # C++ parity: OptionletStripper::OptionletStripper
        # (optionletstripper.cpp:30-85).
        self._term_vol_surface: CapFloorTermVolSurface = term_vol_surface
        self._ibor_index: IborIndex = ibor_index
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        self._volatility_type: VolatilityType = volatility_type
        self._displacement: float = displacement
        self._optionlet_frequency: Period | None = optionlet_frequency
        self._n_strikes: int = len(term_vol_surface.strikes())

        # # C++ parity: optionletstripper.cpp:43-46.
        if volatility_type == VolatilityType.Normal:
            qassert.require(
                displacement == 0.0,
                "non-null displacement is not allowed with Normal model",
            )

        # # C++ parity: optionletstripper.cpp:48-51 — an overnight index has a
        # 1-day tenor, so without an explicit frequency the walk below would
        # emit one optionlet per day.
        if isinstance(ibor_index, OvernightIndex):
            qassert.require(
                optionlet_frequency is not None,
                "an optionlet frequency is required when using an overnight index",
            )

        # # C++ parity: optionletstripper.cpp:58-73 — optionlet tenors and
        # cap/floor lengths. ``capFloorLengths_`` runs one tenor ahead of
        # ``optionletTenors_`` so each successive cap adds exactly one new
        # optionlet, and the walk stops at the surface's longest option tenor.
        index_tenor = (
            optionlet_frequency if optionlet_frequency is not None else ibor_index.tenor()
        )
        max_cap_floor_tenor = term_vol_surface.option_tenors()[-1]
        self._optionlet_tenors: list[Period] = [index_tenor]
        self._cap_lengths: list[Period] = [self._optionlet_tenors[-1] + index_tenor]
        qassert.require(
            _period_le(self._cap_lengths[-1], max_cap_floor_tenor),
            f"too short ({max_cap_floor_tenor}) capfloor term vol termVolSurface",
        )
        next_cap_floor_length = self._cap_lengths[-1] + index_tenor
        while _period_le(next_cap_floor_length, max_cap_floor_tenor):
            self._optionlet_tenors.append(self._cap_lengths[-1])
            self._cap_lengths.append(next_cap_floor_length)
            next_cap_floor_length = next_cap_floor_length + index_tenor
        self._n_option_tenors: int = len(self._optionlet_tenors)

        # # C++ parity: optionletstripper.cpp:75-84 — sized here, filled by
        # ``_perform_calculations``. ``optionletStrikes_`` starts as
        # ``nOptionletTenors_`` copies of the surface's strike vector; only
        # OptionletStripper2 ever mutates the rows apart.
        surface_strikes = list(term_vol_surface.strikes())
        self._optionlet_volatilities: list[list[float]] = [
            [0.0] * self._n_strikes for _ in range(self._n_option_tenors)
        ]
        self._optionlet_strikes: list[list[float]] = [
            list(surface_strikes) for _ in range(self._n_option_tenors)
        ]
        self._optionlet_dates: list[Date] = [Date()] * self._n_option_tenors
        self._optionlet_times: list[float] = [0.0] * self._n_option_tenors
        self._atm_optionlet_rate: list[float] = [0.0] * self._n_option_tenors
        self._optionlet_payment_dates: list[Date] = [Date()] * self._n_option_tenors
        self._optionlet_accrual_periods: list[float] = [0.0] * self._n_option_tenors

        self._calculated: bool = False

    # --- LazyObject ------------------------------------------------------

    @abstractmethod
    def _perform_calculations(self) -> None:
        """Fill the mutable grid. # C++ parity: ``LazyObject::performCalculations``."""

    def _ensure_calculated(self) -> None:
        """# C++ parity: ``LazyObject::calculate()``."""
        if not self._calculated:
            self._perform_calculations()
            self._calculated = True

    def _discount_handle(self) -> YieldTermStructureProtocol:
        """The discount curve, defaulting to the index's forwarding curve.

        # C++ parity: ``discount_.empty() ? iborIndex_->forwardingTermStructure()
        # : discount_`` (optionletstripper1.cpp:94-97). C++ keeps the choice at
        # each use site; the Python port hoists it here because both subclasses
        # make it identically.
        """
        if self._discount_curve is not None:
            return self._discount_curve
        ts = self._ibor_index.forecast_term_structure()
        qassert.require(
            ts is not None,
            "no discount curve and IBOR index has no forecasting curve",
        )
        assert ts is not None
        return ts

    # --- StrippedOptionletBase interface ---------------------------------

    def optionlet_strikes(self, i: int) -> list[float]:
        # # C++ parity: optionletstripper.cpp:87-94.
        self._ensure_calculated()
        qassert.require(
            i < len(self._optionlet_strikes),
            f"index ({i}) must be less than optionletStrikes size "
            f"({len(self._optionlet_strikes)})",
        )
        return list(self._optionlet_strikes[i])

    def optionlet_volatilities(self, i: int) -> list[float]:
        # # C++ parity: optionletstripper.cpp:96-104.
        self._ensure_calculated()
        qassert.require(
            i < len(self._optionlet_volatilities),
            f"index ({i}) must be less than optionletVolatilities size "
            f"({len(self._optionlet_volatilities)})",
        )
        return list(self._optionlet_volatilities[i])

    def optionlet_fixing_dates(self) -> list[Date]:
        # # C++ parity: optionletstripper.cpp:110-113.
        self._ensure_calculated()
        return list(self._optionlet_dates)

    def optionlet_fixing_times(self) -> list[float]:
        # # C++ parity: optionletstripper.cpp:115-118.
        self._ensure_calculated()
        return list(self._optionlet_times)

    def optionlet_maturities(self) -> int:
        # # C++ parity: optionletstripper.cpp:120-122 — constructor-time
        # state, so NO calculate() here.
        return len(self._optionlet_tenors)

    def atm_optionlet_rates(self) -> list[float]:
        # # C++ parity: optionletstripper.cpp:134-137.
        self._ensure_calculated()
        return list(self._atm_optionlet_rate)

    def day_counter(self) -> DayCounter:
        # # C++ parity: optionletstripper.cpp:140-142.
        return self._term_vol_surface.day_counter()

    def calendar(self) -> Calendar:
        # # C++ parity: optionletstripper.cpp:144-146.
        return self._term_vol_surface.calendar()

    def settlement_days(self) -> int:
        """Settlement days of the underlying term-vol surface.

        # C++ parity: optionletstripper.cpp:148-150 — a bare forward to
        # ``termVolSurface_->settlementDays()``.

        DIVERGENCE (pre-existing, preserved here so this refactor moves no
        numbers): C++ ``TermStructure::settlementDays()`` QL_REQUIREs
        ``settlementDays_ != Null<Natural>()`` (ql/termstructure.hpp:127-131),
        i.e. it THROWS for a fixed-reference-date surface, and the probe pins
        that (``settlement_days_throws`` is ``true`` for every fixed-reference
        scenario in ``v143/ts/optionletstripper.json``). Swallowing it and
        returning 0 lets ``StrippedOptionletAdapter`` — and therefore
        ``OptionletStripper2`` — be built on a fixed-reference surface, which
        C++ rejects outright (probe key
        ``stripper2_euribor3m_flat18.fixed_reference_date_surface_throws``).
        Flagged for a separate ``align(termstructures/volatility)`` commit; the
        existing OptionletStripper2 tests depend on the swallow.
        """
        try:
            return self._term_vol_surface.settlement_days()
        except Exception:
            return 0

    def business_day_convention(self) -> BusinessDayConvention:
        # # C++ parity: optionletstripper.cpp:152-154.
        return self._term_vol_surface.business_day_convention()

    def displacement(self) -> float:
        # # C++ parity: optionletstripper.cpp:165-167.
        return self._displacement

    def volatility_type(self) -> VolatilityType:
        # # C++ parity: optionletstripper.cpp:169-171.
        return self._volatility_type

    # --- OptionletStripper's own accessors -------------------------------

    def optionlet_fixing_tenors(self) -> list[Period]:
        # # C++ parity: optionletstripper.cpp:106-108 — constructor-time
        # state, so NO calculate() here.
        return list(self._optionlet_tenors)

    def optionlet_payment_dates(self) -> list[Date]:
        # # C++ parity: optionletstripper.cpp:124-127.
        self._ensure_calculated()
        return list(self._optionlet_payment_dates)

    def optionlet_accrual_periods(self) -> list[float]:
        # # C++ parity: optionletstripper.cpp:129-132.
        self._ensure_calculated()
        return list(self._optionlet_accrual_periods)

    def term_vol_surface(self) -> CapFloorTermVolSurface:
        # # C++ parity: optionletstripper.cpp:156-159.
        return self._term_vol_surface

    def ibor_index(self) -> IborIndex:
        # # C++ parity: optionletstripper.cpp:161-163.
        return self._ibor_index

    def optionlet_frequency(self) -> Period | None:
        """The explicit optionlet frequency, or ``None`` for the index tenor.

        # C++ parity: optionletstripper.cpp:173-175, returning
        # ``ext::optional<Period>``. ``None`` is the ``ext::nullopt`` branch
        # and is NOT interchangeable with ``Period(0, Days)``.
        """
        return self._optionlet_frequency

    def cap_floor_lengths(self) -> list[Period]:
        """The per-optionlet cap tenors used to build the stripping caps.

        # C++ parity: ``capFloorLengths_`` (optionletstripper.hpp:92) is a
        # protected member with no C++ accessor; exposed read-only here
        # because Python has no ``protected`` and the subclasses' tests need it.
        """
        return list(self._cap_lengths)
