"""Italian government bonds — CCTEU / BTP and the Rendistato machinery.

# C++ parity: ql/instruments/bonds/btp.{hpp,cpp} (v1.43).

* :class:`CCTEU` — *Certificato di Credito del Tesoro*, a Euribor6M-indexed
  floating-rate bond with a hard-wired set of conventions.
* :class:`BTP` — *Buono Poliennale del Tesoro*, the fixed-rate counterpart.
* :class:`RendistatoBasket` — a weighted basket of BTPs with clean-price quotes.
* :class:`RendistatoCalculator` — a :class:`LazyObject` mapping the basket onto
  the equivalent plain-vanilla swap (the "Rendistato" index).
* :class:`RendistatoEquivalentSwapLengthQuote` /
  :class:`RendistatoEquivalentSwapSpreadQuote` — ``Quote`` adapters over the
  calculator.

# C++ parity divergence — ``yield``:
# ``yield`` is a Python keyword, so ``BTP::yield(Real cleanPrice, ...)`` and
# ``RendistatoCalculator::yield()`` are ported as ``yield_()``. Note this is
# NOT the same method as the inherited :meth:`Bond.yield_rate` (the C++
# ``Bond::yield`` overload set, which ``BTP::yield`` *hides* by name in C++
# but which stays reachable in Python — Python has no name hiding).

# C++ parity divergence — ``Null<Real>()``:
# ``RendistatoCalculator::performCalculations`` breaks out of its swap loop as
# soon as a swap-bond duration exceeds the basket duration, leaving the tail of
# ``swapRates_`` / ``swapBondDurations_`` at ``Null<Real>()``. Python has no
# poison-value sentinel, so — as elsewhere in this port (see
# ``interest_rate.py``) — the tail is left as ``nan``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.duration import Duration
from pquantlib.daycounters.actual_360 import Actual360
from pquantlib.daycounters.actual_actual import ActualActual
from pquantlib.daycounters.actual_actual import Convention as ActualActualConvention
from pquantlib.indexes.ibor.euribor import Euribor
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.instruments.bonds.fixed_rate_bond import FixedRateBond
from pquantlib.instruments.bonds.floating_rate_bond import FloatingRateBond
from pquantlib.instruments.make_vanilla_swap import make_vanilla_swap
from pquantlib.interest_rate import InterestRate
from pquantlib.math.rounding import ClosestRounding
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.patterns.observer import Observable
from pquantlib.quotes.quote import Quote
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.instruments.vanilla_swap import VanillaSwap
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure

_NULL_DATE: Date = Date()

#: Python stand-in for the C++ ``Null<Real>()`` poison value — see the module
#: docstring. Only ever observed in the *unreached* tail of the swap vectors.
_NULL_REAL: float = math.nan

#: ``ClosestRounding(5)`` is applied to the accrued amount of both Italian
#: government bonds (btp.hpp:193-201): the Italian market quotes accruals to
#: five decimals.
_ACCRUAL_ROUNDING = ClosestRounding(5)


def _btp_schedule(maturity_date: Date, start_date: Date | None) -> Schedule:
    """The schedule shared by ``CCTEU`` and ``BTP``.

    # C++ parity: btp.cpp:39-44 / :60-65 / :74-79 — ``Schedule(startDate, maturityDate,
    # 6*Months, NullCalendar(), Unadjusted, Unadjusted,
    # DateGeneration::Backward, true)``. A null ``startDate`` is legal (the
    # Backward rule falls back to the evaluation date).
    """
    return Schedule.from_rule(
        start_date if start_date is not None else _NULL_DATE,
        maturity_date,
        Period(6, TimeUnit.Months),
        NullCalendar(),
        BusinessDayConvention.Unadjusted,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Backward,
        True,  # end_of_month
    )


def _inner_product(a: Sequence[float], b: Sequence[float]) -> float:
    """``std::inner_product(a.begin(), a.end(), b.begin(), Real(0.0))``.

    Accumulated left-to-right from 0.0 so the floating-point result is
    bit-identical to the C++ one (``math.fsum`` would not be).
    """
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    return total


class CCTEU(FloatingRateBond):
    """Italian CCTEU — a Euribor6M-indexed floating-rate bond.

    # C++ parity: ``class CCTEU : public FloatingRateBond`` (btp.hpp:42-56,
    # btp.cpp:33-52).
    """

    def __init__(
        self,
        maturity_date: Date,
        spread: float,
        fwd_curve: YieldTermStructureProtocol | None = None,
        start_date: Date | None = None,
        issue_date: Date | None = None,
    ) -> None:
        super().__init__(
            settlement_days=2,
            face_amount=100.0,
            schedule=_btp_schedule(maturity_date, start_date),
            ibor_index=Euribor.six_months(fwd_curve),
            accrual_day_counter=Actual360(),
            payment_convention=BusinessDayConvention.Following,
            # C++ reads the fixing days off a throwaway ``Euribor6M()``.
            fixing_days=Euribor.six_months().fixing_days(),
            gearings=[1.0],
            spreads=[spread],
            caps=None,
            floors=None,
            in_arrears=False,
            redemption=100.0,
            issue_date=issue_date,
        )

    def accrued_amount(self, d: Date | None = None) -> float:
        """Accrued amount, rounded to five decimals.

        # C++ parity: btp.hpp:193-196.
        """
        return _ACCRUAL_ROUNDING(FloatingRateBond.accrued_amount(self, d))


class BTP(FixedRateBond):
    """Italian BTP — a fixed-rate government bond.

    # C++ parity: ``class BTP : public FixedRateBond`` (btp.hpp:62-89,
    # btp.cpp:55-80).

    C++ declares two constructors, the second of which inserts a
    ``redemption`` argument for the legacy non-par-redemption BTPs (as of
    today only IT123456789012, redeeming 99.999 in May 2037). Python collapses
    them into one signature with ``redemption`` defaulting to par; the 4-argument
    C++ form is ``BTP(maturity, rate, start_date=..., issue_date=...)``.
    """

    def __init__(
        self,
        maturity_date: Date,
        fixed_rate: float,
        redemption: float = 100.0,
        start_date: Date | None = None,
        issue_date: Date | None = None,
    ) -> None:
        super().__init__(
            settlement_days=2,
            face_amount=100.0,
            schedule=_btp_schedule(maturity_date, start_date),
            coupons=[fixed_rate],
            accrual_day_counter=ActualActual(ActualActualConvention.ISMA),
            payment_convention=BusinessDayConvention.ModifiedFollowing,
            redemption=redemption,
            issue_date=issue_date,
            payment_calendar=TARGET(),
        )

    def accrued_amount(self, d: Date | None = None) -> float:
        """Accrued amount, rounded to five decimals.

        # C++ parity: btp.hpp:198-201.
        """
        return _ACCRUAL_ROUNDING(FixedRateBond.accrued_amount(self, d))

    def yield_(
        self,
        clean_price: float,
        settlement_date: Date | None = None,
        accuracy: float = 1.0e-8,
        max_evaluations: int = 100,
    ) -> float:
        """BTP yield for a given clean price, in BTP market convention.

        Actual/Actual (ISMA), Compounded, Annual; the bond's own settlement
        date is used when none is given.

        # C++ parity: ``BTP::yield`` (btp.cpp:82-89). Named ``yield_`` because
        # ``yield`` is a Python keyword — see the module docstring.
        """
        return self.yield_from_price(
            BondPrice(clean_price, BondPriceType.Clean),
            ActualActual(ActualActualConvention.ISMA),
            Compounding.Compounded,
            Frequency.Annual,
            settlement_date,
            accuracy,
            max_evaluations,
        )


class RendistatoBasket(Observable):
    """Outstanding-weighted basket of BTPs with clean-price quotes.

    # C++ parity: ``class RendistatoBasket : public Observer, public Observable``
    # (btp.hpp:92-116, btp.cpp:92-131). The basket observes its clean-price
    # quotes and re-broadcasts their notifications.
    """

    def __init__(
        self,
        btps: Sequence[BTP],
        outstandings: Sequence[float],
        clean_price_quotes: Sequence[Quote],
    ) -> None:
        super().__init__()
        self._btps: list[BTP] = list(btps)
        self._outstandings: list[float] = list(outstandings)
        self._quotes: list[Quote] = list(clean_price_quotes)

        qassert.require(len(self._btps) > 0, "empty RendistatoCalculator Basket")
        k = len(self._btps)

        qassert.require(
            len(self._outstandings) == k,
            f"mismatch between number of BTPs ({k}) and number of outstandings ({len(self._outstandings)})",
        )
        qassert.require(
            len(self._quotes) == k,
            f"mismatch between number of BTPs ({k}) and number of clean prices quotes ({len(self._quotes)})",
        )

        for i in range(k):
            qassert.require(
                self._outstandings[i] >= 0,
                f"negative outstanding for {i} bond, maturity {self._btps[i].maturity_date()}",
            )

        # C++ TODO (btp.cpp:117): filter out expired / zero-outstanding bonds.
        self._n: int = k

        self._outstanding: float = 0.0
        for i in range(self._n):
            self._outstanding += self._outstandings[i]

        self._weights: list[float] = []
        for i in range(self._n):
            self._weights.append(self._outstandings[i] / self._outstanding)
            self._quotes[i].register_with(self)

    # --- inspectors ------------------------------------------------------

    def size(self) -> int:
        return self._n

    def btps(self) -> list[BTP]:
        return list(self._btps)

    def clean_price_quotes(self) -> list[Quote]:
        return list(self._quotes)

    def outstandings(self) -> list[float]:
        return list(self._outstandings)

    def weights(self) -> list[float]:
        return list(self._weights)

    def outstanding(self) -> float:
        return self._outstanding

    # --- Observer interface ----------------------------------------------

    def update(self) -> None:
        """# C++ parity: btp.hpp:109 — ``void update() { notifyObservers(); }``."""
        self.notify_observers()


class RendistatoCalculator(LazyObject):
    """Maps a :class:`RendistatoBasket` onto its equivalent plain-vanilla swap.

    # C++ parity: ``class RendistatoCalculator : public LazyObject``
    # (btp.hpp:120-166, btp.cpp:134-221).

    Fifteen 1..15-year EUR vanilla swaps are built once (C++ TODO: generalise
    the number of swaps and their lengths). ``performCalculations`` then walks
    them until a swap-bond duration exceeds the basket duration; the previous
    swap is the "equivalent" one.
    """

    _N_SWAPS: int = 15

    def __init__(
        self,
        basket: RendistatoBasket,
        euribor_index: Euribor,
        discount_curve: YieldTermStructure,
    ) -> None:
        super().__init__()
        self._basket: RendistatoBasket = basket
        self._euribor_index: Euribor = euribor_index
        self._discount_curve: YieldTermStructure = discount_curve

        n = basket.size()
        self._yields: list[float] = [0.05] * n
        self._durations: list[float] = [0.0] * n
        self._duration: float = 0.0
        self._equivalent_swap_index: int = self._N_SWAPS - 1

        self._swap_lengths: list[float] = [float(i + 1) for i in range(self._N_SWAPS)]
        self._swap_bond_durations: list[float] = [_NULL_REAL] * self._N_SWAPS
        self._swap_bond_yields: list[float] = [0.05] * self._N_SWAPS
        self._swap_rates: list[float] = [_NULL_REAL] * self._N_SWAPS

        basket.register_with(self)
        euribor_index.register_with(self)
        discount_curve.register_with(self)

        dummy_rate = 0.05
        evaluation_date = ObservableSettings().evaluation_date_or_today()
        self._swaps: list[VanillaSwap] = [
            make_vanilla_swap(
                Period(int(length), TimeUnit.Years),
                euribor_index,
                dummy_rate,
                Period(1, TimeUnit.Days),
                discount_curve=discount_curve,
                evaluation_date=evaluation_date,
            )
            for length in self._swap_lengths
        ]

    # --- calculations ----------------------------------------------------

    def yield_(self) -> float:
        """Outstanding-weighted average basket yield.

        # C++ parity: btp.hpp:213-217. Named ``yield_`` — ``yield`` is a
        # Python keyword.
        """
        return _inner_product(self._basket.weights(), self.yields())

    def duration(self) -> float:
        """# C++ parity: btp.hpp:219-222."""
        self.calculate()
        return self._duration

    def yields(self) -> list[float]:
        """# C++ parity: btp.hpp:224-227."""
        self.calculate()
        return list(self._yields)

    def durations(self) -> list[float]:
        """# C++ parity: btp.hpp:229-232."""
        self.calculate()
        return list(self._durations)

    def swap_lengths(self) -> list[float]:
        """# C++ parity: btp.hpp:234-236 — no ``calculate()``, set in the ctor."""
        return list(self._swap_lengths)

    def swap_rates(self) -> list[float]:
        """# C++ parity: btp.hpp:238-241."""
        self.calculate()
        return list(self._swap_rates)

    def swap_yields(self) -> list[float]:
        """# C++ parity: btp.hpp:243-246."""
        self.calculate()
        return list(self._swap_bond_yields)

    def swap_durations(self) -> list[float]:
        """# C++ parity: btp.hpp:248-251."""
        self.calculate()
        return list(self._swap_bond_durations)

    # --- equivalent-swap proxy -------------------------------------------

    def equivalent_swap(self) -> VanillaSwap:
        """# C++ parity: btp.hpp:253-257."""
        self.calculate()
        return self._swaps[self._equivalent_swap_index]

    def equivalent_swap_rate(self) -> float:
        """# C++ parity: btp.hpp:259-262."""
        self.calculate()
        return self._swap_rates[self._equivalent_swap_index]

    def equivalent_swap_yield(self) -> float:
        """# C++ parity: btp.hpp:264-267."""
        self.calculate()
        return self._swap_bond_yields[self._equivalent_swap_index]

    def equivalent_swap_duration(self) -> float:
        """# C++ parity: btp.hpp:269-272."""
        self.calculate()
        return self._swap_bond_durations[self._equivalent_swap_index]

    def equivalent_swap_length(self) -> float:
        """# C++ parity: btp.hpp:274-277."""
        self.calculate()
        return self._swap_lengths[self._equivalent_swap_index]

    def equivalent_swap_spread(self) -> float:
        """# C++ parity: btp.hpp:279-281."""
        return self.yield_() - self.equivalent_swap_rate()

    # --- LazyObject interface --------------------------------------------

    def _perform_calculations(self) -> None:
        # C++ parity: btp.cpp:156-221.
        isma = ActualActual(ActualActualConvention.ISMA)
        btps = self._basket.btps()
        quotes = self._basket.clean_price_quotes()
        bond_settlement_date = btps[0].settlement_date()

        for i in range(self._basket.size()):
            # accuracy 1e-10, maxIterations 100, guess = previous yield.
            self._yields[i] = btps[i].yield_from_price(
                BondPrice(quotes[i].value(), BondPriceType.Clean),
                isma,
                Compounding.Compounded,
                Frequency.Annual,
                bond_settlement_date,
                1.0e-10,
                100,
                self._yields[i],
            )
            self._durations[i] = CashFlows.duration(
                btps[i].cashflows(),
                InterestRate(self._yields[i], isma, Compounding.Compounded, Frequency.Annual),
                Duration.Modified,
                False,  # include_settlement_date_flows
                bond_settlement_date,
            )

        self._duration = _inner_product(self._basket.weights(), self._durations)

        settl_days = 2
        fixed_day_count = self._swaps[0].fixed_day_count()
        self._equivalent_swap_index = self._N_SWAPS - 1

        for i in range(self._N_SWAPS):
            self._swap_rates[i] = self._swaps[i].fair_rate()
            swap_bond = FixedRateBond(
                settl_days,
                100.0,  # face amount
                self._swaps[i].fixed_schedule(),
                [self._swap_rates[i]],
                fixed_day_count,
                BusinessDayConvention.Following,
                100.0,  # redemption
            )
            # Clean price 100.0 == the floating leg NPV including end payment.
            self._swap_bond_yields[i] = swap_bond.yield_from_price(
                BondPrice(100.0, BondPriceType.Clean),
                isma,
                Compounding.Compounded,
                Frequency.Annual,
                bond_settlement_date,
                1.0e-10,
                100,
                self._swap_bond_yields[i],
            )
            self._swap_bond_durations[i] = CashFlows.duration(
                swap_bond.cashflows(),
                InterestRate(self._swap_bond_yields[i], isma, Compounding.Compounded, Frequency.Annual),
                Duration.Modified,
                False,  # include_settlement_date_flows
                bond_settlement_date,
            )
            # C++ unrolls i == 0 out of the loop precisely so that this test is
            # skipped for it (btp.cpp:179-196 vs the i>=1 loop at :197-220).
            if i > 0 and self._swap_bond_durations[i] > self._duration:
                self._equivalent_swap_index = i - 1
                break


class RendistatoEquivalentSwapLengthQuote(Quote):
    """``Quote`` adapter over ``RendistatoCalculator::equivalentSwapLength``.

    # C++ parity: btp.hpp:170-178 + btp.cpp:223-234.
    """

    def __init__(self, r: RendistatoCalculator) -> None:
        super().__init__()
        self._r: RendistatoCalculator = r

    def value(self) -> float:
        """# C++ parity: btp.hpp:283-285."""
        return self._r.equivalent_swap_length()

    def is_valid(self) -> bool:
        """# C++ parity: btp.cpp:227-234 — ``try { value(); } catch (...)``."""
        try:
            self.value()
        except Exception:
            return False
        return True


class RendistatoEquivalentSwapSpreadQuote(Quote):
    """``Quote`` adapter over ``RendistatoCalculator::equivalentSwapSpread``.

    # C++ parity: btp.hpp:181-189 + btp.cpp:236-247.
    """

    def __init__(self, r: RendistatoCalculator) -> None:
        super().__init__()
        self._r: RendistatoCalculator = r

    def value(self) -> float:
        """# C++ parity: btp.hpp:287-289."""
        return self._r.equivalent_swap_spread()

    def is_valid(self) -> bool:
        """# C++ parity: btp.cpp:240-247 — ``try { value(); } catch (...)``."""
        try:
            self.value()
        except Exception:
            return False
        return True


__all__ = [
    "BTP",
    "CCTEU",
    "RendistatoBasket",
    "RendistatoCalculator",
    "RendistatoEquivalentSwapLengthQuote",
    "RendistatoEquivalentSwapSpreadQuote",
]
