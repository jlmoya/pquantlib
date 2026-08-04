"""ConstNotionalCrossCurrencySwap — generic constant-notional cross-currency swap.

# C++ parity: ql/instruments/constnotionalcrosscurrencyswap.{hpp,cpp} — new in v1.43.

Each leg carries its own currency; the notionals are constant and exchanged at
inception and at maturity. The first leg holds the pay-currency cashflows and
the second the receive-currency cashflows.

On top of ``Swap`` the class adds three per-leg result vectors, all filled by
:class:`~pquantlib.pricingengines.swap.discounting_const_notional_cross_currency_swap_engine.DiscountingConstNotionalCrossCurrencySwapEngine`:

* ``in_ccy_leg_npv`` / ``in_ccy_leg_bps`` — the leg's NPV / BPS *before* the FX
  conversion into the NPV currency (``leg_npv`` / ``leg_bps`` carry the
  converted figures),
* ``npv_date_discounts`` — the discount factor each leg's own curve assigns to
  the NPV date. Note it is a *vector* here, unlike ``Swap``'s scalar
  ``npv_date_discount``, because the two legs discount off different curves.

Python port notes:

- The C++ two-leg and multi-leg constructors become the ``from_legs_and_currencies``
  / ``from_multi_and_currencies`` classmethods, mirroring how ``Swap`` renders
  its own overloaded constructors. They cannot reuse ``Swap.from_legs`` /
  ``Swap.from_multi``: those names are already bound to different signatures.
- The C++ protected ``explicit ConstNotionalCrossCurrencySwap(Size legs)`` maps
  onto ``__init__(n_legs)``, matching ``Swap.__init__``; derived classes call it
  and then assign ``self._legs[j]`` / ``self._payer[j]`` / ``self._currencies[j]``
  exactly as the C++ derived constructors assign ``legs_[j]`` etc.
- ``Null<Real>()`` maps to ``None`` throughout, as elsewhere in PQuantLib.
"""

from __future__ import annotations

from collections.abc import Sequence

from pquantlib import qassert
from pquantlib.cashflows.rate_averaging import RateAveraging
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.currencies.currency import Currency
from pquantlib.instruments.swap import (
    Leg,
    LegInput,
    Swap,
    SwapArguments,
    SwapResults,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit


class ConstNotionalCrossCurrencySwapArguments(SwapArguments):
    """Arguments carrier — ``Swap::arguments`` plus the per-leg currencies.

    # C++ parity: ``ConstNotionalCrossCurrencySwap::arguments``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.currencies: list[Currency] = []

    def validate(self) -> None:
        super().validate()
        qassert.require(
            len(self.legs) == len(self.currencies),
            "number of legs is not equal to number of currencies",
        )


class ConstNotionalCrossCurrencySwapResults(SwapResults):
    """Results carrier — ``Swap::results`` plus the in-currency figures.

    # C++ parity: ``ConstNotionalCrossCurrencySwap::results``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_ccy_leg_npv: list[float] = []
        self.in_ccy_leg_bps: list[float] = []
        self.npv_date_discounts: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.in_ccy_leg_npv = []
        self.in_ccy_leg_bps = []
        self.npv_date_discounts = []


class ConstNotionalCrossCurrencySwapEngine(
    GenericEngine[
        ConstNotionalCrossCurrencySwapArguments, ConstNotionalCrossCurrencySwapResults
    ]
):
    """Base engine for cross-currency swaps.

    # C++ parity: ``ConstNotionalCrossCurrencySwap::engine``.
    """

    def __init__(self) -> None:
        super().__init__(
            ConstNotionalCrossCurrencySwapArguments(),
            ConstNotionalCrossCurrencySwapResults(),
        )


class ConstNotionalCrossCurrencySwap(Swap):
    """Generic constant-notional cross-currency swap."""

    def __init__(self, n_legs: int = 0) -> None:
        """Leg-count constructor, for derived classes that build their own legs.

        # C++ parity: ``explicit ConstNotionalCrossCurrencySwap(Size legs)``.
        """
        super().__init__(n_legs)
        self._currencies: list[Currency] = [Currency() for _ in range(n_legs)]
        self._in_ccy_leg_npv: list[float | None] = [None] * n_legs
        self._in_ccy_leg_bps: list[float | None] = [None] * n_legs
        self._npv_date_discounts: list[float | None] = [None] * n_legs

    @classmethod
    def from_legs_and_currencies(
        cls,
        first_leg: LegInput,
        first_leg_currency: Currency,
        second_leg: LegInput,
        second_leg_currency: Currency,
    ) -> ConstNotionalCrossCurrencySwap:
        """Two-leg shorthand: the first leg is paid, the second received.

        # C++ parity: ``ConstNotionalCrossCurrencySwap(const Leg&, const Currency&,
        # const Leg&, const Currency&)``.
        """
        s = cls(2)
        s._legs = [list(first_leg), list(second_leg)]
        s._payer = [-1.0, 1.0]
        s._currencies = [first_leg_currency, second_leg_currency]
        for leg in s._legs:
            for cf in leg:
                cf.register_with(s)
        return s

    @classmethod
    def from_multi_and_currencies(
        cls,
        legs: Sequence[LegInput],
        payer: Sequence[bool],
        currencies: Sequence[Currency],
    ) -> ConstNotionalCrossCurrencySwap:
        """Multi-leg constructor; ``payer[i]=True`` means leg ``i`` is paid.

        # C++ parity: ``ConstNotionalCrossCurrencySwap(const std::vector<Leg>&,
        # const std::vector<bool>&, const std::vector<Currency>&)``.
        """
        qassert.require(
            len(payer) == len(currencies),
            f"size mismatch between payer ({len(payer)}) "
            f"and currencies ({len(currencies)})",
        )
        qassert.require(
            len(legs) == len(payer),
            f"size mismatch between payer ({len(payer)}) and legs ({len(legs)})",
        )
        s = cls(len(legs))
        s._legs = [list(leg) for leg in legs]
        s._payer = [-1.0 if p else 1.0 for p in payer]
        s._currencies = list(currencies)
        for leg in s._legs:
            for cf in leg:
                cf.register_with(s)
        return s

    # --- Instrument / Swap interface ---------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """Copy legs + payer (from ``Swap``) plus the per-leg currencies.

        # C++ parity: ``ConstNotionalCrossCurrencySwap::setupArguments``.
        """
        super().setup_arguments(args)
        qassert.require(
            isinstance(args, ConstNotionalCrossCurrencySwapArguments),
            "the arguments are not of type cross currency swap",
        )
        assert isinstance(args, ConstNotionalCrossCurrencySwapArguments)
        args.currencies = list(self._currencies)

    def fetch_results(self, results: PricingEngineResults) -> None:
        """Pull the in-currency NPV / BPS and the NPV-date discounts.

        # C++ parity: ``ConstNotionalCrossCurrencySwap::fetchResults``.
        """
        super().fetch_results(results)
        qassert.require(
            isinstance(results, ConstNotionalCrossCurrencySwapResults),
            "the results are not of type cross currency swap",
        )
        assert isinstance(results, ConstNotionalCrossCurrencySwapResults)
        n = len(self._legs)

        if results.in_ccy_leg_npv:
            qassert.require(
                len(results.in_ccy_leg_npv) == n,
                "wrong number of in currency leg NPVs returned by engine",
            )
            self._in_ccy_leg_npv = list(results.in_ccy_leg_npv)
        else:
            self._in_ccy_leg_npv = [None] * n

        if results.in_ccy_leg_bps:
            qassert.require(
                len(results.in_ccy_leg_bps) == n,
                "wrong number of in currency leg BPSs returned by engine",
            )
            self._in_ccy_leg_bps = list(results.in_ccy_leg_bps)
        else:
            self._in_ccy_leg_bps = [None] * n

        if results.npv_date_discounts:
            qassert.require(
                len(results.npv_date_discounts) == n,
                "wrong number of npv date discounts returned by engine",
            )
            self._npv_date_discounts = list(results.npv_date_discounts)
        else:
            self._npv_date_discounts = [None] * n

    def setup_expired(self) -> None:
        """Zero the in-currency caches on top of ``Swap``'s expired setup.

        # C++ parity: ``ConstNotionalCrossCurrencySwap::setupExpired``.
        """
        super().setup_expired()
        n = len(self._legs)
        self._in_ccy_leg_bps = [0.0] * n
        self._in_ccy_leg_npv = [0.0] * n
        self._npv_date_discounts = [0.0] * n

    # --- inspectors --------------------------------------------------------

    def leg_currency(self, j: int) -> Currency:
        """Currency leg ``j``'s cashflows are denominated in."""
        qassert.require(j < len(self._legs), f"leg# {j} doesn't exist!")
        return self._currencies[j]

    def in_ccy_leg_bps(self, j: int) -> float:
        """BPS of leg ``j`` in that leg's own currency (before FX conversion)."""
        qassert.require(j < len(self._legs), f"leg# {j} doesn't exist!")
        self.calculate()
        qassert.require(self._in_ccy_leg_bps[j] is not None, "result not available")
        value = self._in_ccy_leg_bps[j]
        assert value is not None
        return value

    def in_ccy_leg_npv(self, j: int) -> float:
        """NPV of leg ``j`` in that leg's own currency (before FX conversion)."""
        qassert.require(j < len(self._legs), f"leg #{j} doesn't exist!")
        self.calculate()
        qassert.require(self._in_ccy_leg_npv[j] is not None, "result not available")
        value = self._in_ccy_leg_npv[j]
        assert value is not None
        return value

    def npv_date_discounts(self, j: int) -> float:
        """Discount factor leg ``j``'s own curve assigns to the NPV date."""
        qassert.require(j < len(self._legs), f"leg #{j} doesn't exist!")
        self.calculate()
        qassert.require(self._npv_date_discounts[j] is not None, "result not available")
        value = self._npv_date_discounts[j]
        assert value is not None
        return value

    # --- leg construction helper -------------------------------------------

    @staticmethod
    def add_notional_exchanges_to_leg(
        leg: Leg,
        calendar: Calendar,
        earliest_date: Date,
        maturity_date: Date,
        payment_lag: int,
        leg_bdc: BusinessDayConvention,
        nominal: float,
    ) -> None:
        """Prepend the initial and append the final notional exchange, in place.

        Mutates ``leg`` so that a cross-currency leg exchanges principal at both
        ends: ``-nominal`` on the earliest date and ``+nominal`` at maturity,
        both rolled by ``payment_lag`` business days under ``leg_bdc``.

        # C++ parity: ``ConstNotionalCrossCurrencySwap::addNotionalExchangesToLeg``.
        """
        initial_date = calendar.advance(
            earliest_date, payment_lag, TimeUnit.Days, leg_bdc
        )
        leg.insert(0, SimpleCashFlow(-nominal, initial_date))

        final_date = calendar.advance(maturity_date, payment_lag, TimeUnit.Days, leg_bdc)
        leg.append(SimpleCashFlow(nominal, final_date))

    @staticmethod
    def reject_unsupported_overnight_modifiers(
        side: str,
        *,
        compound_spread: bool,
        lookback_days: int | None,
        observation_shift: bool,
        lockout_days: int,
        averaging_method: RateAveraging,
        telescopic_value_dates: bool,
    ) -> None:
        """Raise unless every overnight modifier sits at its C++ default.

        No C++ counterpart — this is a port guard shared by the derived swaps.
        C++ forwards ``compoundSpreadDaily`` / ``lookbackDays`` /
        ``observationShift`` / ``lockoutDays`` / ``averagingMethod`` /
        ``telescopicValueDates`` to ``OvernightLeg``; PQuantLib's
        ``OvernightIndexedCoupon`` implements none of them — it compounds the
        plain daily fixing series and applies gearing/spread once, at coupon
        level. Pricing a leg as though the flag were off would silently answer
        a different question than the caller asked, so the unsupported
        combination is rejected outright.
        """
        unsupported: list[str] = []
        if compound_spread:
            unsupported.append(f"{side}_compound_spread=True")
        if lookback_days is not None:
            unsupported.append(f"{side}_lookback_days={lookback_days}")
        if observation_shift:
            unsupported.append(f"{side}_observation_shift=True")
        if lockout_days != 0:
            unsupported.append(f"{side}_lockout_days={lockout_days}")
        if averaging_method != RateAveraging.Compound:
            unsupported.append(f"{side}_averaging_method={averaging_method.name}")
        if telescopic_value_dates:
            unsupported.append("telescopic_value_dates=True")
        if unsupported:
            raise NotImplementedError(
                "PQuantLib's OvernightIndexedCoupon does not implement "
                + ", ".join(unsupported)
                + "; it compounds the plain daily fixing series and applies "
                "gearing/spread at coupon level. Leave these at their C++ "
                "defaults, or extend OvernightIndexedCoupon first."
            )


__all__ = [
    "ConstNotionalCrossCurrencySwap",
    "ConstNotionalCrossCurrencySwapArguments",
    "ConstNotionalCrossCurrencySwapEngine",
    "ConstNotionalCrossCurrencySwapResults",
]
