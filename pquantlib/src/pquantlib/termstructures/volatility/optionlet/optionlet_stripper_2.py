"""OptionletStripper2 — strip ATM caplet vols from a CapFloorTermVolCurve.

# C++ parity: ql/termstructures/volatility/optionlet/optionletstripper2.{hpp,cpp}
# (v1.42.1).

Extends a pre-stripped :class:`OptionletStripper1` (which gives a
strike grid + per-fixing vols across that grid) by *augmenting* each
caplet row with one extra column — the **ATM strike** for each
option-expiry on the supplied :class:`CapFloorTermVolCurve` — populated
with the ATM caplet vol that, when overlaid as an additive spread on
the stripper1 surface, reproduces the cap NPV implied by the curve.

The Brent root-find solves, for each option-expiry j:

  cap_npv(stripper1.vols + spread_j) = cap_npv(atm_cap_vol_j)

PQuantLib divergences:

* **No Black engine with OptionletVolatilityStructure overload.**
  C++ piggy-backs on ``BlackCapFloorEngine(forwardingTermStructure,
  Handle<OptionletVolatilityStructure>)`` to reprice the cap with a
  spread-adjusted per-caplet vol surface. PQuantLib's
  :class:`BlackCapFloorEngine` only accepts a flat vol; we therefore
  hand-roll the spread-adjusted per-caplet repricing locally — for
  each caplet, we look up its stripper1 vol at the ATM strike (via
  :class:`StrippedOptionletAdapter`), add the trial spread, and
  re-evaluate the caplet via :func:`black_formula`. This matches the
  C++ ``SpreadedOptionletVolatility(adapter, spreadQuote)`` path
  semantically.
* **No MakeCapFloor.** We reuse OptionletStripper1's already-built
  per-tenor coupons (it caches fixing dates, payment dates, accrual
  periods, and ATM forwards on a per-cap basis). For OptionletStripper2
  we re-build minimal cap structures via the same ``_build_cap``
  helper, ATM-strike-pinned. We delegate per-caplet pricing to
  :func:`black_formula` directly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.optimize import brentq  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.cap_floor import Cap
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    black_formula,
)
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_curve import (
    CapFloorTermVolCurve,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_1 import (
    OptionletStripper1,
    _build_cap,  # pyright: ignore[reportPrivateUsage]
)
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_adapter import (
    StrippedOptionletAdapter,
)
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_base import (
    StrippedOptionletBase,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


# Brent search bracket for the implied vol spread (C++ uses
# ``minSpread = -0.1, maxSpread = 0.1``).
_MIN_SPREAD: float = -0.1
_MAX_SPREAD: float = 0.1
_INIT_GUESS: float = 0.0001


class OptionletStripper2(StrippedOptionletBase):
    """Augment OptionletStripper1 with ATM caplet vols from a term-vol curve."""

    def __init__(
        self,
        *,
        optionlet_stripper_1: OptionletStripper1,
        atm_cap_floor_term_vol_curve: CapFloorTermVolCurve,
        accuracy: float = 1.0e-5,
        max_iterations: int = 100,
    ) -> None:
        self._s1: OptionletStripper1 = optionlet_stripper_1
        self._curve: CapFloorTermVolCurve = atm_cap_floor_term_vol_curve
        self._accuracy: float = accuracy
        self._max_iterations: int = max_iterations

        # Day-counter parity check (C++ requires equal day counters
        # between the term-vol surface and the term-vol curve).
        qassert.require(
            self._s1.day_counter() == self._curve.day_counter(),
            "OptionletStripper1 and CapFloorTermVolCurve must share a day counter",
        )

        self._n_expiries: int = len(self._curve.option_tenors())

        # State populated by _perform_calculations.
        self._calculated: bool = False
        self._atm_strikes: list[float] = [0.0] * self._n_expiries
        self._atm_prices: list[float] = [0.0] * self._n_expiries
        self._spreads_vol: list[float] = [0.0] * self._n_expiries
        # Per-row augmented strikes + vols, mirroring stripper1's
        # ``optionlet_strikes(i)`` / ``optionlet_volatilities(i)``.
        self._augmented_strikes: list[list[float]] = []
        self._augmented_vols: list[list[float]] = []

    # --- public diagnostics ----------------------------------------------

    def atm_cap_floor_strikes(self) -> list[float]:
        self._ensure_calculated()
        return list(self._atm_strikes)

    def atm_cap_floor_prices(self) -> list[float]:
        self._ensure_calculated()
        return list(self._atm_prices)

    def spreads_vol(self) -> list[float]:
        self._ensure_calculated()
        return list(self._spreads_vol)

    # --- StrippedOptionletBase interface --------------------------------

    def optionlet_strikes(self, i: int) -> list[float]:
        self._ensure_calculated()
        return list(self._augmented_strikes[i])

    def optionlet_volatilities(self, i: int) -> list[float]:
        self._ensure_calculated()
        return list(self._augmented_vols[i])

    def optionlet_fixing_dates(self) -> list[Date]:
        return self._s1.optionlet_fixing_dates()

    def optionlet_fixing_times(self) -> list[float]:
        return self._s1.optionlet_fixing_times()

    def optionlet_maturities(self) -> int:
        return self._s1.optionlet_maturities()

    def atm_optionlet_rates(self) -> list[float]:
        return self._s1.atm_optionlet_rates()

    def day_counter(self) -> DayCounter:
        return self._s1.day_counter()

    def calendar(self) -> Calendar:
        return self._s1.calendar()

    def settlement_days(self) -> int:
        return self._s1.settlement_days()

    def business_day_convention(self) -> BusinessDayConvention:
        return self._s1.business_day_convention()

    def volatility_type(self) -> VolatilityType:
        return self._s1.volatility_type()

    def displacement(self) -> float:
        return self._s1.displacement()

    # --- internal -------------------------------------------------------

    def _ensure_calculated(self) -> None:
        if not self._calculated:
            self._perform_calculations()
            self._calculated = True

    def _perform_calculations(self) -> None:
        """Compute spreads + augment per-row strike grids.

        Stages:
          1. For each curve expiry j, build an ATM cap at the curve's
             ATM vol; compute the cap NPV (target).
          2. Brent-solve for the additive spread that, when overlaid
             on stripper1's per-caplet vols at the ATM strike,
             reproduces the cap NPV.
          3. Insert (atm_strike, atm_caplet_vol) into stripper1's
             per-row strike grid (sorted).
        """
        # Force stripper1 to do its work. We access stripper1's
        # private buffers because PQuantLib's StrippedOptionletBase
        # interface intentionally omits ``ibor_index``, the term-vol
        # surface handle, and the discount-curve helper — stripper2
        # is the *only* downstream consumer that needs them, so we
        # narrow the coupling here rather than widen the public
        # interface.
        self._s1._ensure_calculated()  # pyright: ignore[reportPrivateUsage]
        ibor_index = self._s1._ibor_index  # pyright: ignore[reportPrivateUsage]
        ref_date = self._s1._term_vol_surface.reference_date()  # pyright: ignore[reportPrivateUsage]
        discount = self._s1._discount_handle()  # pyright: ignore[reportPrivateUsage]
        adapter = StrippedOptionletAdapter(self._s1)
        adapter.enable_extrapolation(True)

        # ---- 1) ATM cap prices.
        for j, tenor in enumerate(self._curve.option_tenors()):
            atm_vol = self._curve.volatility(tenor, 33.3333, True)
            # Build a "1*tenor" cap on the index (parity with C++
            # MakeCapFloor(Cap, tenor, index, Null<Rate>(), 0*Days)).
            # The ATM strike comes from the cap's parRate — we
            # approximate it as the discount-weighted average of the
            # cap's coupon forwards.
            #
            # NOTE: ``Null<Rate>()`` in C++ triggers MakeCapFloor to
            # back out the ATM strike from ``cap_->atmRate(...)``;
            # PQuantLib doesn't port ``atmRate`` so we use a coupon-
            # weighted ATM proxy here. The proxy matches the C++ atm
            # rate to floating-point precision when the curve is flat.
            cap = _build_cap(
                length=tenor,
                index=ibor_index,
                strike=0.04,  # dummy; overwritten after we compute ATM.
                reference_date=ref_date,
            )
            atm_strike = self._cap_atm_strike(cap, discount)
            self._atm_strikes[j] = atm_strike

            # Rebuild the cap at the ATM strike and compute its NPV
            # at the curve's ATM vol.
            cap_atm = _build_cap(
                length=tenor,
                index=ibor_index,
                strike=atm_strike,
                reference_date=ref_date,
            )
            self._atm_prices[j] = self._price_cap_with_flat_vol(
                cap_atm, discount, atm_vol,
            )

        # ---- 2) Per-expiry Brent solve for the implied spread.
        for j, tenor in enumerate(self._curve.option_tenors()):
            cap_atm = _build_cap(
                length=tenor,
                index=ibor_index,
                strike=self._atm_strikes[j],
                reference_date=ref_date,
            )
            target_price = self._atm_prices[j]
            atm_strike_j = self._atm_strikes[j]

            def objective(
                spread: float,
                cap_ref: Cap = cap_atm,
                target: float = target_price,
                atm_k: float = atm_strike_j,
            ) -> float:
                return (
                    self._price_cap_with_adapter_plus_spread(
                        cap_ref, discount, adapter, atm_k, spread,
                    )
                    - target
                )

            try:
                root_raw: object = brentq(  # pyright: ignore[reportUnknownVariableType]
                    objective,
                    _MIN_SPREAD,
                    _MAX_SPREAD,
                    xtol=self._accuracy,
                    maxiter=self._max_iterations,
                )
                root = float(root_raw)  # pyright: ignore[reportArgumentType]
            except Exception as e:
                # Re-raise as LibraryException for caller visibility.
                from pquantlib.exceptions import LibraryException  # noqa: PLC0415

                raise LibraryException(
                    f"OptionletStripper2 Brent solve failed at expiry "
                    f"{tenor}: {e}",
                ) from e
            self._spreads_vol[j] = root

        # ---- 3) Augment per-row strike grids with (atm_strike, atm_vol).
        n_rows = self._s1.optionlet_maturities()
        # Take stripper1's per-row strike grid + vols (these are at
        # stripper1's input strike grid, not the curve's ATM strikes).
        self._augmented_strikes = [self._s1.optionlet_strikes(i) for i in range(n_rows)]
        self._augmented_vols = [self._s1.optionlet_volatilities(i) for i in range(n_rows)]
        # Per the C++ loop: for each curve expiry j, for each row i
        # within the cap's floating-leg length, insert the ATM
        # strike + (stripper1 vol at ATM + spread_j).
        cap_floor_length: list[int] = []
        for j in range(self._n_expiries):
            cap = _build_cap(
                length=self._curve.option_tenors()[j],
                index=ibor_index,
                strike=self._atm_strikes[j],
                reference_date=ref_date,
            )
            cap_floor_length.append(len(cap.floating_leg()))

        for j in range(self._n_expiries):
            length_j = cap_floor_length[j]
            atm_strike_j = self._atm_strikes[j]
            for i in range(min(length_j, n_rows)):
                # Read stripper1's vol-at-ATM via the strike-axis
                # interpolation in the adapter.
                opt_time = self._s1.optionlet_fixing_times()[i]
                unadjusted = adapter.volatility(opt_time, atm_strike_j, True)
                adjusted = unadjusted + self._spreads_vol[j]
                # Insert sorted into row i.
                strikes_i = self._augmented_strikes[i]
                vols_i = self._augmented_vols[i]
                # Find insert index via bisect.
                insert_at = 0
                while insert_at < len(strikes_i) and strikes_i[insert_at] < atm_strike_j:
                    insert_at += 1
                strikes_i.insert(insert_at, atm_strike_j)
                vols_i.insert(insert_at, adjusted)

    # --- pricing helpers ------------------------------------------------

    def _cap_atm_strike(self, cap: Cap, discount: YieldTermStructureProtocol) -> float:
        """Compute the par-coupon ATM strike of a Cap.

        The ATM strike is the discount-weighted forward — equivalently,
        the fixed leg rate that zeroes the cap's intrinsic.
        """
        legs = cap.floating_leg()
        num = 0.0
        den = 0.0
        for cf in legs:
            if not isinstance(cf, FloatingRateCoupon):
                continue
            df = discount.discount(cf.date())
            accrual = cf.accrual_period()
            fwd = cf.adjusted_fixing()
            num += df * accrual * fwd
            den += df * accrual
        return num / den if den > 0.0 else 0.0

    def _price_cap_with_flat_vol(
        self, cap: Cap, discount: YieldTermStructureProtocol, vol: float,
    ) -> float:
        """Price a cap by repricing each caplet via a flat Black vol.

        Mirrors the per-caplet loop in :class:`BlackCapFloorEngine` for
        ShiftedLognormal with displacement=0 / Normal vol type from
        stripper1.
        """
        ref = discount.reference_date()
        dc = self._s1.day_counter()
        vol_type = self._s1.volatility_type()
        displacement = self._s1.displacement()
        strike = cap.cap_rates()[0]
        cap_npv = 0.0
        for cf in cap.floating_leg():
            if not isinstance(cf, FloatingRateCoupon):
                continue
            payment_date = cf.date()
            if payment_date <= ref:
                continue
            df = discount.discount(payment_date)
            accrual = cf.accrual_period()
            fwd = cf.adjusted_fixing()
            fix_date = cf.fixing_date()
            t_fix = dc.year_fraction(ref, fix_date)
            sqrt_t = math.sqrt(max(t_fix, 0.0))
            std_dev = vol * sqrt_t
            discounted_accrual = df * accrual
            if vol_type == VolatilityType.ShiftedLognormal:
                value = black_formula(
                    OptionType.Call, strike, fwd, std_dev,
                    discounted_accrual, displacement,
                )
            else:
                value = bachelier_black_formula(
                    OptionType.Call, strike, fwd, std_dev, discounted_accrual,
                )
            cap_npv += value
        return cap_npv

    def _price_cap_with_adapter_plus_spread(
        self,
        cap: Cap,
        discount: YieldTermStructureProtocol,
        adapter: StrippedOptionletAdapter,
        atm_strike: float,
        spread: float,
    ) -> float:
        """Price a cap by repricing each caplet with the adapter vol + spread.

        Per-caplet vol = ``adapter.volatility(t_fix, atm_strike) + spread``.
        """
        ref = discount.reference_date()
        dc = self._s1.day_counter()
        vol_type = self._s1.volatility_type()
        displacement = self._s1.displacement()
        strike = cap.cap_rates()[0]
        cap_npv = 0.0
        for cf in cap.floating_leg():
            if not isinstance(cf, FloatingRateCoupon):
                continue
            payment_date = cf.date()
            if payment_date <= ref:
                continue
            df = discount.discount(payment_date)
            accrual = cf.accrual_period()
            fwd = cf.adjusted_fixing()
            fix_date = cf.fixing_date()
            t_fix = dc.year_fraction(ref, fix_date)
            sqrt_t = math.sqrt(max(t_fix, 0.0))
            # Look up the stripper1 vol at the ATM strike for this
            # fixing time and add the spread.
            base_vol = adapter.volatility(t_fix, atm_strike, True)
            vol = base_vol + spread
            std_dev = vol * sqrt_t
            discounted_accrual = df * accrual
            if vol_type == VolatilityType.ShiftedLognormal:
                value = black_formula(
                    OptionType.Call, strike, fwd, std_dev,
                    discounted_accrual, displacement,
                )
            else:
                value = bachelier_black_formula(
                    OptionType.Call, strike, fwd, std_dev, discounted_accrual,
                )
            cap_npv += value
        return cap_npv


# Reattach the private cap builder so consumers can fix coupon pricers
# at L10-A-test time without reaching into stripper_1's private module.
__all__ = ["OptionletStripper2"]
