"""OptionletStripper2 — strip ATM caplet vols from a CapFloorTermVolCurve.

# C++ parity: ql/termstructures/volatility/optionlet/optionletstripper2.{hpp,cpp}
# (v1.43).

Extends a pre-stripped :class:`OptionletStripper1` (which gives a
strike grid + per-fixing vols across that grid) by *augmenting* each
caplet row with one extra column — the **ATM strike** for each
option-expiry on the supplied :class:`CapFloorTermVolCurve` — populated
with the ATM caplet vol that, when overlaid as an additive spread on
the stripper1 surface, reproduces the cap NPV implied by the curve.

The Brent root-find solves, for each option-expiry j:

  cap_npv(stripper1.vols + spread_j) = cap_npv(atm_cap_vol_j)

Like C++, this class derives from
:class:`~pquantlib.termstructures.volatility.optionlet.optionlet_stripper.OptionletStripper`
and re-builds its base from stripper1's own surface / index / vol type /
displacement / optionlet frequency (optionletstripper2.cpp:39-44), so its
tenor grid is independently recomputed rather than delegated. Only
``_perform_calculations`` — which copies stripper1's date grid and then
augments the strike/vol rows — lives here.

PQuantLib divergences (pre-existing; each is a real behavioural gap against
v1.43, flagged for a separate ``align()`` commit, NOT fixed by the base-class
refactor because fixing them moves numbers):

* **Cap repricing is hand-rolled, not engine-driven.** C++ prices the trial
  cap with ``BlackCapFloorEngine(forwardingTermStructure,
  Handle<OptionletVolatilityStructure>(SpreadedOptionletVolatility(adapter,
  spreadQuote)))`` (optionletstripper2.cpp:170-179). Both
  :class:`SpreadedOptionletVolatility` and :class:`MakeCapFloor` ARE ported
  now, so the "not available" premise this code was written under no longer
  holds.
* **The root-find is ``scipy.optimize.brentq``, not QuantLib's Brent.**
  ``pquantlib.math.solvers1d.brent.Brent`` is ported and is what
  optionletstripper2.cpp:125-133 uses, with a *guess* and QuantLib's own
  ``xAccuracy`` termination — scipy's ``brentq`` ignores the guess and
  terminates differently, so the two roots agree only to ``accuracy``.
* **The ATM strike is a discount-weighted-forward proxy**
  (:meth:`_cap_atm_strike`), where C++ uses
  ``caps_[j]->atmRate(**iborIndex_->forwardingTermStructure())``
  (optionletstripper2.cpp:87-88); ``CashFlows.atm_rate`` is ported.
* **The augmentation row bound is ``i < legSize``**, where C++ is
  ``i <= caps_[j]->floatingLeg().size()`` (optionletstripper2.cpp:100) — an
  inclusive bound that gives one MORE augmented row per expiry. The probe
  pins C++'s row widths at
  ``v143/ts/optionletstripper.json → stripper2_euribor3m_flat18.row_widths``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.optimize import brentq  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.cashflows.floating_rate_coupon import FloatingRateCoupon
from pquantlib.instruments.cap_floor import Cap
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import (
    bachelier_black_formula,
    black_formula,
)
from pquantlib.termstructures.volatility.capfloor.cap_floor_term_vol_curve import (
    CapFloorTermVolCurve,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper import (
    OptionletStripper,
)
from pquantlib.termstructures.volatility.optionlet.optionlet_stripper_1 import (
    OptionletStripper1,
    _build_cap,  # pyright: ignore[reportPrivateUsage]
)
from pquantlib.termstructures.volatility.optionlet.stripped_optionlet_adapter import (
    StrippedOptionletAdapter,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


# # C++ parity: ``Volatility guess = 0.0001, minSpread = -0.1, maxSpread = 0.1;``
# (optionletstripper2.cpp:127). The bracket is honoured; ``_INIT_GUESS`` is
# NOT — ``scipy.optimize.brentq`` takes no starting guess. Recorded rather
# than deleted, because it is the concrete evidence for the "use the ported
# ``pquantlib.math.solvers1d.brent.Brent``" item in the module docstring.
_MIN_SPREAD: float = -0.1
_MAX_SPREAD: float = 0.1
_INIT_GUESS: float = 0.0001


class OptionletStripper2(OptionletStripper):
    """Augment OptionletStripper1 with ATM caplet vols from a term-vol curve."""

    def __init__(
        self,
        *,
        optionlet_stripper_1: OptionletStripper1,
        atm_cap_floor_term_vol_curve: CapFloorTermVolCurve,
        accuracy: float = 1.0e-5,
        max_iterations: int = 100,
    ) -> None:
        # # C++ parity: OptionletStripper2::OptionletStripper2
        # (optionletstripper2.cpp:36-55) — the base is re-built from
        # stripper1's own inputs, with an EMPTY discount handle, so the tenor
        # walk is recomputed rather than delegated.
        super().__init__(
            optionlet_stripper_1.term_vol_surface(),
            optionlet_stripper_1.ibor_index(),
            None,
            optionlet_stripper_1.volatility_type(),
            optionlet_stripper_1.displacement(),
            optionlet_stripper_1.optionlet_frequency(),
        )
        self._s1: OptionletStripper1 = optionlet_stripper_1
        self._curve: CapFloorTermVolCurve = atm_cap_floor_term_vol_curve
        self._accuracy: float = accuracy
        self._max_iterations: int = max_iterations
        # # C++ parity: ``dc_(stripper1_->termVolSurface()->dayCounter())``
        # (optionletstripper2.cpp:46).
        self._dc = optionlet_stripper_1.term_vol_surface().day_counter()

        # # C++ parity: optionletstripper2.cpp:53-54.
        qassert.require(
            self._dc == self._curve.day_counter(),
            "different day counters provided",
        )

        self._n_expiries: int = len(self._curve.option_tenors())

        # State populated by _perform_calculations.
        self._atm_strikes: list[float] = [0.0] * self._n_expiries
        self._atm_prices: list[float] = [0.0] * self._n_expiries
        self._spreads_vol: list[float] = [0.0] * self._n_expiries

    # --- public diagnostics ----------------------------------------------

    def atm_cap_floor_strikes(self) -> list[float]:
        """# C++ parity: optionletstripper2.cpp:143-146."""
        self._ensure_calculated()
        return list(self._atm_strikes)

    def atm_cap_floor_prices(self) -> list[float]:
        """# C++ parity: optionletstripper2.cpp:148-151."""
        self._ensure_calculated()
        return list(self._atm_prices)

    def spreads_vol(self) -> list[float]:
        """# C++ parity: optionletstripper2.cpp:138-141."""
        self._ensure_calculated()
        return list(self._spreads_vol)

    # --- internal -------------------------------------------------------
    #
    # The whole StrippedOptionletBase read interface is inherited from
    # OptionletStripper — # C++ parity: optionletstripper.cpp:87-175. C++
    # OptionletStripper2 overrides NONE of it; the values arrive by copying
    # stripper1's grid into the base's own vectors below.

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
        # # C++ parity: optionletstripper2.cpp:60-68 — copy stripper1's date
        # grid into THIS object's base vectors. Reading any of stripper1's
        # accessors forces its own calculate() first, exactly as in C++.
        self._optionlet_dates = self._s1.optionlet_fixing_dates()
        self._optionlet_payment_dates = self._s1.optionlet_payment_dates()
        self._optionlet_accrual_periods = self._s1.optionlet_accrual_periods()
        self._optionlet_times = self._s1.optionlet_fixing_times()
        self._atm_optionlet_rate = self._s1.atm_optionlet_rates()
        for i in range(len(self._optionlet_times)):
            self._optionlet_strikes[i] = self._s1.optionlet_strikes(i)
            self._optionlet_volatilities[i] = self._s1.optionlet_volatilities(i)

        # # C++ parity: ``iborIndex_`` / ``termVolSurface_`` are the base's
        # own members (identical objects to stripper1's, per the ctor).
        ibor_index = self.ibor_index()
        # # C++ parity: MakeCapFloor works off Settings::evaluationDate()
        # (makevanillaswap.cpp:67), not the surface's reference date.
        eval_date = ObservableSettings().evaluation_date
        discount = self._discount_handle()
        # # C++ parity: optionletstripper2.cpp:94-95.
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
                evaluation_date=eval_date,
            )
            atm_strike = self._cap_atm_strike(cap, discount)
            self._atm_strikes[j] = atm_strike

            # Rebuild the cap at the ATM strike and compute its NPV
            # at the curve's ATM vol.
            cap_atm = _build_cap(
                length=tenor,
                index=ibor_index,
                strike=atm_strike,
                evaluation_date=eval_date,
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
                evaluation_date=eval_date,
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
        # The rows were seeded from stripper1 above; C++ augments the base's
        # own ``optionletStrikes_`` / ``optionletVolatilities_`` in place
        # (optionletstripper2.cpp:98-120).
        n_rows = self.optionlet_maturities()
        cap_floor_length: list[int] = []
        for j in range(self._n_expiries):
            cap = _build_cap(
                length=self._curve.option_tenors()[j],
                index=ibor_index,
                strike=self._atm_strikes[j],
                evaluation_date=eval_date,
            )
            cap_floor_length.append(len(cap.floating_leg()))

        for j in range(self._n_expiries):
            length_j = cap_floor_length[j]
            atm_strike_j = self._atm_strikes[j]
            # DIVERGENCE (pre-existing): C++ is
            # ``if (i <= caps_[j]->floatingLeg().size())``
            # (optionletstripper2.cpp:100) — an INCLUSIVE bound, so C++
            # augments one more row per expiry than this ``i < length_j``.
            # C++'s row widths are pinned at
            # ``v143/ts/optionletstripper.json →
            # stripper2_euribor3m_flat18.row_widths``. Not fixed here: this
            # commit is a pure re-seating that must move no numbers.
            for i in range(min(length_j, n_rows)):
                # Read stripper1's vol-at-ATM via the strike-axis
                # interpolation in the adapter.
                opt_time = self._optionlet_times[i]
                unadjusted = adapter.volatility(opt_time, atm_strike_j, True)
                adjusted = unadjusted + self._spreads_vol[j]
                # Insert sorted into row i.
                # # C++ parity: ``std::lower_bound`` + ``insert``
                # (optionletstripper2.cpp:106-117).
                strikes_i = self._optionlet_strikes[i]
                vols_i = self._optionlet_volatilities[i]
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

        DIVERGENCE (pre-existing): C++ calls
        ``caps_[j]->atmRate(**iborIndex_->forwardingTermStructure())``
        (optionletstripper2.cpp:87-88), which routes to
        ``CashFlows::atmRate`` — ported at
        ``pquantlib/src/pquantlib/cashflows/cash_flows.py:980``. This proxy
        only coincides with it on a flat curve. C++'s values are pinned at
        ``v143/ts/optionletstripper.json →
        stripper2_euribor3m_flat18.atm_cap_floor_strikes``.
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
        dc = self._dc
        vol_type = self.volatility_type()
        displacement = self.displacement()
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
        dc = self._dc
        vol_type = self.volatility_type()
        displacement = self.displacement()
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


__all__ = ["OptionletStripper2"]
