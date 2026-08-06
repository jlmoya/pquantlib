"""Gaussian1dSmileSection — smile section backed by a Gaussian1d model.

# C++ parity: ql/termstructures/volatility/gaussian1dsmilesection.{hpp,cpp} (v1.43)

Built for a fixed expiry (``fixing_date``) and a swap index. The ATM rate and
annuity are computed at construction from the model; each call to
``_volatility_impl(strike)`` builds a Swaption at the given strike, prices it
with the supplied engine, normalizes by the annuity, and inverts via
Black-implied stdev.

This lived as a private ``_Gaussian1dSmileSection`` inside
``termstructures/volatility/swaption/gaussian1d_swaption_volatility.py``. C++
gives it its own public header, and it is a public class there; the module now
mirrors the C++ path and the name is public.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula_implied_std_dev
from pquantlib.termstructures.volatility.smile_section import SmileSection

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.indexes.swap_index import SwapIndex
    from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
    from pquantlib.pricingengines.pricing_engine import PricingEngine
    from pquantlib.time.date import Date


class Gaussian1dSmileSection(SmileSection):
    """Smile section backed by a Gaussian1d model + pricing engine.

    # C++ parity: ``class Gaussian1dSmileSection`` in
    # ql/termstructures/volatility/gaussian1dsmilesection.{hpp,cpp}.

    Built for a fixed expiry (``fixing_date``) and a swap index. The
    ATM rate and annuity are computed at construction from the model;
    each call to ``_volatility_impl(strike)`` builds a Swaption at the
    given strike, prices it with the user engine, normalizes by the
    annuity, and inverts via Black-implied stdev.

    The C++ class accepts both SwapIndex (-> swaption back-out) and
    IborIndex (-> cap/floor back-out) constructors; only the SwapIndex
    variant is ported.

    The IborIndex variant's blocker, re-checked against v1.43: it needs
    ``MakeCapFloor``, which PQuantLib does not have. It does NOT need
    ``Gaussian1dCapFloorEngine`` to be written — an earlier note listed both,
    but that engine is present at
    ``pquantlib/src/pquantlib/pricingengines/capfloor/gaussian1d_capfloor_engine.py``.
    So the remaining work is the ``MakeCapFloor`` factory, not the engine.
    """

    def __init__(
        self,
        fixing_date: Date,
        swap_index: SwapIndex,
        model: Gaussian1dModel,
        day_counter: DayCounter,
        engine: PricingEngine,
    ) -> None:
        # C++ parity: gaussian1dsmilesection.cpp:30-48.
        super().__init__(
            exercise_date=fixing_date,
            day_counter=day_counter,
            reference_date=model.term_structure.reference_date(),
        )
        self._fixing_date: Date = fixing_date
        self._swap_index: SwapIndex = swap_index
        self._model: Gaussian1dModel = model
        self._engine: PricingEngine = engine

        self._atm: float = model.swap_rate(
            fixing_date, swap_index.tenor(), None, 0.0, swap_index
        )
        self._annuity: float = model.swap_annuity(
            fixing_date, swap_index.tenor(), None, 0.0, swap_index
        )

    # --- SmileSection surface ------------------------------------------

    def min_strike(self) -> float:
        # C++ parity: gaussian1dsmilesection.hpp:58 — lognormal section.
        return 0.0

    def max_strike(self) -> float:
        # C++ parity: gaussian1dsmilesection.hpp:59.
        return float("inf")

    def atm_level(self) -> float:
        # C++ parity: gaussian1dsmilesection.cpp:73.
        return self._atm

    def option_price(
        self,
        strike: float,
        option_type: int = 1,
        discount: float = 1.0,
    ) -> float:
        """Normalized swaption price (NPV / annuity * discount).

        # C++ parity: gaussian1dsmilesection.cpp:75-95.

        ``option_type`` is the int-encoded ``OptionType`` (1 = Call /
        Payer, -1 = Put / Receiver). The integer signature matches the
        ``SmileSection`` base.
        """
        from typing import cast as type_cast  # noqa: PLC0415

        from pquantlib.exercise import EuropeanExercise  # noqa: PLC0415
        from pquantlib.instruments.fixed_vs_floating_swap import (  # noqa: PLC0415
            FixedVsFloatingSwap,
        )
        from pquantlib.instruments.swap import SwapType  # noqa: PLC0415
        from pquantlib.instruments.swaption import Swaption  # noqa: PLC0415

        # Build the underlying swap at the (potentially non-ATM) strike.
        # C++ uses ``MakeSwaption(swap_index, fixing, strike)``; PQuantLib
        # builds the swap via the swap index and then constructs the
        # Swaption explicitly.
        underlying = self._swap_index.underlying_swap(self._fixing_date)
        ot = OptionType(option_type)
        swap_type = SwapType.Payer if ot == OptionType.Call else SwapType.Receiver

        # Re-skin the underlying with the requested fixed rate. We
        # rebuild a VanillaSwap with the same schedule/conventions but
        # the new rate.
        new_underlying = type_cast(
            "FixedVsFloatingSwap",
            _rebuild_swap_at_strike(underlying, strike, swap_type),
        )

        exercise = EuropeanExercise(self._fixing_date)
        swp: Swaption = Swaption(new_underlying, exercise)
        swp.set_pricing_engine(self._engine)
        try:
            tmp = swp.npv()
            return tmp / self._annuity * discount
        except (LibraryException, ZeroDivisionError, ValueError):
            # C++ silently returns 0 on engine failure — match it.
            return 0.0

    def _volatility_impl(self, strike: float) -> float:
        """Black-implied vol — invert ``option_price(strike)`` via Newton-safe.

        # C++ parity: gaussian1dsmilesection.cpp:97-107.

        Returns 0.0 on any exception (Brenner-Subrahmanyan divergence,
        Newton failure, etc.) — matches the C++ catch-all.
        """
        try:
            option_type = OptionType.Call if strike >= self._atm else OptionType.Put
            opt_price = self.option_price(strike, int(option_type))
            if opt_price <= 0.0:
                return 0.0
            stdev = black_formula_implied_std_dev(
                option_type, strike, self._atm, opt_price
            )
            return stdev / (self.exercise_time() ** 0.5)
        except (LibraryException, ZeroDivisionError, ValueError):
            return 0.0


def _rebuild_swap_at_strike(
    underlying: object,
    strike: float,
    swap_type: object,
) -> object:
    """Clone an existing VanillaSwap-like instrument at a new fixed rate.

    # C++ parity: ``MakeSwaption(swap_index, fixing, strike)`` is the
    # C++ idiom — it builds a fresh swap with the requested rate from
    # the swap index. PQuantLib's MakeVanillaSwap is the equivalent but
    # requires more state (effective date, schedules). We take a
    # different approach: reflect the existing underlying schedule +
    # day count + leg notional, but replace the fixed rate.

    The runtime-untyped signature here mirrors the duck-typed style
    used in ``Gaussian1dModel.swap_rate`` and ``swap_annuity`` —
    ``VanillaSwap`` lives below the termstructures layer in the
    dependency graph; using ``object`` for the arguments lets pyright
    type the call site cleanly.
    """
    from typing import Any, cast  # noqa: PLC0415

    from pquantlib.instruments.vanilla_swap import VanillaSwap  # noqa: PLC0415
    u: Any = cast("Any", underlying)
    st: Any = cast("Any", swap_type)
    return VanillaSwap(
        swap_type=st,
        nominal=u.nominal(),
        fixed_schedule=u.fixed_schedule(),
        fixed_rate=strike,
        fixed_day_count=u.fixed_day_count(),
        float_schedule=u.floating_schedule(),
        ibor_index=u.ibor_index(),
        spread=u.spread(),
        floating_day_count=u.floating_day_count(),
        payment_convention=u.payment_convention(),
    )


__all__ = ["Gaussian1dSmileSection"]
