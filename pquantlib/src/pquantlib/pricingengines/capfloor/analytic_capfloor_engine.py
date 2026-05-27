"""Analytic cap/floor engine under an affine short-rate model.

# C++ parity: ql/pricingengines/capfloor/analyticcapfloorengine.{hpp,cpp}
# (v1.42.1).

Prices each optionlet under any affine short-rate model that exposes
``discount(t)`` and ``discount_bond_option(option_type, strike,
maturity, bond_maturity)``. Per-coupon:

- If the fixing has already happened (fixing_time <= 0), the optionlet
  pays the intrinsic ``max(forward - strike, 0)`` (or floor analog) at
  the model-implied discount factor.
- Otherwise, the cap optionlet is a Put on a discount bond with strike
  ``1 / (1 + capRate * tau)`` (with face = ``1 + capRate * tau``);
  symmetric for the floor.

The 4-arg ``discount_bond_option`` is the AffineModel surface — it's
strictly weaker than the 5-arg variant the Jamshidian engine needs
(no separate value_time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pquantlib import qassert
from pquantlib.instruments.cap_floor import (
    CapFloorArguments,
    CapFloorResults,
    CapFloorType,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swaption.jamshidian_swaption_engine import (
    TermStructureConsistentModelLike,
)

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


@runtime_checkable
class AffineModelLike(Protocol):
    """Structural surface needed by AnalyticCapFloorEngine.

    # C++ parity: ``ql/models/model.hpp`` ``class AffineModel``.

    A subset of the full short-rate model surface — just
    ``discount(t)`` + 4-arg ``discount_bond_option`` (no per-instance
    state needed beyond model parameters).
    """

    def discount(self, t: float) -> float: ...

    def discount_bond_option(
        self,
        option_type: int,
        strike: float,
        maturity: float,
        bond_maturity: float,
    ) -> float:
        """4-arg discount-bond option price."""
        ...


class AnalyticCapFloorEngine(GenericEngine[CapFloorArguments, CapFloorResults]):
    """Analytic cap/floor engine under an affine short-rate model.

    # C++ parity: ``class AnalyticCapFloorEngine`` in
    # analyticcapfloorengine.{hpp,cpp} (v1.42.1).
    """

    def __init__(
        self,
        model: AffineModelLike,
        term_structure: YieldTermStructureProtocol | None = None,
        day_counter: DayCounter | None = None,
    ) -> None:
        super().__init__(CapFloorArguments(), CapFloorResults())
        self._model: AffineModelLike = model
        self._term_structure: YieldTermStructureProtocol | None = term_structure
        self._day_counter: DayCounter | None = day_counter

    def calculate(self) -> None:
        # # C++ parity: analyticcapfloorengine.cpp:34-118 (v1.42.1).
        args = self._arguments
        results = self._results
        results.reset()

        # Resolve reference date / day-counter (same priority as
        # JamshidianSwaptionEngine — model TS takes precedence).
        ref_date = None
        dc: DayCounter | None = None
        if isinstance(self._model, TermStructureConsistentModelLike):
            ts = self._model.term_structure
            ref_date = ts.reference_date()
            dc = ts.day_counter()
        if ref_date is None:
            qassert.require(
                self._term_structure is not None,
                "no term_structure available; pass one explicitly or use a TermStructureConsistentModel",
            )
            assert self._term_structure is not None
            ref_date = self._term_structure.reference_date()
            dc = self._term_structure.day_counter()
        if self._day_counter is not None:
            dc = self._day_counter
        assert dc is not None

        value = 0.0
        cap_type = args.type
        n = len(args.end_dates)

        for i in range(n):
            fixing_time = dc.year_fraction(ref_date, args.fixing_dates[i])
            payment_time = dc.year_fraction(ref_date, args.end_dates[i])

            # Skip strictly-past payments (matches C++ default — we
            # don't separately track ``includeReferenceDateEvents``).
            if payment_time <= 0.0:
                continue

            tenor = args.accrual_times[i]
            fixing = args.forwards[i]

            if fixing_time <= 0.0:
                # Past fixing — intrinsic payout, discounted.
                if cap_type in (CapFloorType.Cap, CapFloorType.Collar):
                    discount = self._model.discount(payment_time)
                    strike = args.cap_rates[i]
                    value += (
                        discount
                        * args.nominals[i]
                        * tenor
                        * args.gearings[i]
                        * max(0.0, fixing - strike)
                    )
                if cap_type in (CapFloorType.Floor, CapFloorType.Collar):
                    discount = self._model.discount(payment_time)
                    strike = args.floor_rates[i]
                    mult = 1.0 if cap_type == CapFloorType.Floor else -1.0
                    value += (
                        discount
                        * args.nominals[i]
                        * tenor
                        * mult
                        * args.gearings[i]
                        * max(0.0, strike - fixing)
                    )
            else:
                # Live optionlet — discount-bond option formula.
                maturity = dc.year_fraction(ref_date, args.start_dates[i])
                if cap_type in (CapFloorType.Cap, CapFloorType.Collar):
                    temp = 1.0 + args.cap_rates[i] * tenor
                    value += (
                        args.nominals[i]
                        * args.gearings[i]
                        * temp
                        * self._model.discount_bond_option(
                            int(OptionType.Put),
                            1.0 / temp,
                            maturity,
                            payment_time,
                        )
                    )
                if cap_type in (CapFloorType.Floor, CapFloorType.Collar):
                    temp = 1.0 + args.floor_rates[i] * tenor
                    mult = 1.0 if cap_type == CapFloorType.Floor else -1.0
                    value += (
                        args.nominals[i]
                        * args.gearings[i]
                        * temp
                        * mult
                        * self._model.discount_bond_option(
                            int(OptionType.Call),
                            1.0 / temp,
                            maturity,
                            payment_time,
                        )
                    )

        results.value = value


__all__ = [
    "AffineModelLike",
    "AnalyticCapFloorEngine",
]
