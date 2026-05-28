"""CPICouponPricer — analytic / Black pricer for CPI coupons.

# C++ parity: ql/cashflows/cpicouponpricer.{hpp,cpp} (v1.42.1).

C++ class hierarchy:

    InflationCouponPricer (abstract)
      -> CPICouponPricer    (concrete swaplet; cap/floor needs vol surface)

The C++ ``CPICouponPricer`` has a single concrete class — there's no
distinct Black/Bachelier subclass for CPI (those are only on the YoY
side). However, the docstring + L7-C spec asks for a ``BlackCPICouponPricer``
alias so callers can write ``BlackCPICouponPricer()`` symmetrically with
the YoY family; we expose that as a no-op subclass.

Python divergences from C++:

- The C++ pricer takes the nominal-term-structure handle (and optional
  ``CPIVolatilitySurface``) at construction. We accept both as Protocol-
  typed optionals; the vol-surface arg is only consulted in cap/floor
  rate paths.
- Cap/floor methods require a CPI volatility surface (L7-D); without
  one, they raise ``LibraryException`` rather than silently returning 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.inflation_coupon_pricer import InflationCouponPricer
from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.cpi_coupon import CPICoupon
    from pquantlib.cashflows.inflation_coupon import InflationCoupon
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


class CPICouponPricer(InflationCouponPricer):
    """Analytic CPI-coupon pricer.

    # C++ parity: ``CPICouponPricer`` in cpicouponpricer.hpp.

    Swaplet rate is the simple ``gearing * indexRatio(end)`` product —
    i.e. ``fixedRate * (CPI(end) / CPI(base))``. Cap/floor rates require
    a ``CPIVolatilitySurface`` (deferred to L7-D).
    """

    def __init__(
        self,
        nominal_term_structure: YieldTermStructureProtocol | None = None,
        caplet_vol: object | None = None,
    ) -> None:
        """Construct a CPICouponPricer.

        ``caplet_vol`` is the CPI volatility surface (L7-D); typed as
        ``object`` here to keep the cluster decoupled from
        ``CPIVolatilitySurface`` until L7-D lands.
        """
        super().__init__()
        self._nominal_term_structure: YieldTermStructureProtocol | None = nominal_term_structure
        self._caplet_vol: object | None = caplet_vol
        # Per-coupon state populated by ``initialize``:
        self._coupon: CPICoupon | None = None
        self._gearing: float = 0.0
        self._discount: float | None = 1.0

    # ---- inspectors --------------------------------------------------

    def caplet_volatility(self) -> object | None:
        """C++ parity: ql/cashflows/cpicouponpricer.hpp:46-48 (inline)."""
        return self._caplet_vol

    def nominal_term_structure(self) -> YieldTermStructureProtocol | None:
        """C++ parity: ql/cashflows/cpicouponpricer.hpp:50-52 (inline)."""
        return self._nominal_term_structure

    def set_caplet_volatility(self, caplet_vol: object) -> None:
        """C++ parity: ql/cashflows/cpicouponpricer.cpp:38-43."""
        qassert.require(caplet_vol is not None, "empty capletVol handle")
        self._caplet_vol = caplet_vol

    # ---- pricer wiring -----------------------------------------------

    def initialize(self, coupon: InflationCoupon) -> None:
        """Capture per-coupon state.

        # C++ parity: ql/cashflows/cpicouponpricer.cpp:110-126.
        # The C++ code casts to ``CPICoupon*``; we use Python isinstance.
        """
        # Local import for the coupon ↔ pricer cycle.
        from pquantlib.cashflows.cpi_coupon import CPICoupon  # noqa: PLC0415

        qassert.require(
            isinstance(coupon, CPICoupon),
            "CPICouponPricer requires a CPICoupon",
        )
        assert isinstance(coupon, CPICoupon)
        self._coupon = coupon
        self._gearing = coupon.fixed_rate()
        self._payment_date = coupon.date()

        # Discount = nominal_TS.discount(payment_date), if a curve is set
        # and the payment is after its reference date. Mark None == invalid
        # for prices (allows rate methods to work without a curve).
        self._discount = 1.0
        if self._nominal_term_structure is None:
            self._discount = None
        else:
            ref = self._nominal_term_structure.reference_date()
            if self._payment_date > ref:
                self._discount = self._nominal_term_structure.discount(self._payment_date)

    # ---- InflationCouponPricer interface -----------------------------

    def swaplet_rate(self) -> float:
        """``gearing * indexRatio(end)`` == ``fixedRate * CPI(end)/CPI(base)``.

        # C++ parity: ql/cashflows/cpicouponpricer.cpp:135-137 — delegates
        # to ``accrued_rate(accrualEndDate)``.
        """
        return self.accrued_rate(self._coupon_required().accrual_end_date())

    def swaplet_price(self) -> float:
        """``swaplet_rate * accrual_period * discount``.

        # C++ parity: ql/cashflows/cpicouponpricer.cpp:129-132.
        """
        qassert.require(
            self._discount is not None,
            "no nominal term structure provided",
        )
        assert self._discount is not None
        return self.swaplet_rate() * self._coupon_required().accrual_period() * self._discount

    def caplet_price(self, effective_cap: float) -> float:
        """C++ parity: ql/cashflows/cpicouponpricer.cpp:51-54."""
        return self._gearing * self._optionlet_price(OptionType.Call, effective_cap)

    def caplet_rate(self, effective_cap: float) -> float:
        """C++ parity: ql/cashflows/cpicouponpricer.cpp:61-63."""
        return self._gearing * self._optionlet_rate(OptionType.Call, effective_cap)

    def floorlet_price(self, effective_floor: float) -> float:
        """C++ parity: ql/cashflows/cpicouponpricer.cpp:46-49."""
        return self._gearing * self._optionlet_price(OptionType.Put, effective_floor)

    def floorlet_rate(self, effective_floor: float) -> float:
        """C++ parity: ql/cashflows/cpicouponpricer.cpp:57-59."""
        return self._gearing * self._optionlet_rate(OptionType.Put, effective_floor)

    # ---- CPI-specific accessors --------------------------------------

    def accrued_rate(self, settlement_date: Date) -> float:
        """``gearing * indexRatio(d)`` — rate at an arbitrary settlement date.

        # C++ parity: ql/cashflows/cpicouponpricer.cpp:140-142.
        """
        return self._gearing * self._coupon_required().index_ratio(settlement_date)

    # ---- helpers / internal ------------------------------------------

    def _coupon_required(self) -> CPICoupon:
        qassert.require(self._coupon is not None, "coupon not initialized")
        assert self._coupon is not None
        return self._coupon

    def _optionlet_price(self, option_type: OptionType, eff_strike: float) -> float:
        """C++ parity: ql/cashflows/cpicouponpricer.cpp:74-78."""
        qassert.require(
            self._discount is not None,
            "no nominal term structure provided",
        )
        assert self._discount is not None
        cpn = self._coupon_required()
        return self._optionlet_rate(option_type, eff_strike) * cpn.accrual_period() * self._discount

    def _optionlet_rate(self, option_type: OptionType, eff_strike: float) -> float:
        """C++ parity: ql/cashflows/cpicouponpricer.cpp:81-107.

        The non-vol-surface branch raises ``LibraryException`` (== C++
        ``QL_REQUIRE(!capletVolatility().empty(), "missing optionlet
        volatility")``); the vol-surface branch routes through
        ``_optionlet_price_imp`` which subclasses override.

        The C++ "fixingDate <= evaluationDate" branch (pricer.cpp:84-94)
        is handled here against the L2-D ``CashFlow.has_occurred`` style:
        we let the *caller* check whether the fixing has been published
        (since this port has no global ``Settings::evaluationDate``).
        If the surface is missing we raise; otherwise we delegate to
        ``_optionlet_price_imp``.
        """
        qassert.require(
            self._caplet_vol is not None,
            "missing optionlet volatility",
        )
        # No way to get to optionlet_price_imp in the base class without
        # the vol surface; subclasses (BlackCPICouponPricer) override.
        msg = (
            "cap/floor optionlet pricing requires CPIVolatilitySurface "
            "wiring (deferred to L7-D)"
        )
        raise LibraryException(msg)


class BlackCPICouponPricer(CPICouponPricer):
    """Black-formula CPI cap/floor pricer.

    # C++ parity: C++ does not ship a distinct ``BlackCPICouponPricer``
    # because the CPI hierarchy doesn't separate the analytic models the
    # way YoY does (Black / Bachelier / UnitDisplacedBlack). We expose
    # this alias to mirror the symmetric YoY API and to give the cluster
    # a canonical entry point for L7-D's CPI cap/floor engines.

    The Black-formula branch fires when a ``CPIVolatilitySurface`` is
    plugged in. Until L7-D lands, this routes to the same
    ``LibraryException`` as the base class for the cap/floor methods
    (swaplet pricing works identically).
    """

    def _black_price(
        self,
        option_type: OptionType,
        eff_strike: float,
        forward: float,
        std_dev: float,
    ) -> float:
        """Closed-form Black call/put for the optionlet.

        # C++ parity divergence: the C++ ``optionletPriceImp`` for CPI is
        # only invoked once a vol surface is present. We expose this as
        # a helper so the L7-D CPI cap/floor engine can call it directly
        # once the vol surface is wired.
        """
        return black_formula(option_type, eff_strike, forward, std_dev)
