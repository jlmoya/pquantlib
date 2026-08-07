"""Shared fixtures for the v1.43 American / digital / jump-diffusion probes.

Every test module in this cluster reconstructs its market from the ``inputs``
block of ``migration-harness/references/v143/pe/american.json`` rather than
restating constants, so this module only carries the conventions the probe
fixed (``migration-harness/cpp/probes/v143_pe_american/probe.cpp``):

* ``TODAY = Date(1, March, 2025)`` — the probe's ``Settings::evaluationDate``
  and the reference date of every term structure it builds;
* ``Actual365Fixed`` + ``NullCalendar``;
* maturities expressed as an integer day offset from ``TODAY``.

It also carries :func:`expect_number`, which decodes the four kinds of slot
the probe emits:

``float``
    a real value -> compare at the TIGHT tier.
``"unset"``
    the engine left the field ``Null<Real>()``; the instrument accessor must
    raise.
``"nan"`` / ``"inf"`` / ``"-inf"``
    JSON cannot carry non-finite doubles, so the probe quotes them.
``"indeterminate"``
    C++ reads an uninitialised member here (see the probe header); there is
    nothing to cross-validate and the test skips the slot.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    OptionType,
    PlainVanillaPayoff,
    StrikedTypePayoff,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

# probe.cpp — ``const Date TODAY(1, March, 2025);`` and
# ``Settings::instance().evaluationDate() = TODAY;`` in main().
TODAY = Date.from_ymd(1, Month.March, 2025)
DC = Actual365Fixed()
CAL = NullCalendar()

_NON_FINITE: dict[str, float] = {
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
}


def market(spot: float, q: float, r: float, vol: float) -> BlackScholesMertonProcess:
    """Rebuild the probe's ``makeMarket(spot, q, r, vol)``."""
    return BlackScholesMertonProcess(
        x0=SimpleQuote(spot),
        dividend_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=q, day_counter=DC
        ),
        risk_free_ts=FlatForward.from_rate(
            reference_date=TODAY, forward_rate=r, day_counter=DC
        ),
        black_vol_ts=BlackConstantVol(
            reference_date=TODAY, calendar=CAL, day_counter=DC, volatility=vol
        ),
    )


def option_type(name: str) -> OptionType:
    """``"Call"`` / ``"Put"`` -> :class:`OptionType`."""
    return OptionType[name]


def build_payoff(inputs: dict[str, Any]) -> StrikedTypePayoff:
    """Rebuild the probe's payoff from the ``inputs`` block."""
    kind = inputs.get("payoff", "PlainVanilla")
    otype = option_type(inputs["type"])
    strike = float(inputs["strike"])
    if kind == "CashOrNothing":
        return CashOrNothingPayoff(otype, strike, float(inputs["cash"]))
    if kind == "AssetOrNothing":
        return AssetOrNothingPayoff(otype, strike)
    return PlainVanillaPayoff(otype, strike)


def expect_number(actual_getter: Callable[[], float], expected: Any, label: str) -> None:
    """Assert one probe slot, decoding the four encodings described above."""
    if expected == "indeterminate":
        # C++ reads an uninitialised member; nothing to cross-validate.
        return
    if expected == "unset":
        try:
            value = actual_getter()
        except LibraryException:
            return
        raise AssertionError(f"{label}: expected unset, got {value!r}")
    actual = actual_getter()
    if isinstance(expected, str):
        target = _NON_FINITE[expected]
        if math.isnan(target):
            assert math.isnan(actual), f"{label}: expected NaN, got {actual!r}"
        else:
            assert actual == target, f"{label}: expected {target!r}, got {actual!r}"
        return
    tolerance.tight(actual, float(expected), reason=label)
