"""Shared fixture for the ``v143/experimental/creditloss`` cross-validation.

Rebuilds, in Python, the exact basket + latent model that
``migration-harness/cpp/probes/v143_experimental_creditloss/probe.cpp``
(``makeFixture``, probe.cpp:589-625) builds in C++ v1.43, so every reference
value in ``creditloss.json`` can be compared against the Python port.

Not a test module: the leading underscore keeps pytest from collecting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from pquantlib.currencies.europe import EURCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.credit.basket import Basket
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentmodel,
)
from pquantlib.experimental.credit.default_probability_key import (
    NorthAmericaCorpDefaultKey,
)
from pquantlib.experimental.credit.default_probability_latent_model import (
    LatentModelIntegrationType,
)
from pquantlib.experimental.credit.default_type import Seniority
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.experimental.credit.pool import Pool
from pquantlib.experimental.math.gaussian_copula_policy import GaussianCopulaPolicy
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.testing import reference_reader
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from pquantlib.time.calendar import Calendar

#: probe.cpp:591-593.
HAZARD_RATES: list[float] = [0.005, 0.010, 0.020, 0.035, 0.060]
NOTIONALS: list[float] = [100.0] * 5
RECOVERIES: list[float] = [0.4] * 5
#: probe.cpp:616 — the squared factor loading, so a_i = 0.5.
FACTOR_VALUE: float = 0.25


def load_reference() -> dict[str, Any]:
    """The C++ v1.43 reference emitted by the creditloss probe."""
    return reference_reader.load("v143/experimental/creditloss")


class CreditLossFixture(NamedTuple):
    """The probe's ``Fixture`` struct (probe.cpp:580-587) plus the calendar."""

    today: Date
    calendar: Calendar
    basket: Basket
    latent_model: ConstantLossLatentmodel
    calc_date: Date
    calc_date_2y: Date


def build_fixture() -> CreditLossFixture:
    """Rebuild the C++ probe fixture. Pins the global evaluation date."""
    cal = TARGET()
    # probe.cpp:985-987.
    today = cal.adjust(Date.from_ymd(19, Month.March, 2014))
    ObservableSettings().evaluation_date = today

    dc = Actual365Fixed()
    def_ts = [
        FlatHazardRate.with_settlement_days(0, cal, SimpleQuote(h), dc)
        for h in HAZARD_RATES
    ]
    # probe.cpp:606 — Period() is the *null* period, not the 30-day default.
    # C++ writes ``SeniorSec``, the Markit synonym the enum defines as
    # ``SeniorSec = SecDom`` (defaulttype.hpp:38 + :46). ``SecDom`` is spelled
    # here because the Python aliases are attached dynamically and are
    # therefore invisible to the type checker.
    key = NorthAmericaCorpDefaultKey(
        EURCurrency(), Seniority.SecDom, Period(0, TimeUnit.Days), 1.0
    )
    names = [f"Name{i}" for i in range(len(HAZARD_RATES))]
    pool = Pool()
    for i, nm in enumerate(names):
        pool.add(nm, Issuer(probabilities=[(key, def_ts[i])]), key)

    basket = Basket(today, names, NOTIONALS, pool, 0.03, 0.06)

    weights = [[FACTOR_VALUE**0.5] for _ in range(len(HAZARD_RATES))]
    latent_model = ConstantLossLatentmodel(
        weights,
        RECOVERIES,
        GaussianCopulaPolicy(weights),
        LatentModelIntegrationType.GaussianQuadrature,
    )
    calc_date = cal.advance_period(
        today, Period(60, TimeUnit.Months), BusinessDayConvention.Following
    )
    calc_date_2y = cal.advance_period(
        basket.ref_date(), Period(24, TimeUnit.Months), BusinessDayConvention.Following
    )
    return CreditLossFixture(
        today, cal, basket, latent_model, calc_date, calc_date_2y
    )
