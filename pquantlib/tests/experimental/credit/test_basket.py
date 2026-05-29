"""Cross-validate Basket structure + tranche fields.

Probe source: migration-harness/cpp/probes/cluster_w3c/probe.cpp
Reference:    migration-harness/references/cluster/w3c.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.currencies.america import USDCurrency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.basket import Basket
from pquantlib.experimental.credit.default_loss_model import DefaultLossModel
from pquantlib.experimental.credit.default_probability_key import DefaultProbKey
from pquantlib.experimental.credit.default_type import (
    AtomicDefault,
    DefaultType,
    Restructuring,
    Seniority,
)
from pquantlib.experimental.credit.issuer import Issuer
from pquantlib.experimental.credit.pool import Pool
from pquantlib.instruments.claim import FaceValueClaim
from pquantlib.patterns.observer import Observable
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.credit.flat_hazard_rate import FlatHazardRate
from pquantlib.testing import reference_reader, tolerance
from pquantlib.time.date import Date
from pquantlib.time.month import Month


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3c")


def _make_5_issuer_basket(
    attach: float = 0.0,
    detach: float = 0.1,
) -> Basket:
    today = Date.from_ymd(15, Month.January, 2024)
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(dt,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    issuer = Issuer(probabilities=[(key, curve)])
    pool = Pool()
    names = [f"N{i}" for i in range(5)]
    notionals = [1.0e6 for _ in range(5)]
    for n in names:
        pool.add(n, issuer, key)
    return Basket(
        today, names, notionals, pool,
        attach, detach,
        FaceValueClaim(),
    )


def test_basket_structure_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    basket = _make_5_issuer_basket()
    ref = cpp_ref["basket"]
    # Structural fields — TIGHT.
    assert basket.size() == ref["size"]
    tolerance.tight(basket.basket_notional(), ref["basket_notional"])
    tolerance.tight(basket.attachment_amount(), ref["attachment_amount"])
    tolerance.tight(basket.detachment_amount(), ref["detachment_amount"])
    tolerance.tight(basket.tranche_notional(), ref["tranche_notional"])
    tolerance.tight(basket.attachment_ratio(), ref["attachment_ratio"])
    tolerance.tight(basket.detachment_ratio(), ref["detachment_ratio"])


def test_basket_empty_notionals_raise() -> None:
    today = Date.from_ymd(15, Month.January, 2024)
    pool = Pool()
    with pytest.raises(LibraryException):
        Basket(today, [], [], pool)


def test_basket_invalid_attach_detach_raises() -> None:
    today = Date.from_ymd(15, Month.January, 2024)
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(dt,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    issuer = Issuer(probabilities=[(key, curve)])
    pool = Pool()
    pool.add("N0", issuer, key)

    # detach > 1 raises.
    with pytest.raises(LibraryException):
        Basket(today, ["N0"], [1.0e6], pool, 0.0, 1.5)
    # attach > detach raises.
    with pytest.raises(LibraryException):
        Basket(today, ["N0"], [1.0e6], pool, 0.5, 0.3)


def test_basket_notional_mismatch_raises() -> None:
    today = Date.from_ymd(15, Month.January, 2024)
    usd = USDCurrency()
    dt = DefaultType(AtomicDefault.Bankruptcy, Restructuring.NoRestructuring)
    key = DefaultProbKey(
        event_types=(dt,),
        currency=usd,
        seniority=Seniority.SnrFor,
    )
    curve = FlatHazardRate(today, SimpleQuote(0.02), Actual365Fixed())
    issuer = Issuer(probabilities=[(key, curve)])
    pool = Pool()
    pool.add("N0", issuer, key)
    # pool.size() == 1 but 2 notionals.
    with pytest.raises(LibraryException):
        Basket(today, ["N0"], [1.0e6, 1.0e6], pool)


def test_basket_names_match_pool_names() -> None:
    basket = _make_5_issuer_basket()
    assert basket.names() == [f"N{i}" for i in range(5)]


def test_basket_default_keys_match_pool() -> None:
    basket = _make_5_issuer_basket()
    keys = basket.default_keys()
    assert len(keys) == basket.size()


def test_basket_equity_tranche() -> None:
    """attach=0, detach=0.05 -> equity tranche 250k on 5M basket."""
    basket = _make_5_issuer_basket(attach=0.0, detach=0.05)
    tolerance.tight(basket.attachment_amount(), 0.0)
    tolerance.tight(basket.detachment_amount(), 250_000.0)
    tolerance.tight(basket.tranche_notional(), 250_000.0)


def test_basket_senior_tranche() -> None:
    """attach=0.5, detach=1 -> senior tranche 2.5M on 5M basket."""
    basket = _make_5_issuer_basket(attach=0.5, detach=1.0)
    tolerance.tight(basket.attachment_amount(), 2_500_000.0)
    tolerance.tight(basket.detachment_amount(), 5_000_000.0)
    tolerance.tight(basket.tranche_notional(), 2_500_000.0)


class _StubLossModel(Observable):
    """A trivial DefaultLossModel for wiring tests."""

    def __init__(self, etl: float) -> None:
        super().__init__()
        self._etl = etl
        self._basket: Basket | None = None

    def set_basket(self, basket: object) -> None:
        assert isinstance(basket, Basket)
        self._basket = basket

    def expected_tranche_loss(self, d: Date) -> float:
        del d
        return self._etl


def test_basket_loss_model_wiring() -> None:
    basket = _make_5_issuer_basket()
    model = _StubLossModel(etl=12345.0)
    # Verify the stub satisfies the Protocol structurally.
    assert isinstance(model, DefaultLossModel)
    basket.set_loss_model(model)
    today = Date.from_ymd(15, Month.January, 2024)
    tolerance.tight(basket.expected_tranche_loss(today), 12345.0)


def test_basket_expected_tranche_loss_without_model_raises() -> None:
    basket = _make_5_issuer_basket()
    today = Date.from_ymd(15, Month.January, 2024)
    with pytest.raises(LibraryException):
        basket.expected_tranche_loss(today)
