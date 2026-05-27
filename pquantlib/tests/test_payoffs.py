"""Cross-validate Payoff hierarchy against the L3-A C++ probe.

Probe key: ``l3a/foundations`` → ``payoffs.{plain_vanilla,
cash_or_nothing, asset_or_nothing, gap, super_fund, super_share}``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.payoffs import (
    AssetOrNothingPayoff,
    CashOrNothingPayoff,
    GapPayoff,
    OptionType,
    Payoff,
    PlainVanillaPayoff,
    StrikedTypePayoff,
    SuperFundPayoff,
    SuperSharePayoff,
    TypePayoff,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("l3a/foundations")


# --- abstract guardrails ------------------------------------------------


def test_cannot_instantiate_payoff_directly() -> None:
    with pytest.raises(TypeError):
        Payoff()  # type: ignore[abstract]


def test_cannot_instantiate_type_payoff_directly() -> None:
    with pytest.raises(TypeError):
        TypePayoff(OptionType.Call)  # type: ignore[abstract]


def test_cannot_instantiate_striked_type_payoff_directly() -> None:
    with pytest.raises(TypeError):
        StrikedTypePayoff(OptionType.Call, 100.0)  # type: ignore[abstract]


# --- OptionType ---------------------------------------------------------


def test_option_type_call_is_plus_one() -> None:
    assert int(OptionType.Call) == 1


def test_option_type_put_is_minus_one() -> None:
    assert int(OptionType.Put) == -1


def test_option_type_str_matches_cpp() -> None:
    assert str(OptionType.Call) == "Call"
    assert str(OptionType.Put) == "Put"


# --- PlainVanillaPayoff -------------------------------------------------


def test_plain_vanilla_call_matches_cpp(cpp: dict[str, Any]) -> None:
    pv = cpp["payoffs"]["plain_vanilla"]
    call = PlainVanillaPayoff(OptionType.Call, 100.0)
    assert call.name() == pv["name"]
    tolerance.exact(call(120.0), pv["call_at_120"])
    tolerance.exact(call(100.0), pv["call_at_100"])
    tolerance.exact(call(80.0), pv["call_at_80"])


def test_plain_vanilla_put_matches_cpp(cpp: dict[str, Any]) -> None:
    pv = cpp["payoffs"]["plain_vanilla"]
    put = PlainVanillaPayoff(OptionType.Put, 100.0)
    tolerance.exact(put(120.0), pv["put_at_120"])
    tolerance.exact(put(100.0), pv["put_at_100"])
    tolerance.exact(put(80.0), pv["put_at_80"])


def test_plain_vanilla_description() -> None:
    call = PlainVanillaPayoff(OptionType.Call, 100.0)
    assert "Vanilla" in call.description()
    assert "Call" in call.description()
    assert "100" in call.description()


def test_plain_vanilla_strike_accessor() -> None:
    pv = PlainVanillaPayoff(OptionType.Call, 95.5)
    assert pv.strike() == 95.5
    assert pv.option_type() == OptionType.Call


# --- CashOrNothingPayoff ------------------------------------------------


def test_cash_or_nothing_call_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["cash_or_nothing"]
    call = CashOrNothingPayoff(OptionType.Call, 100.0, 1.0)
    assert call.name() == p["name"]
    tolerance.exact(call(120.0), p["call_at_120"])
    tolerance.exact(call(100.0), p["call_at_100"])
    tolerance.exact(call(80.0), p["call_at_80"])


def test_cash_or_nothing_put_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["cash_or_nothing"]
    put = CashOrNothingPayoff(OptionType.Put, 100.0, 1.0)
    tolerance.exact(put(120.0), p["put_at_120"])
    tolerance.exact(put(100.0), p["put_at_100"])
    tolerance.exact(put(80.0), p["put_at_80"])


def test_cash_or_nothing_accessors() -> None:
    c = CashOrNothingPayoff(OptionType.Call, 100.0, 1.5)
    assert c.cash_payoff() == 1.5
    assert "1.5 cash payoff" in c.description()


# --- AssetOrNothingPayoff -----------------------------------------------


def test_asset_or_nothing_call_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["asset_or_nothing"]
    call = AssetOrNothingPayoff(OptionType.Call, 100.0)
    assert call.name() == p["name"]
    tolerance.exact(call(120.0), p["call_at_120"])
    tolerance.exact(call(100.0), p["call_at_100"])
    tolerance.exact(call(80.0), p["call_at_80"])


def test_asset_or_nothing_put_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["asset_or_nothing"]
    put = AssetOrNothingPayoff(OptionType.Put, 100.0)
    tolerance.exact(put(120.0), p["put_at_120"])
    tolerance.exact(put(100.0), p["put_at_100"])
    tolerance.exact(put(80.0), p["put_at_80"])


# --- GapPayoff ----------------------------------------------------------


def test_gap_call_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["gap"]
    call = GapPayoff(OptionType.Call, 100.0, 110.0)
    assert call.name() == p["name"]
    tolerance.exact(call(120.0), p["call_at_120"])
    tolerance.exact(call(100.0), p["call_at_100"])
    tolerance.exact(call(80.0), p["call_at_80"])


def test_gap_put_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["gap"]
    put = GapPayoff(OptionType.Put, 100.0, 90.0)
    tolerance.exact(put(120.0), p["put_at_120"])
    tolerance.exact(put(100.0), p["put_at_100"])
    tolerance.exact(put(80.0), p["put_at_80"])


def test_gap_payoff_second_strike_accessor() -> None:
    g = GapPayoff(OptionType.Call, 100.0, 110.0)
    assert g.second_strike() == 110.0


# --- SuperFundPayoff ----------------------------------------------------


def test_super_fund_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["super_fund"]
    sf = SuperFundPayoff(100.0, 120.0)
    assert sf.name() == p["name"]
    tolerance.exact(sf(90.0), p["at_90"])
    tolerance.exact(sf(100.0), p["at_100"])
    tolerance.tight(sf(110.0), p["at_110"])
    tolerance.exact(sf(120.0), p["at_120"])
    tolerance.exact(sf(130.0), p["at_130"])


def test_super_fund_rejects_non_positive_strike() -> None:
    with pytest.raises(LibraryException, match=r"strike .* must be positive"):
        SuperFundPayoff(0.0, 120.0)


def test_super_fund_rejects_second_strike_le_first() -> None:
    with pytest.raises(LibraryException, match=r"second strike .* higher"):
        SuperFundPayoff(100.0, 100.0)


# --- SuperSharePayoff ---------------------------------------------------


def test_super_share_matches_cpp(cpp: dict[str, Any]) -> None:
    p = cpp["payoffs"]["super_share"]
    ss = SuperSharePayoff(100.0, 120.0, 1.0)
    assert ss.name() == p["name"]
    tolerance.exact(ss(90.0), p["at_90"])
    tolerance.exact(ss(100.0), p["at_100"])
    tolerance.exact(ss(110.0), p["at_110"])
    tolerance.exact(ss(120.0), p["at_120"])
    tolerance.exact(ss(130.0), p["at_130"])


def test_super_share_rejects_second_strike_le_first() -> None:
    with pytest.raises(LibraryException, match=r"second strike .* higher"):
        SuperSharePayoff(100.0, 100.0, 1.0)
