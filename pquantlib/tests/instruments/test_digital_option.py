"""Tests for DigitalOption.

# C++ parity: there is no separate ``DigitalOption`` C++ class — this
# alias exists so user code can ``isinstance``-discriminate a binary
# option from a plain vanilla.
"""

from __future__ import annotations

from pquantlib.exercise import EuropeanExercise
from pquantlib.instruments.digital_option import DigitalOption
from pquantlib.instruments.vanilla_option import VanillaOption
from pquantlib.payoffs import AssetOrNothingPayoff, CashOrNothingPayoff, OptionType
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _exercise() -> EuropeanExercise:
    return EuropeanExercise(Date.from_ymd(15, Month.June, 2027))


def test_digital_option_is_a_vanilla_option() -> None:
    payoff = CashOrNothingPayoff(OptionType.Call, 100.0, 10.0)
    opt = DigitalOption(payoff, _exercise())
    assert isinstance(opt, VanillaOption)
    assert opt.payoff() is payoff


def test_digital_option_with_asset_or_nothing() -> None:
    payoff = AssetOrNothingPayoff(OptionType.Put, 100.0)
    opt = DigitalOption(payoff, _exercise())
    assert opt.payoff() is payoff
