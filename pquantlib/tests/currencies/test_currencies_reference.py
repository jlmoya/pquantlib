"""Cross-validate every ISO-4217 currency against C++ QuantLib v1.43.

Probe: ``currencies/all`` — all 111 currency classes in
``ql/currencies/{africa,america,asia,europe,oceania,crypto}.hpp``.

The test walks the *reference* rather than a hand-written list, so a currency
added to the probe is checked the moment it lands, and it walks the Python
modules in the other direction too, so a currency that exists only in Python
is caught as well. Which continent module a code lives in is deliberately not
asserted: the C++ split is followed, but the test only cares that the class
exists somewhere and carries the right data.

The Python classes are transcribed from the C++ ``.cpp`` sources; the
reference is read off the compiled C++ objects. The two paths are independent,
which is what makes the comparison worth running.

Tier: EXACT — currency data is static literal data, not computation.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from pquantlib.currencies.currency import Currency
from pquantlib.math.rounding import Type as RoundingType
from pquantlib.testing import reference_reader

#: The per-continent modules, mirroring C++'s per-continent headers.
_MODULES = ("africa", "america", "asia", "europe", "oceania", "crypto")

_CPP: dict[str, Any] = reference_reader.load("currencies/all")
_EXPECTED: dict[str, Any] = _CPP["currencies"]


def _currency_classes() -> dict[str, type[Currency]]:
    """Map ISO code -> currency class, discovered across the continent modules."""
    found: dict[str, type[Currency]] = {}
    for module_name in _MODULES:
        module = importlib.import_module(f"pquantlib.currencies.{module_name}")
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Currency) and obj is not Currency:
                found[obj.__name__.removesuffix("Currency")] = obj
    return found


_CLASSES = _currency_classes()


def test_reference_covers_every_cpp_currency() -> None:
    # 14 africa + 16 america + 29 asia + 42 europe + 2 oceania + 8 crypto.
    assert len(_EXPECTED) == 111


def test_every_pinned_currency_has_a_class() -> None:
    missing = sorted(set(_EXPECTED) - set(_CLASSES))
    assert not missing, f"no Python class for pinned currencies: {missing}"


def test_no_currency_class_outside_the_reference() -> None:
    extra = sorted(set(_CLASSES) - set(_EXPECTED))
    assert not extra, f"Python currency classes absent from C++ v1.43: {extra}"


@pytest.mark.exact
@pytest.mark.parametrize("code", sorted(_EXPECTED))
def test_currency_data_matches_cpp(code: str) -> None:
    expected = _EXPECTED[code]
    ccy = _CLASSES[code]()

    assert ccy.name == expected["name"], f"{code}: name"
    assert ccy.code == expected["code"], f"{code}: code"
    assert ccy.numeric_code == expected["numeric_code"], f"{code}: numeric code"
    assert ccy.symbol == expected["symbol"], f"{code}: symbol"
    assert ccy.fraction_symbol == expected["fraction_symbol"], f"{code}: fraction symbol"
    assert ccy.fractions_per_unit == expected["fractions_per_unit"], f"{code}: fractions per unit"


@pytest.mark.exact
@pytest.mark.parametrize("code", sorted(_EXPECTED))
def test_currency_rounding_matches_cpp(code: str) -> None:
    expected = _EXPECTED[code]
    rounding = _CLASSES[code]().rounding

    assert int(rounding.type) == expected["rounding_type"], f"{code}: rounding type"
    if rounding.type == RoundingType.None_:
        # C++ `Rounding() = default` leaves precision_ and digit_ without an
        # initialiser (only type_ is `= None`), so both are indeterminate there
        # and the probe emits null rather than pinning stack garbage. They are
        # never read either: operator() returns early on type None.
        assert expected["rounding_precision"] is None
        assert expected["rounding_digit"] is None
    else:
        assert rounding.precision == expected["rounding_precision"], f"{code}: precision"
        assert rounding.rounding_digit == expected["rounding_digit"], f"{code}: rounding digit"


@pytest.mark.exact
@pytest.mark.parametrize("code", sorted(_EXPECTED))
def test_currency_triangulation_matches_cpp(code: str) -> None:
    expected_code = _EXPECTED[code]["triangulation_code"]
    triangulation = _CLASSES[code]().triangulation_currency

    if expected_code is None:
        assert triangulation is None, f"{code}: expected no triangulation currency"
    else:
        assert triangulation is not None, f"{code}: expected triangulation {expected_code}"
        assert triangulation.code == expected_code, f"{code}: triangulation currency"


@pytest.mark.exact
@pytest.mark.parametrize("code", sorted(_EXPECTED))
def test_currency_minor_unit_codes_match_cpp(code: str) -> None:
    expected_codes = _EXPECTED[code]["minor_unit_codes"]
    assert _CLASSES[code]().minor_unit_codes == frozenset(expected_codes), f"{code}: minor units"


def test_pre_euro_currencies_triangulate_through_eur() -> None:
    """The only currencies C++ v1.43 gives a triangulation currency are pre-euro."""
    triangulating = {
        code: value["triangulation_code"]
        for code, value in _EXPECTED.items()
        if value["triangulation_code"] is not None
    }
    assert set(triangulating.values()) == {"EUR"}
    assert sorted(triangulating) == [
        "ATS",
        "BEF",
        "DEM",
        "ESP",
        "FIM",
        "FRF",
        "GRD",
        "IEP",
        "ITL",
        "LUF",
        "NLG",
        "PTE",
    ]
