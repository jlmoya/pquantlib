"""Unit tests for the ``Compounding`` enum.

# C++ parity: ql/compounding.hpp (v1.42.1) — enum Compounding.

The five compounding conventions are integer constants in C++. We
cross-validate the numeric values against the C++ enum order by
exercising them through ``InterestRate.compound_factor`` and
``InterestRate.implied_rate`` (see ``test_interest_rate.py``).
"""

from __future__ import annotations

from pquantlib.time.compounding import Compounding


def test_compounding_values() -> None:
    # C++ parity: ql/compounding.hpp v1.42.1
    # enum Compounding { Simple = 0, Compounded = 1, Continuous = 2,
    #                    SimpleThenCompounded, CompoundedThenSimple };
    assert int(Compounding.Simple) == 0
    assert int(Compounding.Compounded) == 1
    assert int(Compounding.Continuous) == 2
    assert int(Compounding.SimpleThenCompounded) == 3
    assert int(Compounding.CompoundedThenSimple) == 4


def test_compounding_count() -> None:
    assert len(list(Compounding)) == 5
