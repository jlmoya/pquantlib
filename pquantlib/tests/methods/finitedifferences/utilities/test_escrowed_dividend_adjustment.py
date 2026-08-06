"""Cross-validation of ``EscrowedDividendAdjustment`` against C++ v1.43.

# C++ parity: ql/methods/finitedifferences/utilities/escroweddividendadjustment.{hpp,cpp}
# @ v1.43 (6b57206e0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.methods.finitedifferences.utilities.escrowed_dividend_adjustment import (
    EscrowedDividendAdjustment,
)
from pquantlib.testing.tolerance import tight


@pytest.mark.parametrize(
    ("t", "key"),
    [
        (0.00, "escrowed_div_adj_t000"),
        (0.25, "escrowed_div_adj_t025"),
        (0.50, "escrowed_div_adj_t050"),
        (0.80, "escrowed_div_adj_t080"),
        (1.00, "escrowed_div_adj_t100"),
    ],
)
def test_dividend_adjustment(
    reference_data: dict[str, Any],
    escrowed_adjustment: EscrowedDividendAdjustment,
    t: float,
    key: str,
) -> None:
    """Only dividends with ``t <= divTime <= maturity`` contribute.

    The probe's schedule pays at +90d (0.2466y), +250d (0.6849y) and +500d
    (1.3699y); with maturity = 1 the third one never counts, and past
    ``t = 0.6849`` the sum empties out entirely (hence the exact zeros at
    t = 0.80 and t = 1.00).
    """
    tight(escrowed_adjustment.dividend_adjustment(t), float(reference_data[key]))


def test_adjustment_is_non_positive(
    escrowed_adjustment: EscrowedDividendAdjustment,
) -> None:
    """# C++ parity: the accumulator only ever subtracts."""
    for t in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        assert escrowed_adjustment.dividend_adjustment(t) <= 0.0


def test_curve_accessors(
    reference_data: dict[str, Any], escrowed_adjustment: EscrowedDividendAdjustment
) -> None:
    """# C++ parity: ``riskFreeRate()`` / ``dividendYield()`` return the handles."""
    tight(
        escrowed_adjustment.risk_free_rate().discount(1.0),
        float(reference_data["escrowed_div_adj_risk_free_discount_1y"]),
    )
    tight(
        escrowed_adjustment.dividend_yield().discount(1.0),
        float(reference_data["escrowed_div_adj_dividend_yield_discount_1y"]),
    )
