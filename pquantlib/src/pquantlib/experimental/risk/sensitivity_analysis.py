"""Bump-and-revalue sensitivity helpers.

# C++ parity: ql/experimental/risk/sensitivityanalysis.{hpp,cpp}
#             (recovered from 6f379f4e9~1 — see package note).

PV01-style finite-difference sensitivities of a portfolio of instruments
to a set of ``SimpleQuote`` inputs:

- ``aggregate_npv`` — weighted sum of instrument NPVs.
- ``parallel_analysis`` — shift all quotes together; return (delta, gamma).
- ``bucket_analysis`` — shift a single quote; return (delta, gamma).

``SensitivityAnalysis`` selects one-sided vs centred differencing.

# C++ parity divergence — the recovered C++ functions take
# ``Handle<SimpleQuote>``; PQuantLib threads plain ``SimpleQuote`` objects
# (the observer wiring that a Handle provides is already on the Quote).
# The richer matrix / parameter-bucket overloads from the C++ source are
# deferred (see docs/carve-outs.md): the spec-named core (enum +
# aggregate_npv + parallel_analysis + bucket_analysis) is ported.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from pquantlib import qassert

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.instruments.instrument import Instrument
    from pquantlib.quotes.simple_quote import SimpleQuote


class SensitivityAnalysis(IntEnum):
    """Finite-difference scheme selector.

    # C++ parity: ``enum SensitivityAnalysis { OneSide, Centered }``.
    """

    OneSide = 0
    Centered = 1


def aggregate_npv(instruments: Sequence[Instrument], quantities: Sequence[float]) -> float:
    """Weighted sum of instrument NPVs.

    # C++ parity: sensitivityanalysis.cpp ``aggregateNPV``.

    An empty (or single unit) ``quantities`` is treated as a unit vector.
    """
    n = len(instruments)
    npv = 0.0
    if not quantities or (len(quantities) == 1 and quantities[0] == 1.0):
        for k in range(n):
            npv += instruments[k].npv()
    else:
        qassert.require(
            len(quantities) == n,
            f"dimension mismatch between instruments ({n}) and quantities ({len(quantities)})",
        )
        for k in range(n):
            npv += quantities[k] * instruments[k].npv()
    return npv


def parallel_analysis(
    quotes: Sequence[SimpleQuote],
    instruments: Sequence[Instrument],
    quantities: Sequence[float],
    shift: float = 0.0001,
    sensitivity_type: SensitivityAnalysis = SensitivityAnalysis.Centered,
    reference_npv: float | None = None,
) -> tuple[float, float | None]:
    """Parallel-shift PV01 sensitivity for a vector of quotes.

    # C++ parity: sensitivityanalysis.cpp ``parallelAnalysis``.

    Returns ``(first_derivative, second_derivative)``; the second is
    ``None`` for one-sided differencing.  All quotes are tweaked together.
    """
    qassert.require(len(quotes) > 0, "empty SimpleQuote vector")
    n = len(quotes)
    qassert.require(shift != 0.0, "zero shift not allowed")

    result_first = 0.0
    result_second: float | None = 0.0
    if not instruments:
        return result_first, result_second

    if reference_npv is None:
        reference_npv = aggregate_npv(instruments, quantities)

    quote_values: list[float | None] = [None] * n
    for i in range(n):
        if quotes[i].is_valid():
            quote_values[i] = quotes[i].value()
    try:
        for i in range(n):
            if quote_values[i] is not None:
                quotes[i].set_value(quote_values[i] + shift)  # type: ignore[operator]
        npv = aggregate_npv(instruments, quantities)
        if sensitivity_type == SensitivityAnalysis.OneSide:
            result_first = (npv - reference_npv) / shift
            result_second = None
        else:  # Centered
            for i in range(n):
                if quote_values[i] is not None:
                    quotes[i].set_value(quote_values[i] - shift)  # type: ignore[operator]
            npv2 = aggregate_npv(instruments, quantities)
            result_first = (npv - npv2) / (2.0 * shift)
            result_second = (npv - 2.0 * reference_npv + npv2) / (shift * shift)
        for i in range(n):
            if quote_values[i] is not None:
                quotes[i].set_value(quote_values[i])
    except Exception:
        for i in range(n):
            if quote_values[i] is not None:
                quotes[i].set_value(quote_values[i])
        raise

    return result_first, result_second


def bucket_analysis(
    quote: SimpleQuote,
    instruments: Sequence[Instrument],
    quantities: Sequence[float],
    shift: float = 0.0001,
    sensitivity_type: SensitivityAnalysis = SensitivityAnalysis.Centered,
    reference_npv: float | None = None,
) -> tuple[float, float | None]:
    """(Bucket) PV01 sensitivity for a single quote.

    # C++ parity: sensitivityanalysis.cpp ``bucketAnalysis`` (single-quote
    # overload).  Returns ``(first_derivative, second_derivative)``.
    """
    qassert.require(shift != 0.0, "zero shift not allowed")

    result_first = 0.0
    result_second: float | None = 0.0
    if not instruments:
        return result_first, result_second

    if reference_npv is None:
        reference_npv = aggregate_npv(instruments, quantities)

    if not quote.is_valid():
        return result_first, result_second
    quote_value = quote.value()

    try:
        quote.set_value(quote_value + shift)
        npv = aggregate_npv(instruments, quantities)
        if sensitivity_type == SensitivityAnalysis.OneSide:
            result_first = (npv - reference_npv) / shift
            result_second = None
        else:  # Centered
            quote.set_value(quote_value - shift)
            npv2 = aggregate_npv(instruments, quantities)
            result_first = (npv - npv2) / (2.0 * shift)
            result_second = (npv - 2.0 * reference_npv + npv2) / (shift * shift)
        quote.set_value(quote_value)
    except Exception:
        quote.set_value(quote_value)
        raise

    return result_first, result_second


__all__ = ["SensitivityAnalysis", "aggregate_npv", "bucket_analysis", "parallel_analysis"]
