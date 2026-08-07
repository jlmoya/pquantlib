"""Concrete ``AdditionalBootstrapVariables`` backed by ``SimpleQuote``s.

# C++ parity: ql/termstructures/globalbootstrapvars.{hpp,cpp} (v1.43)

The canonical use is a futures convexity adjustment: the volatility that
feeds ``FuturesConvAdjustmentQuote`` is not a curve value, but it does
have to be solved for alongside the curve. Wrapping it in a
:class:`SimpleQuoteVariables` hands it to ``GlobalBootstrap`` as an extra
optimization variable, and the optimizer writes each trial value straight
back into the quote.

Bounds are enforced by a change of variables rather than by a constrained
optimizer: with a lower bound ``lb``, the optimizer works in ``y`` and the
quote sees ``exp(y) + lb``, which is ``> lb`` for every finite ``y``. With
no lower bound both transforms are the identity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.termstructures.global_bootstrap import AdditionalBootstrapVariables

if TYPE_CHECKING:
    from pquantlib.math.array import Array
    from pquantlib.quotes.simple_quote import SimpleQuote


def _get[T](v: Sequence[T], i: int, default: T) -> T:
    """``QuantLib::detail::get`` — ql/utilities/vectors.hpp:33-42.

    The fallback past the end of a NON-EMPTY sequence is ``v[-1]``, NOT the
    default: a short ``lowerBounds`` therefore applies its last bound to
    every remaining quote. Only a completely empty sequence yields the
    default. Getting this backwards silently unbounds the tail of the
    variable vector, so it is pinned by
    ``test_global_bootstrap.py::test_simple_quote_variables_short_vectors``.
    """
    if not v:
        return default
    if i < len(v):
        return v[i]
    return v[-1]


class SimpleQuoteVariables(AdditionalBootstrapVariables):
    """Additional bootstrap variables held in ``SimpleQuote``s.

    # C++ parity: ``class SimpleQuoteVariables`` at
    # globalbootstrapvars.hpp:12-28 and globalbootstrapvars.cpp:10-48.

    ``initial_guesses`` and ``lower_bounds`` may be shorter than
    ``quotes`` (each entry then falls back per :func:`_get`) but never
    longer.
    """

    __slots__ = ("_initial_guesses", "_lower_bounds", "_quotes")

    def __init__(
        self,
        quotes: Sequence[SimpleQuote],
        initial_guesses: Sequence[float] | None = None,
        lower_bounds: Sequence[float] | None = None,
    ) -> None:
        self._quotes: list[SimpleQuote] = list(quotes)
        self._initial_guesses: list[float] = (
            list(initial_guesses) if initial_guesses is not None else []
        )
        self._lower_bounds: list[float] = (
            list(lower_bounds) if lower_bounds is not None else []
        )
        # C++ parity: globalbootstrapvars.cpp:15-16.
        qassert.require(
            len(self._initial_guesses) <= len(self._quotes), "too many initialGuesses"
        )
        qassert.require(
            len(self._lower_bounds) <= len(self._quotes), "too many lowerBounds"
        )

    def initialize(self, valid_data: bool) -> Array:
        """Seed the quotes and return the guesses in optimizer space.

        # C++ parity: ``SimpleQuoteVariables::initialize``, cpp:19-32.

        With ``valid_data`` the CURRENT quote values are reused as the
        guess and the quotes are left alone; without it each quote is
        overwritten with its configured initial guess. Either way the
        returned value is in the optimizer's coordinates, i.e. run
        through :meth:`_transform_inverse`.
        """
        guesses = np.empty(len(self._quotes), dtype=np.float64)
        for i, quote in enumerate(self._quotes):
            if valid_data:
                guess = quote.value()
            else:
                guess = _get(self._initial_guesses, i, 0.0)
                quote.set_value(guess)
            guesses[i] = self._transform_inverse(guess, i)
        return guesses

    def update(self, x: Array) -> None:
        """Write the optimizer's values back into the quotes.

        # C++ parity: ``SimpleQuoteVariables::update``, cpp:34-38. The loop
        # runs over ``x``, not over the quotes, so a shorter ``x`` updates
        # only a prefix.
        """
        for i in range(len(x)):
            self._quotes[i].set_value(self._transform_direct(float(x[i]), i))

    # -- transforms --------------------------------------------------------

    def _transform_direct(self, x: float, i: int) -> float:
        """Optimizer space -> quote space. C++ parity: cpp:40-43."""
        lb = _get(self._lower_bounds, i, None)
        # C++ compares against ``Null<Real>()``; the pquantlib port uses
        # ``None`` as the "no bound" sentinel, so the branch is an identity
        # test rather than a float comparison.
        return x if lb is None else math.exp(x) + lb

    def _transform_inverse(self, x: float, i: int) -> float:
        """Quote space -> optimizer space. C++ parity: cpp:45-48."""
        lb = _get(self._lower_bounds, i, None)
        return x if lb is None else math.log(x - lb)

    # -- inspectors --------------------------------------------------------

    def quotes(self) -> list[SimpleQuote]:
        """C++ ``quotes_`` (globalbootstrapvars.hpp:26)."""
        return list(self._quotes)

    def initial_guesses(self) -> list[float]:
        """C++ ``initialGuesses_`` (globalbootstrapvars.hpp:27)."""
        return list(self._initial_guesses)

    def lower_bounds(self) -> list[float]:
        """C++ ``lowerBounds_`` (globalbootstrapvars.hpp:27)."""
        return list(self._lower_bounds)


__all__ = ["SimpleQuoteVariables"]
