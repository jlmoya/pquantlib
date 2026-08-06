"""EurodollarFuturesImpliedStdDevQuote — implied stddev from futures options.

# C++ parity: ql/quotes/eurodollarfuturesquote.{hpp,cpp} (v1.43)

The quote works in *rate* space while its inputs are quoted in *price* space,
which produces two inversions that are easy to lose in a port and are both
reproduced verbatim:

* the constructor stores ``strike_ = 100.0 - strike`` and ``value()`` computes
  ``forwardValue = 100.0 - forward->value()``;
* when the rate strike is **above** the rate forward the class prices a
  ``Option::Call`` off the **put** quote, and otherwise a ``Option::Put`` off
  the **call** quote — a futures put is a rate call.

Unlike :class:`~pquantlib.quotes.implied_std_dev_quote.ImpliedStdDevQuote`, this
class does **not** swallow inversion failures: a non-converging solve
propagates out of ``value()``.

# C++ parity divergence (Handle vs object): C++ takes ``Handle<Quote>`` for the
forward and the two option prices. This port threads the ``Quote`` directly
(see ``cashflows/cms_coupon_pricer.py``); ``None`` models C++'s *empty* handle,
which ``isValid()`` explicitly tests for — and which is genuinely useful here,
since only the branch-relevant price handle has to be populated.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.patterns.lazy_object import LazyObject
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_formula import black_formula_implied_std_dev
from pquantlib.quotes.quote import Quote

# C++ ``static const Real discount = 1.0; static const Real displacement = 0.0;``
_DISCOUNT: float = 1.0
_DISPLACEMENT: float = 0.0


class EurodollarFuturesImpliedStdDevQuote(Quote, LazyObject):
    """Quote for the Eurodollar-future implied standard deviation."""

    def __init__(
        self,
        forward: Quote | None,
        call_price: Quote | None,
        put_price: Quote | None,
        strike: float,
        guess: float = 0.15,
        accuracy: float = 1.0e-6,
        max_iter: int = 100,
    ) -> None:
        super().__init__()
        self._implied_stdev: float = guess
        # C++ parity: the constructor converts the quoted futures strike into
        # a rate strike. ``strike`` is a price, ``self._strike`` is a rate.
        self._strike: float = 100.0 - strike
        self._accuracy: float = accuracy
        self._max_iter: int = max_iter
        self._forward: Quote | None = forward
        self._call_price: Quote | None = call_price
        self._put_price: Quote | None = put_price
        for element in (forward, call_price, put_price):
            if element is not None:
                element.register_with(self)

    # --- Quote interface --------------------------------------------------

    def value(self) -> float:
        self.calculate()
        return self._implied_stdev

    def is_valid(self) -> bool:
        if self._forward is None or not self._forward.is_valid():
            return False
        forward_value = 100.0 - self._forward.value()
        if self._strike > forward_value:
            return self._put_price is not None and self._put_price.is_valid()
        return self._call_price is not None and self._call_price.is_valid()

    # --- LazyObject interface ---------------------------------------------

    def _perform_calculations(self) -> None:
        qassert.require(self._forward is not None, "empty Handle cannot be dereferenced")
        assert self._forward is not None
        forward_value = 100.0 - self._forward.value()
        if self._strike > forward_value:
            option_type, price = OptionType.Call, self._put_price
        else:
            option_type, price = OptionType.Put, self._call_price
        qassert.require(price is not None, "empty Handle cannot be dereferenced")
        assert price is not None
        self._implied_stdev = black_formula_implied_std_dev(
            option_type,
            self._strike,
            forward_value,
            price.value(),
            _DISCOUNT,
            _DISPLACEMENT,
            self._implied_stdev,
            self._accuracy,
            self._max_iter,
        )
