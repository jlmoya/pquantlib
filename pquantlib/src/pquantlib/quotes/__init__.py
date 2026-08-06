"""Quote handle abstraction — market observables.

# C++ parity: ql/quote.hpp + ql/quotes/{simplequote,derivedquote,
              compositequote}.hpp (v1.42.1)
# C++ parity: ql/quotes/{impliedstddevquote,eurodollarfuturesquote,
              forwardvaluequote,forwardswapquote,futuresconvadjustmentquote,
              lastfixingquote,multicompositequote}.hpp (v1.43)

One snake_case module per C++ class; import the class from its own module
(``from pquantlib.quotes.last_fixing_quote import LastFixingQuote``) — this
package exports nothing itself.
"""
