"""The RNG-policy parameter carried by every Monte-Carlo engine builder.

# C++ parity: the ``template <class RNG = PseudoRandom, class S = Statistics>``
# head of ``MCEverestEngine`` / ``MCHimalayaEngine`` / ``MCPagodaEngine`` /
# ``MCPathBasketEngine`` / ``MCAmericanPathEngine`` and of the matching
# ``MakeMC*`` factories (v1.43).

C++ instantiates those engines on a *traits type*, not on an instance: the
engine reads ``RNG::allowsErrorEstimate`` at compile time and calls
``RNG::make_sequence_generator(dimension, seed)`` to build its Gaussian
sequence. Python has the same two members on
:class:`~pquantlib.math.randomnumbers.rng_traits.PseudoRandom` and
:class:`~pquantlib.math.randomnumbers.rng_traits.LowDiscrepancy`, so the
template argument survives the port as an ordinary *class object* passed to
the constructor.

:data:`RngTraits` is the type of that argument. It exists in one place so the
five ``MakeMC*`` builders and their engines all spell it the same way.
"""

from __future__ import annotations

from pquantlib.math.randomnumbers.random_number_generator import (
    InverseCumulative,
    RandomNumberGenerator,
    UniformSequenceGenerator,
)
from pquantlib.math.randomnumbers.rng_traits import (
    GenericLowDiscrepancy,
    GenericPseudoRandom,
)

type RngTraits = (
    type[GenericPseudoRandom[RandomNumberGenerator, InverseCumulative]]
    | type[GenericLowDiscrepancy[UniformSequenceGenerator, InverseCumulative]]
)
"""A random-number policy class: ``PseudoRandom`` or ``LowDiscrepancy``.

# C++ parity: the ``RNG`` template parameter. Only ever passed as a class,
# never instantiated — exactly as C++ never instantiates the traits struct.
"""


__all__ = ["RngTraits"]
