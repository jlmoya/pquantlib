"""ZeroCondition — floor the solution at zero on every step.

# C++ parity: ql/methods/finitedifferences/zerocondition.hpp (v1.43) —
# ``template <class array_type> class ZeroCondition``.

C++ doc-comment: *"Zero exercise condition. Used in CEV models."* The
whole class is four lines:

.. code-block:: cpp

    void applyTo(array_type& a, Time) const {
        for(Size i=0; i < a.size(); i++) {
            a[i] = std::max(a[i], 0.0);
        }
    }

Two things are worth being precise about, because both are observable:

* **The template parameter collapses.** ``StepCondition`` in PQuantLib is
  already ``Array``-specific (see ``step_conditions/step_condition.py``),
  matching the only instantiation the library ever uses.
* **``std::max(a[i], 0.0)`` is not ``numpy.maximum``.** ``std::max(a, b)``
  is specified as ``a < b ? b : a``, so it returns the *first* argument
  whenever the comparison is false — including when ``a[i]`` is ``-0.0``
  (``-0.0 < 0.0`` is false, so ``-0.0`` survives) and when ``a[i]`` is
  ``NaN`` (also false, so ``NaN`` survives). ``numpy.maximum`` follows
  different rules for signed zero. The port therefore writes the
  comparison out — ``a[a < 0.0] = 0.0`` — which reproduces C++ element
  for element, signed zeros and NaNs included.
"""

from __future__ import annotations

from typing import final

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)


@final
class ZeroCondition(StepCondition):
    """Replace every negative entry with zero, at every step.

    # C++ parity: ``class ZeroCondition : public StepCondition<array_type>``.
    """

    __slots__ = ()

    def apply_to(self, a: Array, t: float) -> None:
        """Apply ``a[i] <- max(a[i], 0.0)`` in place.

        # C++ parity: ``ZeroCondition::applyTo`` — ``t`` is unnamed in
        # C++ and genuinely unused.
        """
        del t
        # See the module docstring: this IS std::max(a[i], 0.0), not
        # numpy.maximum(a, 0.0).
        a[a < 0.0] = 0.0


__all__ = ["ZeroCondition"]
