"""Payoffs for single and double nested options of sticky or ratchet type.

# C++ parity: ql/instruments/stickyratchet.hpp + .cpp (v1.43).

``DoubleStickyRatchetPayoff`` is the one implementation; the six named
subclasses are presets over its eleven scalars:

===================  =====  =====  =====================  ====================
class                type1  type2  gearings               initial values
===================  =====  =====  =====================  ====================
``RatchetPayoff``     -1      0    ``g1, 0, g2``          ``iv, 0``
``StickyPayoff``      +1      0    ``g1, 0, g2``          ``iv, 0``
``RatchetMaxPayoff``  -1     -1    ``g1, g2, g3``         ``iv1, iv2``
``RatchetMinPayoff``  -1     +1    ``g1, g2, g3``         ``iv1, iv2``
``StickyMaxPayoff``   +1     -1    ``g1, g2, g3``         ``iv1, iv2``
``StickyMinPayoff``   +1     +1    ``g1, g2, g3``         ``iv1, iv2``
===================  =====  =====  =====================  ====================

Note the slotting for the two single-option variants: their *second*
gearing/spread goes to ``gearing3``/``spread3``, not to ``gearing2``/
``spread2`` — ``gearing2``/``spread2`` are the second option's, which a
single-option payoff does not have. Getting that wrong yields a plausible
number, so ``tests/instruments/test_sticky_ratchet.py`` pins each preset
against the equivalent explicit ``DoubleStickyRatchetPayoff``.

Not ported: ``StickyRatchetPayoff``, ``RatchetPayoff_2`` and
``StickyPayoff_2``. Those three appear only inside the ``/*--- ... ---*/``
comment block at the foot of stickyratchet.hpp (and of the .cpp), where the
C++ source itself labels them "Old code ... superated by
DoubleStickyRatchetPayoff class above". They are dead source text, not API.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.payoffs import Payoff


class DoubleStickyRatchetPayoff(Payoff):
    """Intermediate payoff for single/double sticky/ratchet options.

    ``initial_value1`` / ``initial_value2`` can be a (forward) rate or a
    coupon/accrual factor.

    C++ parity: stickyratchet.hpp:35-58, stickyratchet.cpp:24-41.
    """

    def __init__(
        self,
        type1: float,
        type2: float,
        gearing1: float,
        gearing2: float,
        gearing3: float,
        spread1: float,
        spread2: float,
        spread3: float,
        initial_value1: float,
        initial_value2: float,
        accrual_factor: float,
    ) -> None:
        self._type1 = type1
        self._type2 = type2
        self._gearing1 = gearing1
        self._gearing2 = gearing2
        self._gearing3 = gearing3
        self._spread1 = spread1
        self._spread2 = spread2
        self._spread3 = spread3
        self._initial_value1 = initial_value1
        self._initial_value2 = initial_value2
        self._accrual_factor = accrual_factor

    def name(self) -> str:
        return "DoubleStickyRatchetPayoff"

    def description(self) -> str:
        # C++ parity: stickyratchet.cpp:43-47 — description() is name().
        return self.name()

    def __call__(self, price: float) -> float:
        """Payoff at ``price`` (the forward).

        C++ parity: stickyratchet.cpp:25-41. The C++ parameter is named
        ``forward``; the base class calls it ``price``, and pyright strict
        requires override parameter names to match the base, so ``price``
        it is.
        """
        # C++ parity: the two guards are checked on every evaluation, not in
        # the constructor — stickyratchet.cpp:26-29.
        qassert.require(
            abs(self._type1) == 1.0 or self._type1 == 0.0,
            "unknown/illegal type1 value (only 0.0 and +/-1,0 are allowed))",
        )
        qassert.require(
            abs(self._type2) == 1.0 or self._type2 == 0.0,
            "unknown/illegal type2 value(only 0.0 and +/-1,0 are allowed)",
        )
        swaplet = self._gearing3 * price + self._spread3
        eff_strike1 = self._gearing1 * self._initial_value1 + self._spread1
        eff_strike2 = self._gearing2 * self._initial_value2 + self._spread2
        eff_strike3 = self._type1 * self._type2 * max(self._type2 * (swaplet - eff_strike2), 0.0)
        return self._accrual_factor * (
            swaplet - self._type1 * max(self._type1 * (swaplet - eff_strike1), eff_strike3)
        )

    # --- inspectors (not in C++, which leaves the members protected; added
    #     so tests can assert an argument reached the slot it configures) ---

    def type1(self) -> float:
        return self._type1

    def type2(self) -> float:
        return self._type2

    def gearings(self) -> tuple[float, float, float]:
        return (self._gearing1, self._gearing2, self._gearing3)

    def spreads(self) -> tuple[float, float, float]:
        return (self._spread1, self._spread2, self._spread3)

    def initial_values(self) -> tuple[float, float]:
        return (self._initial_value1, self._initial_value2)

    def accrual_factor(self) -> float:
        return self._accrual_factor


class RatchetPayoff(DoubleStickyRatchetPayoff):
    """Ratchet payoff (single option). C++ parity: stickyratchet.hpp:60-76."""

    def __init__(
        self,
        gearing1: float,
        gearing2: float,
        spread1: float,
        spread2: float,
        initial_value: float,
        accrual_factor: float,
    ) -> None:
        super().__init__(
            -1.0,
            0.0,
            gearing1,
            0.0,
            gearing2,
            spread1,
            0.0,
            spread2,
            initial_value,
            0.0,
            accrual_factor,
        )

    def name(self) -> str:
        return "Ratchet"


class StickyPayoff(DoubleStickyRatchetPayoff):
    """Sticky payoff (single option). C++ parity: stickyratchet.hpp:78-93."""

    def __init__(
        self,
        gearing1: float,
        gearing2: float,
        spread1: float,
        spread2: float,
        initial_value: float,
        accrual_factor: float,
    ) -> None:
        super().__init__(
            +1.0,
            0.0,
            gearing1,
            0.0,
            gearing2,
            spread1,
            0.0,
            spread2,
            initial_value,
            0.0,
            accrual_factor,
        )

    def name(self) -> str:
        return "Sticky"


class RatchetMaxPayoff(DoubleStickyRatchetPayoff):
    """RatchetMax payoff (double option). C++ parity: stickyratchet.hpp:95-112."""

    def __init__(
        self,
        gearing1: float,
        gearing2: float,
        gearing3: float,
        spread1: float,
        spread2: float,
        spread3: float,
        initial_value1: float,
        initial_value2: float,
        accrual_factor: float,
    ) -> None:
        super().__init__(
            -1.0,
            -1.0,
            gearing1,
            gearing2,
            gearing3,
            spread1,
            spread2,
            spread3,
            initial_value1,
            initial_value2,
            accrual_factor,
        )

    def name(self) -> str:
        return "RatchetMax"


class RatchetMinPayoff(DoubleStickyRatchetPayoff):
    """RatchetMin payoff (double option). C++ parity: stickyratchet.hpp:114-131."""

    def __init__(
        self,
        gearing1: float,
        gearing2: float,
        gearing3: float,
        spread1: float,
        spread2: float,
        spread3: float,
        initial_value1: float,
        initial_value2: float,
        accrual_factor: float,
    ) -> None:
        super().__init__(
            -1.0,
            +1.0,
            gearing1,
            gearing2,
            gearing3,
            spread1,
            spread2,
            spread3,
            initial_value1,
            initial_value2,
            accrual_factor,
        )

    def name(self) -> str:
        return "RatchetMin"


class StickyMaxPayoff(DoubleStickyRatchetPayoff):
    """StickyMax payoff (double option). C++ parity: stickyratchet.hpp:133-150."""

    def __init__(
        self,
        gearing1: float,
        gearing2: float,
        gearing3: float,
        spread1: float,
        spread2: float,
        spread3: float,
        initial_value1: float,
        initial_value2: float,
        accrual_factor: float,
    ) -> None:
        super().__init__(
            +1.0,
            -1.0,
            gearing1,
            gearing2,
            gearing3,
            spread1,
            spread2,
            spread3,
            initial_value1,
            initial_value2,
            accrual_factor,
        )

    def name(self) -> str:
        return "StickyMax"


class StickyMinPayoff(DoubleStickyRatchetPayoff):
    """StickyMin payoff (double option). C++ parity: stickyratchet.hpp:152-169."""

    def __init__(
        self,
        gearing1: float,
        gearing2: float,
        gearing3: float,
        spread1: float,
        spread2: float,
        spread3: float,
        initial_value1: float,
        initial_value2: float,
        accrual_factor: float,
    ) -> None:
        super().__init__(
            +1.0,
            +1.0,
            gearing1,
            gearing2,
            gearing3,
            spread1,
            spread2,
            spread3,
            initial_value1,
            initial_value2,
            accrual_factor,
        )

    def name(self) -> str:
        return "StickyMin"


__all__ = [
    "DoubleStickyRatchetPayoff",
    "RatchetMaxPayoff",
    "RatchetMinPayoff",
    "RatchetPayoff",
    "StickyMaxPayoff",
    "StickyMinPayoff",
    "StickyPayoff",
]
