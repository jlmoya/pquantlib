"""AlphaForm — the alpha(t) functional form for coterminal calibration.

# C++ parity: ql/models/marketmodels/models/alphaform.hpp (v1.42.1).

The alpha form supplies a per-rate multiplier ``alpha(i)`` parametrised by a
single scalar ``alpha`` that the AlphaFinder root-solves so that the modified
swap-rate vols reprice a target caplet variance while staying as
time-homogeneous as possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AlphaForm(ABC):
    """Abstract alpha(t) functional form.

    # C++ parity: AlphaForm (pure-virtual ``operator()(Integer)`` + ``setAlpha``).
    """

    @abstractmethod
    def __call__(self, i: int) -> float:
        """Evaluate the form at rate index ``i``.

        # C++ parity: AlphaForm::operator()(Integer i).
        """
        ...

    @abstractmethod
    def set_alpha(self, alpha: float) -> None:
        """Set the single scalar parameter.

        # C++ parity: AlphaForm::setAlpha(Real alpha).
        """
        ...
