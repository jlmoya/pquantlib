"""Thin, typed wrappers over :mod:`pquantlib`.

Every function here takes plain Python inputs and returns plain data
(dataclasses, floats, numpy arrays) so the Streamlit pages never touch the
library directly and results are trivially cacheable. This is the single
source of truth for how the showcase drives PQuantLib.
"""

from __future__ import annotations
