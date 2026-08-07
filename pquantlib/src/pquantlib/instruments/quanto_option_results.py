"""Re-export of :class:`QuantoOptionResults`.

C++ declares ``template <class ResultsType> class QuantoOptionResults`` inside
``ql/instruments/quantovanillaoption.hpp`` (line 35), so this port keeps the
definition next to the vanilla option, in
:mod:`pquantlib.instruments.quanto_vanilla_option`.

This module exists because a separate wave of the port introduced it as the
home of the class, and code written against that layout imports from here.
Both import paths must yield **the same class object**: the results carrier is
matched with ``isinstance`` in every quanto instrument's ``fetch_results``, so
two definitions of the same name meant a correctly-priced engine handed results
the instrument then rejected with "no quanto results returned from pricing
engine". Re-exporting, rather than defining a second class, is what makes the
two paths interchangeable.
"""

from __future__ import annotations

from pquantlib.instruments.quanto_vanilla_option import QuantoOptionResults

__all__ = ["QuantoOptionResults"]
