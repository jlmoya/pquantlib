"""QuantoOptionResults — quanto-specific option result fields.

# C++ parity: ql/instruments/quantovanillaoption.hpp:33-45 (v1.43) —
# ``template<class ResultsType> class QuantoOptionResults``.

C++ makes this a template mixing the three quanto sensitivities into any
results type (``OneAssetOption::results`` for ``QuantoVanillaOption``,
``DoubleBarrierOption::results`` for ``QuantoDoubleBarrierOption``). Both
instantiations end up on top of ``OneAssetOption::results``, which is the
single Python results class, so the port is a plain subclass rather than a
generic.

``QuantoVanillaOption`` itself, which shares this header, is out of scope
for the barrier wave and is not ported here.
"""

from __future__ import annotations

from pquantlib.instruments.one_asset_option import OneAssetOptionResults


class QuantoOptionResults(OneAssetOptionResults):
    """One-asset option results plus qvega / qrho / qlambda.

    # C++ parity: ``QuantoOptionResults<ResultsType>``
    # (quantovanillaoption.hpp:33-45).
    """

    def __init__(self) -> None:
        super().__init__()
        self.qvega: float | None = None
        self.qrho: float | None = None
        self.qlambda: float | None = None

    def reset(self) -> None:
        """# C++ parity: ``QuantoOptionResults::reset``
        # (quantovanillaoption.hpp:36-40)."""
        super().reset()
        self.qvega = None
        self.qrho = None
        self.qlambda = None


__all__ = ["QuantoOptionResults"]
