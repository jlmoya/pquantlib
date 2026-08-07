"""GenericModelEngine — pricing engine bound to a model.

# C++ parity: ql/pricingengines/genericmodelengine.hpp (v1.43),
#             ``template<class ModelType, class ArgumentsType, class ResultsType>
#              class GenericModelEngine : public GenericEngine<ArgumentsType, ResultsType>``.

This is **not** a tag type. It carries state and behaviour: it holds the model
and registers itself as the model's observer, so that recalibrating the model
invalidates every instrument priced with the engine. Both C++ constructors
(``Handle<ModelType>`` and ``ext::shared_ptr<ModelType>``) do exactly that; the
Python port has one constructor because there is no ``Handle`` indirection —
the empty-``Handle`` default is spelled ``model=None``.

Derived engines only implement ``calculate()``, as in C++.
"""

from __future__ import annotations

from pquantlib.patterns.observer import Observable
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)


class GenericModelEngine[
    ModelT: Observable,
    ArgsT: PricingEngineArguments,
    ResultsT: PricingEngineResults,
](GenericEngine[ArgsT, ResultsT]):
    """Engine parameterised on a model, registered as that model's observer.

    # C++ parity: ``GenericModelEngine`` (genericmodelengine.hpp:38-52).
    # ``model_`` is a ``Handle<ModelType>``; an empty handle is ``None`` here,
    # and ``model_.empty()`` is ``self.model() is None``.
    """

    def __init__(self, arguments: ArgsT, results: ResultsT, model: ModelT | None = None) -> None:
        super().__init__(arguments, results)
        self._model: ModelT | None = model
        if model is not None:
            # C++ ``this->registerWith(model_);`` — an empty Handle registers
            # with nothing, which is what ``model is None`` means here.
            model.register_with(self)

    def model(self) -> ModelT | None:
        """The held model, or ``None`` for C++'s empty ``Handle``."""
        return self._model

    def set_model(self, model: ModelT | None) -> None:
        """Relink the model, re-registering as its observer.

        There is no C++ counterpart method — in C++ one links the ``Handle``
        instead, which fires the same notification. Provided so the Python
        engine can be re-pointed without rebuilding it.
        """
        if self._model is not None:
            self._model.unregister_with(self)
        self._model = model
        if model is not None:
            model.register_with(self)
        self.update()


__all__ = ["GenericModelEngine"]
