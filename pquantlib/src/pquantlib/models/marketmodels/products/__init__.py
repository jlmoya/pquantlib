"""MarketModels (BGM) MultiProduct framework + concrete products (W11).

# C++ parity: ql/models/marketmodels/products/ (v1.42.1).
"""

from __future__ import annotations

from pquantlib.models.marketmodels.products.call_specified_multiproduct import (
    CallSpecifiedMultiProduct,
    ExerciseStrategy,
)
from pquantlib.models.marketmodels.products.cash_rebate import MarketModelCashRebate
from pquantlib.models.marketmodels.products.composite_product import (
    MarketModelComposite,
)
from pquantlib.models.marketmodels.products.exercise_adapter import (
    ExerciseAdapter,
    MarketModelExerciseValue,
)
from pquantlib.models.marketmodels.products.multiproduct_composite import (
    MultiProductComposite,
)
from pquantlib.models.marketmodels.products.multiproduct_multistep import (
    MultiProductMultiStep,
)
from pquantlib.models.marketmodels.products.multiproduct_onestep import (
    MultiProductOneStep,
)
from pquantlib.models.marketmodels.products.multistep_coterminal_swaps import (
    MultiStepCoterminalSwaps,
)
from pquantlib.models.marketmodels.products.multistep_coterminal_swaptions import (
    MultiStepCoterminalSwaptions,
)
from pquantlib.models.marketmodels.products.multistep_forwards import MultiStepForwards
from pquantlib.models.marketmodels.products.multistep_nothing import MultiStepNothing
from pquantlib.models.marketmodels.products.multistep_optionlets import (
    MultiStepOptionlets,
)
from pquantlib.models.marketmodels.products.multistep_swap import MultiStepSwap
from pquantlib.models.marketmodels.products.onestep_forwards import OneStepForwards
from pquantlib.models.marketmodels.products.onestep_optionlets import OneStepOptionlets
from pquantlib.models.marketmodels.products.single_product_composite import (
    SingleProductComposite,
)

__all__ = [
    "CallSpecifiedMultiProduct",
    "ExerciseAdapter",
    "ExerciseStrategy",
    "MarketModelCashRebate",
    "MarketModelComposite",
    "MarketModelExerciseValue",
    "MultiProductComposite",
    "MultiProductMultiStep",
    "MultiProductOneStep",
    "MultiStepCoterminalSwaps",
    "MultiStepCoterminalSwaptions",
    "MultiStepForwards",
    "MultiStepNothing",
    "MultiStepOptionlets",
    "MultiStepSwap",
    "OneStepForwards",
    "OneStepOptionlets",
    "SingleProductComposite",
]
