"""MarketModels (BGM) MultiProduct framework + concrete products (W11).

# C++ parity: ql/models/marketmodels/products/ (v1.42.1).
"""

from __future__ import annotations

from pquantlib.models.marketmodels.products.cash_rebate import MarketModelCashRebate
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

__all__ = [
    "MarketModelCashRebate",
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
]
