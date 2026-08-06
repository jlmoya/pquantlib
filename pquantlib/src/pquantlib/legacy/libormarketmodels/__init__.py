"""LIBOR market model (BGM / LFM) — volatility, correlation, process, model.

# C++ parity: ql/legacy/libormarketmodels/ (v1.43).

One module per C++ header:

===================================  ==========================================
C++ header                           Python module
===================================  ==========================================
``lmvolmodel.hpp``                   :mod:`.lm_vol_model`
``lmfixedvolmodel.hpp``              :mod:`.lm_fixed_vol_model`
``lmlinexpvolmodel.hpp``             :mod:`.lm_lin_exp_vol_model`
``lmextlinexpvolmodel.hpp``          :mod:`.lm_ext_lin_exp_vol_model`
``lmconstwrappervolmodel.hpp``       :mod:`.lm_const_wrapper_vol_model`
``lmcorrmodel.hpp``                  :mod:`.lm_corr_model`
``lmexpcorrmodel.hpp``               :mod:`.lm_exp_corr_model`
``lmlinexpcorrmodel.hpp``            :mod:`.lm_lin_exp_corr_model`
``lmconstwrappercorrmodel.hpp``      :mod:`.lm_const_wrapper_corr_model`
``lfmcovarparam.hpp``                :mod:`.lfm_covar_param`
``lfmcovarproxy.hpp``                :mod:`.lfm_covar_proxy`
``lfmhullwhiteparam.hpp``            :mod:`.lfm_hull_white_param`
``lfmprocess.hpp``                   :mod:`.lfm_process`
``liborforwardmodel.hpp``            :mod:`.libor_forward_model`
``lfmswaptionengine.hpp``            :mod:`.lfm_swaption_engine`
===================================  ==========================================

Two C++ names are overloaded on arity and cannot be in Python; the scalar
variants take a ``_scalar`` suffix, following
``OneFactorAffineModel.discount_bond_scalar``:

- ``volatility(Size, Time, Array)`` -> ``volatility_scalar``
- ``correlation(Size, Size, Time, Array)`` -> ``correlation_scalar``
- ``LfmCovarianceProxy::integratedCovariance(Size, Size, Time, Array)`` ->
  ``integrated_covariance_scalar`` (the matrix-valued
  ``integratedCovariance(Time, Array)`` keeps the plain name).
"""

from __future__ import annotations

from pquantlib.legacy.libormarketmodels.lfm_covar_param import (
    LfmCovarianceParameterization,
)
from pquantlib.legacy.libormarketmodels.lfm_covar_proxy import LfmCovarianceProxy
from pquantlib.legacy.libormarketmodels.lfm_hull_white_param import (
    LfmHullWhiteParameterization,
)
from pquantlib.legacy.libormarketmodels.lfm_process import LiborForwardModelProcess
from pquantlib.legacy.libormarketmodels.lfm_swaption_engine import LfmSwaptionEngine
from pquantlib.legacy.libormarketmodels.libor_forward_model import LiborForwardModel
from pquantlib.legacy.libormarketmodels.lm_const_wrapper_corr_model import (
    LmConstWrapperCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_const_wrapper_vol_model import (
    LmConstWrapperVolatilityModel,
)
from pquantlib.legacy.libormarketmodels.lm_corr_model import LmCorrelationModel
from pquantlib.legacy.libormarketmodels.lm_exp_corr_model import (
    LmExponentialCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_ext_lin_exp_vol_model import (
    LmExtLinearExponentialVolModel,
)
from pquantlib.legacy.libormarketmodels.lm_fixed_vol_model import LmFixedVolatilityModel
from pquantlib.legacy.libormarketmodels.lm_lin_exp_corr_model import (
    LmLinearExponentialCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_lin_exp_vol_model import (
    LmLinearExponentialVolatilityModel,
)
from pquantlib.legacy.libormarketmodels.lm_vol_model import LmVolatilityModel

__all__ = [
    "LfmCovarianceParameterization",
    "LfmCovarianceProxy",
    "LfmHullWhiteParameterization",
    "LfmSwaptionEngine",
    "LiborForwardModel",
    "LiborForwardModelProcess",
    "LmConstWrapperCorrelationModel",
    "LmConstWrapperVolatilityModel",
    "LmCorrelationModel",
    "LmExponentialCorrelationModel",
    "LmExtLinearExponentialVolModel",
    "LmFixedVolatilityModel",
    "LmLinearExponentialCorrelationModel",
    "LmLinearExponentialVolatilityModel",
    "LmVolatilityModel",
]
