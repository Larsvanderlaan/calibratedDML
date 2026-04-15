"""Native Python interface for calibrated debiased machine learning.

Primary import path: ``calibrateddml``.
"""

from importlib.metadata import PackageNotFoundError, version

from .adaptive import AdaptiveCalibratedDML
from .calibration import (
    calibrate_outcome_regression,
    calibrate_propensity_scores,
    isoreg_with_xgboost,
)
from .core import CalibratedDML, calibrated_dml, calibrated_dml_from_nuisances
from .legacy import calibratedDML, calibratedDML_bootstrap

try:
    __version__ = version("calibratedDML")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "AdaptiveCalibratedDML",
    "CalibratedDML",
    "calibrated_dml",
    "calibrated_dml_from_nuisances",
    "calibratedDML",
    "calibratedDML_bootstrap",
    "calibrate_outcome_regression",
    "calibrate_propensity_scores",
    "isoreg_with_xgboost",
    "__version__",
]
