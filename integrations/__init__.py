import operator
import os

from torchmetrics.utilities.imports import _compare_version

_INTEGRATION_ROOT = os.path.realpath(os.path.dirname(__file__))
_PACKAGE_ROOT = os.path.dirname(_INTEGRATION_ROOT)
_PATH_DATASETS = os.path.join(_PACKAGE_ROOT, "datasets")

_LIGHTNING_GREATER_EQUAL_1_3 = _compare_version("pytorch_lightning", operator.ge, "1.3.0")
