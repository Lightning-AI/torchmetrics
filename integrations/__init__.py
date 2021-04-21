import operator

from torchmetrics.utilities.imports import _compare_version

_LIGHTNING_GREATER_EQUAL_1_3 = _compare_version("pytorch_lightning", operator.ge, "1.3.0")
