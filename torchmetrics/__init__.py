"""Root package info."""
import logging as __logging
import os

from torchmetrics.info import *

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics.classification import (  # noqa: F401
    AUC,
    AUROC,
    F1,
    ROC,
    Accuracy,
    AveragePrecision,
    CohenKappa,
    ConfusionMatrix,
    FBeta,
    HammingDistance,
    IoU,
    Precision,
    PrecisionRecallCurve,
    Recall,
    StatScores,
)
from torchmetrics.collections import MetricCollection  # noqa: F401
from torchmetrics.metric import Metric  # noqa: F401
from torchmetrics.regression import (  # noqa: F401
    PSNR,
    SSIM,
    ExplainedVariance,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    R2Score,
)
