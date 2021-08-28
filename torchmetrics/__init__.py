r"""Root package info."""
import logging as __logging
import os

from torchmetrics.__about__ import *  # noqa: F403

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics.average import AverageMeter  # noqa: E402
from torchmetrics.collections import MetricCollection  # noqa: E402
from torchmetrics.image import FID, IS, KID, LPIPS, PSNR, SSIM  # noqa: E402
from torchmetrics.metric import Metric  # noqa: E402

__all__ = [
    "AverageMeter",
    "AveragePrecision",
    "BinnedAveragePrecision",
    "BinnedPrecisionRecallCurve",
    "BinnedRecallAtFixedPrecision",
    "BERTScore",
    "BLEUScore",
    "BootStrapper",
    "CalibrationError",
    "CohenKappa",
    "ConfusionMatrix",
    "CosineSimilarity",
    "ExplainedVariance",
    "F1",
    "FBeta",
    "FID",
    "HammingDistance",
    "Hinge",
    "IoU",
    "IS",
    "KID",
    "KLDivergence",
    "LPIPS",
    "MatthewsCorrcoef",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "Metric",
    "MetricCollection",
]
