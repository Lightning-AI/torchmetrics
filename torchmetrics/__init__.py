r"""Root package info."""
import logging as __logging
import os

from torchmetrics.__about__ import *  # noqa: F401, F403

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics import functional  # noqa: E402
from torchmetrics.audio import PESQ, PIT, SI_SDR, SI_SNR, SNR  # noqa: E402
from torchmetrics.average import AverageMeter  # noqa: E402
from torchmetrics.classification import (  # noqa: E402
    AUC,
    AUROC,
    F1,
    ROC,
    Accuracy,
    AveragePrecision,
    BinnedAveragePrecision,
    BinnedPrecisionRecallCurve,
    BinnedRecallAtFixedPrecision,
    CalibrationError,
    CohenKappa,
    ConfusionMatrix,
    FBeta,
    HammingDistance,
    Hinge,
    IoU,
    KLDivergence,
    MatthewsCorrcoef,
    Precision,
    PrecisionRecallCurve,
    Recall,
    Specificity,
    StatScores,
)
from torchmetrics.collections import MetricCollection  # noqa: E402
from torchmetrics.image import FID, IS, KID, LPIPS, PSNR, SSIM  # noqa: E402
from torchmetrics.metric import Metric  # noqa: E402
from torchmetrics.regression import (  # noqa: E402
    CosineSimilarity,
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    PearsonCorrcoef,
    R2Score,
    SpearmanCorrcoef,
    SymmetricMeanAbsolutePercentageError,
    TweedieDevianceScore,
)
from torchmetrics.retrieval import (  # noqa: E402
    RetrievalFallOut,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)
from torchmetrics.text import WER, BERTScore, BLEUScore, ROUGEScore, SacreBLEUScore  # noqa: E402
from torchmetrics.wrappers import BootStrapper, MetricTracker, MinMaxMetric, MultioutputWrapper  # noqa: E402

__all__ = [
    "functional",
    "Accuracy",
    "AUC",
    "AUROC",
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
    "TweedieDevianceScore",
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
    "MetricTracker",
    "MultioutputWrapper",
    "PearsonCorrcoef",
    "PESQ",
    "PIT",
    "Precision",
    "PrecisionRecallCurve",
    "PSNR",
    "R2Score",
    "Recall",
    "RetrievalFallOut",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalRecall",
    "ROC",
    "ROUGEScore",
    "SacreBLEUScore",
    "SI_SDR",
    "SI_SNR",
    "SNR",
    "SpearmanCorrcoef",
    "Specificity",
    "SSIM",
    "StatScores",
    "SymmetricMeanAbsolutePercentageError",
    "WER",
]
