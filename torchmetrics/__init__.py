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
from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric  # noqa: E402
from torchmetrics.audio import PESQ, PIT, SI_SDR, SI_SNR, SNR, STOI  # noqa: E402
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
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalRPrecision,
)
from torchmetrics.text import WER, BERTScore, BLEUScore, CharErrorRate, ROUGEScore, SacreBLEUScore  # noqa: E402
from torchmetrics.wrappers import BootStrapper, MetricTracker, MultioutputWrapper  # noqa: E402

__all__ = [
    "functional",
    "Accuracy",
    "AUC",
    "AUROC",
    "AveragePrecision",
    "BinnedAveragePrecision",
    "BinnedPrecisionRecallCurve",
    "BinnedRecallAtFixedPrecision",
    "BERTScore",
    "BLEUScore",
    "BootStrapper",
    "CalibrationError",
    "CatMetric",
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
    "MaxMetric",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanMetric",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinMetric",
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
    "RetrievalHitRate",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalRecall",
    "RetrievalRPrecision",
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
    "STOI",
    "SumMetric",
    "SymmetricMeanAbsolutePercentageError",
    "WER",
    "CharErrorRate",
]
