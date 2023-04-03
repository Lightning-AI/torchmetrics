r"""Root package info."""
import logging as __logging
import os

from torchmetrics.__about__ import *  # noqa: F403

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics import functional  # noqa: E402
from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric  # noqa: E402
from torchmetrics.classification import (  # noqa: E402
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    CalibrationError,
    CohenKappa,
    ConfusionMatrix,
    Dice,
    ExactMatch,
    F1Score,
    FBetaScore,
    HammingDistance,
    HingeLoss,
    JaccardIndex,
    MatthewsCorrCoef,
    Precision,
    PrecisionRecallCurve,
    Recall,
    Specificity,
    StatScores,
)
from torchmetrics.collections import MetricCollection  # noqa: E402
from torchmetrics.metric import Metric  # noqa: E402
from torchmetrics.regression import (  # noqa: E402
    ConcordanceCorrCoef,
    CosineSimilarity,
    ExplainedVariance,
    KendallRankCorrCoef,
    KLDivergence,
    LogCoshError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    MinkowskiDistance,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
    SymmetricMeanAbsolutePercentageError,
    TweedieDevianceScore,
    WeightedMeanAbsolutePercentageError,
)

__all__ = [
    "Metric",
    "MetricCollection",
    "functional",
    "CatMetric",
    "MaxMetric",
    "MeanMetric",
    "MinMetric",
    "SumMetric",
    "AUROC",
    "ROC",
    "Accuracy",
    "AveragePrecision",
    "CalibrationError",
    "CohenKappa",
    "ConfusionMatrix",
    "Dice",
    "ExactMatch",
    "F1Score",
    "FBetaScore",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "MatthewsCorrCoef",
    "Precision",
    "PrecisionRecallCurve",
    "Recall",
    "Specificity",
    "StatScores",
    "ConcordanceCorrCoef",
    "CosineSimilarity",
    "ExplainedVariance",
    "KendallRankCorrCoef",
    "KLDivergence",
    "LogCoshError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "MinkowskiDistance",
    "PearsonCorrCoef",
    "R2Score",
    "SpearmanCorrCoef",
    "SymmetricMeanAbsolutePercentageError",
    "TweedieDevianceScore",
    "WeightedMeanAbsolutePercentageError",
]
