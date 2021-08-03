r"""Root package info."""
import logging as __logging
import os

from torchmetrics.__about__ import *  # noqa: F401, F403

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics.audio import PIT, SI_SDR, SI_SNR, SNR  # noqa: E402, F401
from torchmetrics.average import AverageMeter  # noqa: E402, F401
from torchmetrics.classification import (  # noqa: E402, F401
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
from torchmetrics.collections import MetricCollection  # noqa: E402, F401
from torchmetrics.image import FID, IS, KID, PSNR, SSIM  # noqa: E402, F401
from torchmetrics.metric import Metric  # noqa: E402, F401
from torchmetrics.regression import (  # noqa: E402, F401
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
)
from torchmetrics.retrieval import (  # noqa: E402, F401
    RetrievalFallOut,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)
from torchmetrics.text import WER, BERTScore, BLEUScore, ROUGEScore  # noqa: E402, F401
from torchmetrics.wrappers import BootStrapper  # noqa: E402, F401
