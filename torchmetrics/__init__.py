"""Root package info."""
import logging as __logging
import os

from torchmetrics.info import (  # noqa: F401
    __author__,
    __author_email__,
    __copyright__,
    __docs__,
    __homepage__,
    __license__,
    __version__,
)

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics.classification import (  # noqa: F401 E402
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
    Hinge,
    IoU,
    MatthewsCorrcoef,
    Precision,
    PrecisionRecallCurve,
    Recall,
    StatScores,
)
from torchmetrics.collections import MetricCollection  # noqa: F401 E402
from torchmetrics.metric import Metric  # noqa: F401 E402
from torchmetrics.regression import (  # noqa: F401 E402
    PSNR,
    SSIM,
    ExplainedVariance,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    PearsonCorrcoef,
    R2Score,
)
from torchmetrics.retrieval import (  # noqa: F401 E402
    RetrievalFallOut,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)
from torchmetrics.wrappers import BootStrapper  # noqa: F401 E402
