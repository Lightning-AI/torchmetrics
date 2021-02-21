"""Root package info."""

__version__ = '0.0.0rc1'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'name@pytorchlightning.ai'
__license__ = 'TBD'
__copyright__ = 'Copyright (c) 2020-2020, %s.' % __author__
__homepage__ = 'https://github.com/PyTorchLightning/torchmetrics'
__docs__ = "PyTorch Lightning Sample project."
__long_doc__ = """
What is it?
-----------
This is starter project template which shall simplify initial steps for each new PL project...

Except the implemented sections:
 - sample package
 - setting CI
 - setting Docs
"""

from torchmetrics.classification import (  # noqa: F401
    Accuracy,
    AUC,
    AUROC,
    AveragePrecision,
    ConfusionMatrix,
    F1,
    FBeta,
    HammingDistance,
    IoU,
    Precision,
    PrecisionRecallCurve,
    Recall,
    ROC,
    StatScores,
)
from torchmetrics.metric import Metric, MetricCollection  # noqa: F401
from torchmetrics.regression import (  # noqa: F401
    ExplainedVariance,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    PSNR,
    R2Score,
    SSIM,
)