"""Root package info."""
import logging as __logging
import os

__version__ = '0.2.0'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'name@pytorchlightning.ai'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2020-2021, {__author__}.'
__homepage__ = 'https://github.com/PyTorchLightning/metrics'
__docs__ = "PyTorch native Metrics"
__long_doc__ = """
Torchmetrics is a metrics API created for easy metric development and usage in both PyTorch and
[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/). It was originally a part of
Pytorch Lightning, but got split off so users could take advantage of the large collection of metrics
implemented without having to install Pytorch Lightning (even though we would love for you to try it out).
We currently have around 25+ metrics implemented and we continuously is adding more metrics, both within
already covered domains (classification, regression ect.) but also new domains (object detection ect.).
We make sure that all our metrics are rigorously tested such that you can trust them.
"""

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    _ = None if __LIGHTNING_SETUP__ else None
except NameError:
    __LIGHTNING_SETUP__: bool = False

if __LIGHTNING_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(f'Partial import of `{__name__}` during the build process.\n')  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:

    from torchmetrics.classification import (  # noqa: F401
        AUC,
        AUROC,
        F1,
        ROC,
        Accuracy,
        AveragePrecision,
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
