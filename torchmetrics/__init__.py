"""Root package info."""

__version__ = '0.1.0'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'name@pytorchlightning.ai'
__license__ = 'TBD'
__copyright__ = 'Copyright (c) 2020-2020, %s.' % __author__
__homepage__ = 'https://github.com/PyTorchLightning/torchmetrics'
__docs__ = "PyTorch Lightning Sample project."
__long_doc__ = """
What is it?
-----------
metrics...
"""

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
        Accuracy, AUC, AUROC, AveragePrecision, ConfusionMatrix, F1, FBeta,
        HammingDistance, IoU, Precision, PrecisionRecallCurve, Recall, ROC,
        StatScores,
    )
    from torchmetrics.metric import Metric, MetricCollection  # noqa: F401
    from torchmetrics.regression import (  # noqa: F401
        ExplainedVariance, MeanAbsoluteError, MeanSquaredError,
        MeanSquaredLogError, PSNR, R2Score, SSIM,
    )
