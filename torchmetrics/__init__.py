
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
        F,
        ROC,
        Accuracy,
        AveragePrecision,
        ConfusionMatrix,
        FBeta,
        Precision,
        PrecisionRecallCurve,
        Recall,
    )
    from torchmetrics.metric import Metric  # noqa: F401
    from torchmetrics.regression import (  # noqa: F401
        PSNR,
        SSIM,
        ExplainedVariance,
        MeanAbsoluteError,
        MeanSquaredError,
        MeanSquaredLogError,
    )1
