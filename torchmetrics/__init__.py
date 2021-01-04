"""Root package info."""

__version__ = '0.0.0rc'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'teddy@pytorchlightning.ai'
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
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pytorch_lightning.metrics.metric import Metric

from pytorch_lightning.metrics.classification import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
    PrecisionRecallCurve,
    AveragePrecision,
    ROC,
    FBeta,
    F1,
)

from pytorch_lightning.metrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredLogError,
    ExplainedVariance,
    PSNR,
    SSIM,
)
