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
from torchmetrics.classification.accuracy import Accuracy  # noqa: F401
from torchmetrics.classification.auc import AUC  # noqa: F401
from torchmetrics.classification.auroc import AUROC  # noqa: F401
from torchmetrics.classification.avg_precision import AveragePrecision  # noqa: F401
from torchmetrics.classification.binned_precision_recall import BinnedAveragePrecision  # noqa: F401
from torchmetrics.classification.binned_precision_recall import BinnedPrecisionRecallCurve  # noqa: F401
from torchmetrics.classification.binned_precision_recall import BinnedRecallAtFixedPrecision  # noqa: F401
from torchmetrics.classification.calibration_error import CalibrationError  # noqa: F401
from torchmetrics.classification.cohen_kappa import CohenKappa  # noqa: F401
from torchmetrics.classification.confusion_matrix import ConfusionMatrix  # noqa: F401
from torchmetrics.classification.dice import Dice  # noqa: F401
from torchmetrics.classification.f_beta import F1Score, FBetaScore  # noqa: F401
from torchmetrics.classification.hamming import HammingDistance  # noqa: F401
from torchmetrics.classification.hinge import HingeLoss  # noqa: F401
from torchmetrics.classification.jaccard import JaccardIndex  # noqa: F401
from torchmetrics.classification.kl_divergence import KLDivergence  # noqa: F401
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef  # noqa: F401
from torchmetrics.classification.precision_recall import Precision, Recall  # noqa: F401
from torchmetrics.classification.precision_recall_curve import PrecisionRecallCurve  # noqa: F401
from torchmetrics.classification.ranking import (  # noqa: F401
    CoverageError,
    LabelRankingAveragePrecision,
    LabelRankingLoss,
)
from torchmetrics.classification.roc import ROC  # noqa: F401
from torchmetrics.classification.specificity import Specificity  # noqa: F401
from torchmetrics.classification.stat_scores import StatScores  # noqa: F401
