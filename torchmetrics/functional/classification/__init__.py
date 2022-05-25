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
from torchmetrics.functional.classification.accuracy import accuracy  # noqa: F401
from torchmetrics.functional.classification.auc import auc  # noqa: F401
from torchmetrics.functional.classification.auroc import auroc  # noqa: F401
from torchmetrics.functional.classification.average_precision import average_precision  # noqa: F401
from torchmetrics.functional.classification.calibration_error import calibration_error  # noqa: F401
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa  # noqa: F401
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix  # noqa: F401
from torchmetrics.functional.classification.dice import dice, dice_score  # noqa: F401
from torchmetrics.functional.classification.f_beta import f1_score, fbeta_score  # noqa: F401
from torchmetrics.functional.classification.hamming import hamming_distance  # noqa: F401
from torchmetrics.functional.classification.hinge import hinge_loss  # noqa: F401
from torchmetrics.functional.classification.jaccard import jaccard_index  # noqa: F401
from torchmetrics.functional.classification.kl_divergence import kl_divergence  # noqa: F401
from torchmetrics.functional.classification.matthews_corrcoef import matthews_corrcoef  # noqa: F401
from torchmetrics.functional.classification.precision_recall import precision, precision_recall, recall  # noqa: F401
from torchmetrics.functional.classification.precision_recall_curve import precision_recall_curve  # noqa: F401
from torchmetrics.functional.classification.ranking import (  # noqa: F401
    coverage_error,
    label_ranking_average_precision,
    label_ranking_loss,
)
from torchmetrics.functional.classification.roc import roc  # noqa: F401
from torchmetrics.functional.classification.specificity import specificity  # noqa: F401
from torchmetrics.functional.classification.stat_scores import stat_scores  # noqa: F401
