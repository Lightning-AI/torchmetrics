# Copyright The Lightning team.
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
from torchmetrics.functional.classification.accuracy import (
    accuracy,
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
)
from torchmetrics.functional.classification.auroc import auroc, binary_auroc, multiclass_auroc, multilabel_auroc
from torchmetrics.functional.classification.average_precision import (
    average_precision,
    binary_average_precision,
    multiclass_average_precision,
    multilabel_average_precision,
)
from torchmetrics.functional.classification.calibration_error import (
    binary_calibration_error,
    calibration_error,
    multiclass_calibration_error,
)
from torchmetrics.functional.classification.cohen_kappa import binary_cohen_kappa, cohen_kappa, multiclass_cohen_kappa
from torchmetrics.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from torchmetrics.functional.classification.dice import dice
from torchmetrics.functional.classification.exact_match import (
    exact_match,
    multiclass_exact_match,
    multilabel_exact_match,
)
from torchmetrics.functional.classification.f_beta import (
    binary_f1_score,
    binary_fbeta_score,
    f1_score,
    fbeta_score,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multilabel_f1_score,
    multilabel_fbeta_score,
)
from torchmetrics.functional.classification.group_fairness import (
    binary_fairness,
    binary_groups_stat_rates,
    demographic_parity,
    equal_opportunity,
)
from torchmetrics.functional.classification.hamming import (
    binary_hamming_distance,
    hamming_distance,
    multiclass_hamming_distance,
    multilabel_hamming_distance,
)
from torchmetrics.functional.classification.hinge import binary_hinge_loss, hinge_loss, multiclass_hinge_loss
from torchmetrics.functional.classification.jaccard import (
    binary_jaccard_index,
    jaccard_index,
    multiclass_jaccard_index,
    multilabel_jaccard_index,
)
from torchmetrics.functional.classification.matthews_corrcoef import (
    binary_matthews_corrcoef,
    matthews_corrcoef,
    multiclass_matthews_corrcoef,
    multilabel_matthews_corrcoef,
)
from torchmetrics.functional.classification.precision_fixed_recall import (
    binary_precision_at_fixed_recall,
    multiclass_precision_at_fixed_recall,
    multilabel_precision_at_fixed_recall,
)
from torchmetrics.functional.classification.precision_recall import (
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
    precision,
    recall,
)
from torchmetrics.functional.classification.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
    precision_recall_curve,
)
from torchmetrics.functional.classification.ranking import (
    multilabel_coverage_error,
    multilabel_ranking_average_precision,
    multilabel_ranking_loss,
)
from torchmetrics.functional.classification.recall_fixed_precision import (
    binary_recall_at_fixed_precision,
    multiclass_recall_at_fixed_precision,
    multilabel_recall_at_fixed_precision,
)
from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc, roc
from torchmetrics.functional.classification.specificity import (
    binary_specificity,
    multiclass_specificity,
    multilabel_specificity,
    specificity,
)
from torchmetrics.functional.classification.specificity_sensitivity import (
    binary_specificity_at_sensitivity,
    multiclass_specificity_at_sensitivity,
    multilabel_specificity_at_sensitivity,
    specicity_at_sensitivity,
)
from torchmetrics.functional.classification.stat_scores import (
    binary_stat_scores,
    multiclass_stat_scores,
    multilabel_stat_scores,
    stat_scores,
)

__all__ = [
    "accuracy",
    "binary_accuracy",
    "multiclass_accuracy",
    "multilabel_accuracy",
    "auroc",
    "binary_auroc",
    "multiclass_auroc",
    "multilabel_auroc",
    "average_precision",
    "binary_average_precision",
    "multiclass_average_precision",
    "multilabel_average_precision",
    "binary_calibration_error",
    "calibration_error",
    "multiclass_calibration_error",
    "binary_cohen_kappa",
    "cohen_kappa",
    "multiclass_cohen_kappa",
    "binary_confusion_matrix",
    "confusion_matrix",
    "multiclass_confusion_matrix",
    "multilabel_confusion_matrix",
    "dice",
    "exact_match",
    "multiclass_exact_match",
    "multilabel_exact_match",
    "binary_f1_score",
    "binary_fbeta_score",
    "f1_score",
    "fbeta_score",
    "multiclass_f1_score",
    "multiclass_fbeta_score",
    "multilabel_f1_score",
    "multilabel_fbeta_score",
    "binary_fairness",
    "binary_groups_stat_rates",
    "demographic_parity",
    "equal_opportunity",
    "binary_hamming_distance",
    "hamming_distance",
    "multiclass_hamming_distance",
    "multilabel_hamming_distance",
    "binary_hinge_loss",
    "hinge_loss",
    "multiclass_hinge_loss",
    "binary_jaccard_index",
    "jaccard_index",
    "multiclass_jaccard_index",
    "multilabel_jaccard_index",
    "binary_matthews_corrcoef",
    "matthews_corrcoef",
    "multiclass_matthews_corrcoef",
    "multilabel_matthews_corrcoef",
    "binary_precision",
    "binary_recall",
    "multiclass_precision",
    "multiclass_recall",
    "multilabel_precision",
    "multilabel_recall",
    "precision",
    "recall",
    "binary_precision_recall_curve",
    "multiclass_precision_recall_curve",
    "multilabel_precision_recall_curve",
    "precision_recall_curve",
    "multilabel_coverage_error",
    "multilabel_ranking_average_precision",
    "multilabel_ranking_loss",
    "binary_recall_at_fixed_precision",
    "multiclass_recall_at_fixed_precision",
    "multilabel_recall_at_fixed_precision",
    "binary_roc",
    "multiclass_roc",
    "multilabel_roc",
    "roc",
    "binary_specificity",
    "multiclass_specificity",
    "multilabel_specificity",
    "specificity",
    "binary_specificity_at_sensitivity",
    "multiclass_specificity_at_sensitivity",
    "multilabel_specificity_at_sensitivity",
    "specicity_at_sensitivity",
    "binary_stat_scores",
    "multiclass_stat_scores",
    "multilabel_stat_scores",
    "stat_scores",
    "binary_precision_at_fixed_recall",
    "multilabel_precision_at_fixed_recall",
    "multiclass_precision_at_fixed_recall",
]
