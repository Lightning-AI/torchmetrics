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
from torchmetrics.functional.average_precision import average_precision
from torchmetrics.functional.classification import (
    accuracy,
    auc,
    auroc,
    dice_score,
    f1_score,
    fbeta_score,
    get_num_classes,
    iou,
    multiclass_auroc,
    precision,
    precision_recall,
    recall,
    stat_scores,
    stat_scores_multiple_classes,
    to_categorical,
    to_onehot,
)
from torchmetrics.functional.confusion_matrix import confusion_matrix
# TODO: unify metrics between class and functional, add below
from torchmetrics.functional.explained_variance import explained_variance
from torchmetrics.functional.f_beta import fbeta, f1
from torchmetrics.functional.mean_absolute_error import mean_absolute_error
from torchmetrics.functional.mean_squared_error import mean_squared_error
from torchmetrics.functional.mean_squared_log_error import mean_squared_log_error
from torchmetrics.functional.nlp import bleu_score
from torchmetrics.functional.precision_recall_curve import precision_recall_curve
from torchmetrics.functional.psnr import psnr
from torchmetrics.functional.roc import roc
from torchmetrics.functional.self_supervised import embedding_similarity
from torchmetrics.functional.ssim import ssim
