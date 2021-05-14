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
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa  # noqa: F401
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix  # noqa: F401
from torchmetrics.functional.classification.dice import dice_score  # noqa: F401
from torchmetrics.functional.classification.f_beta import f1, fbeta  # noqa: F401
from torchmetrics.functional.classification.hamming_distance import hamming_distance  # noqa: F401
from torchmetrics.functional.classification.hinge import hinge  # noqa: F401
from torchmetrics.functional.classification.iou import iou  # noqa: F401
from torchmetrics.functional.classification.matthews_corrcoef import matthews_corrcoef  # noqa: F401
from torchmetrics.functional.classification.precision_recall import precision, precision_recall, recall  # noqa: F401
from torchmetrics.functional.classification.precision_recall_curve import precision_recall_curve  # noqa: F401
from torchmetrics.functional.classification.roc import roc  # noqa: F401
from torchmetrics.functional.classification.specificity import specificity  # noqa: F401
from torchmetrics.functional.classification.stat_scores import stat_scores  # noqa: F401
from torchmetrics.functional.image_gradients import image_gradients  # noqa: F401
from torchmetrics.functional.nlp import bleu_score  # noqa: F401
from torchmetrics.functional.regression.explained_variance import explained_variance  # noqa: F401
from torchmetrics.functional.regression.mean_absolute_error import mean_absolute_error  # noqa: F401
from torchmetrics.functional.regression.mean_absolute_percentage_error import (  # noqa: F401
    mean_absolute_percentage_error,
)
from torchmetrics.functional.regression.mean_relative_error import mean_relative_error  # noqa: F401
from torchmetrics.functional.regression.mean_squared_error import mean_squared_error  # noqa: F401
from torchmetrics.functional.regression.mean_squared_log_error import mean_squared_log_error  # noqa: F401
from torchmetrics.functional.regression.pearson import pearson_corrcoef  # noqa: F401
from torchmetrics.functional.regression.psnr import psnr  # noqa: F401
from torchmetrics.functional.regression.r2score import r2score  # noqa: F401
from torchmetrics.functional.regression.spearman import spearman_corrcoef  # noqa: F401
from torchmetrics.functional.regression.ssim import ssim  # noqa: F401
from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision  # noqa: F401
from torchmetrics.functional.retrieval.fall_out import retrieval_fall_out  # noqa: F401
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg  # noqa: F401
from torchmetrics.functional.retrieval.precision import retrieval_precision  # noqa: F401
from torchmetrics.functional.retrieval.recall import retrieval_recall  # noqa: F401
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank  # noqa: F401
from torchmetrics.functional.self_supervised import embedding_similarity  # noqa: F401
