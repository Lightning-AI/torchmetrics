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
from torchmetrics.functional.audio.pit import pit, pit_permutate
from torchmetrics.functional.audio.si_sdr import si_sdr
from torchmetrics.functional.audio.si_snr import si_snr
from torchmetrics.functional.audio.snr import snr
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional.classification.auc import auc
from torchmetrics.functional.classification.auroc import auroc
from torchmetrics.functional.classification.average_precision import average_precision
from torchmetrics.functional.classification.calibration_error import calibration_error
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix
from torchmetrics.functional.classification.dice import dice_score
from torchmetrics.functional.classification.f_beta import f1, fbeta
from torchmetrics.functional.classification.hamming_distance import hamming_distance
from torchmetrics.functional.classification.hinge import hinge
from torchmetrics.functional.classification.iou import iou
from torchmetrics.functional.classification.kl_divergence import kl_divergence, kldivergence
from torchmetrics.functional.classification.matthews_corrcoef import matthews_corrcoef
from torchmetrics.functional.classification.precision_recall import precision, precision_recall, recall
from torchmetrics.functional.classification.precision_recall_curve import precision_recall_curve
from torchmetrics.functional.classification.roc import roc
from torchmetrics.functional.classification.specificity import specificity
from torchmetrics.functional.classification.stat_scores import stat_scores
from torchmetrics.functional.image.gradients import image_gradients
from torchmetrics.functional.image.psnr import psnr
from torchmetrics.functional.image.ssim import ssim
from torchmetrics.functional.regression.cosine_similarity import cosine_similarity
from torchmetrics.functional.regression.explained_variance import explained_variance
from torchmetrics.functional.regression.mean_absolute_error import mean_absolute_error
from torchmetrics.functional.regression.mean_absolute_percentage_error import mean_absolute_percentage_error
from torchmetrics.functional.regression.mean_squared_error import mean_squared_error
from torchmetrics.functional.regression.mean_squared_log_error import mean_squared_log_error
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression.r2 import r2_score
from torchmetrics.functional.regression.r2score import r2score
from torchmetrics.functional.regression.spearman import spearman_corrcoef
from torchmetrics.functional.regression.symmetric_mean_absolute_percentage_error import (
    symmetric_mean_absolute_percentage_error,
)
from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.functional.retrieval.fall_out import retrieval_fall_out
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics.functional.retrieval.precision import retrieval_precision
from torchmetrics.functional.retrieval.recall import retrieval_recall
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank
from torchmetrics.functional.self_supervised import embedding_similarity
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.wer import wer

__all__ = [
    "accuracy",
    "auc",
    "auroc",
    "average_precision",
    "bert_score",
    "bleu_score",
    "calibration_error",
    "cohen_kappa",
    "confusion_matrix",
    "cosine_similarity",
    "dice_score",
    "embedding_similarity",
    "explained_variance",
    "f1",
    "fbeta",
    "hamming_distance",
    "hinge",
    "image_gradients",
    "iou",
    "kl_divergence",
    "kldivergence",
    "matthews_corrcoef",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "pearson_corrcoef",
    "pit",
    "pit_permutate",
    "precision",
    "precision_recall",
    "precision_recall_curve",
    "psnr",
    "r2_score",
    "r2score",
    "recall",
    "retrieval_average_precision",
    "retrieval_fall_out",
    "retrieval_normalized_dcg",
    "retrieval_precision",
    "retrieval_recall",
    "retrieval_reciprocal_rank",
    "roc",
    "rouge_score",
    "si_sdr",
    "si_snr",
    "snr",
    "spearman_corrcoef",
    "specificity",
    "ssim",
    "stat_scores",
    "symmetric_mean_absolute_percentage_error",
    "wer",
]
