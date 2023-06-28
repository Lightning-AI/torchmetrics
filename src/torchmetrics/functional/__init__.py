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
from torchmetrics.functional.audio._deprecated import _permutation_invariant_training as permutation_invariant_training
from torchmetrics.functional.audio._deprecated import _pit_permutate as pit_permutate
from torchmetrics.functional.audio._deprecated import (
    _scale_invariant_signal_distortion_ratio as scale_invariant_signal_distortion_ratio,
)
from torchmetrics.functional.audio._deprecated import (
    _scale_invariant_signal_noise_ratio as scale_invariant_signal_noise_ratio,
)
from torchmetrics.functional.audio._deprecated import _signal_distortion_ratio as signal_distortion_ratio
from torchmetrics.functional.audio._deprecated import _signal_noise_ratio as signal_noise_ratio
from torchmetrics.functional.classification import (
    accuracy,
    auroc,
    average_precision,
    binary_precision_at_fixed_recall,
    calibration_error,
    cohen_kappa,
    confusion_matrix,
    dice,
    exact_match,
    f1_score,
    fbeta_score,
    hamming_distance,
    hinge_loss,
    jaccard_index,
    matthews_corrcoef,
    multiclass_precision_at_fixed_recall,
    multilabel_precision_at_fixed_recall,
    precision,
    precision_recall_curve,
    recall,
    roc,
    specificity,
    stat_scores,
)
from torchmetrics.functional.detection._deprecated import _modified_panoptic_quality as modified_panoptic_quality
from torchmetrics.functional.detection._deprecated import _panoptic_quality as panoptic_quality
from torchmetrics.functional.image._deprecated import (
    _error_relative_global_dimensionless_synthesis as error_relative_global_dimensionless_synthesis,
)
from torchmetrics.functional.image._deprecated import _image_gradients as image_gradients
from torchmetrics.functional.image._deprecated import (
    _multiscale_structural_similarity_index_measure as multiscale_structural_similarity_index_measure,
)
from torchmetrics.functional.image._deprecated import _peak_signal_noise_ratio as peak_signal_noise_ratio
from torchmetrics.functional.image._deprecated import (
    _relative_average_spectral_error as relative_average_spectral_error,
)
from torchmetrics.functional.image._deprecated import (
    _root_mean_squared_error_using_sliding_window as root_mean_squared_error_using_sliding_window,
)
from torchmetrics.functional.image._deprecated import _spectral_angle_mapper as spectral_angle_mapper
from torchmetrics.functional.image._deprecated import _spectral_distortion_index as spectral_distortion_index
from torchmetrics.functional.image._deprecated import (
    _structural_similarity_index_measure as structural_similarity_index_measure,
)
from torchmetrics.functional.image._deprecated import _total_variation as total_variation
from torchmetrics.functional.image._deprecated import _universal_image_quality_index as universal_image_quality_index
from torchmetrics.functional.nominal import (
    cramers_v,
    cramers_v_matrix,
    fleiss_kappa,
    pearsons_contingency_coefficient,
    pearsons_contingency_coefficient_matrix,
    theils_u,
    theils_u_matrix,
    tschuprows_t,
    tschuprows_t_matrix,
)
from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_linear_similarity,
    pairwise_manhattan_distance,
    pairwise_minkowski_distance,
)
from torchmetrics.functional.regression import (
    concordance_corrcoef,
    cosine_similarity,
    explained_variance,
    kendall_rank_corrcoef,
    kl_divergence,
    log_cosh_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    minkowski_distance,
    pearson_corrcoef,
    r2_score,
    relative_squared_error,
    spearman_corrcoef,
    symmetric_mean_absolute_percentage_error,
    tweedie_deviance_score,
    weighted_mean_absolute_percentage_error,
)
from torchmetrics.functional.retrieval._deprecated import _retrieval_average_precision as retrieval_average_precision
from torchmetrics.functional.retrieval._deprecated import _retrieval_fall_out as retrieval_fall_out
from torchmetrics.functional.retrieval._deprecated import _retrieval_hit_rate as retrieval_hit_rate
from torchmetrics.functional.retrieval._deprecated import _retrieval_normalized_dcg as retrieval_normalized_dcg
from torchmetrics.functional.retrieval._deprecated import _retrieval_precision as retrieval_precision
from torchmetrics.functional.retrieval._deprecated import (
    _retrieval_precision_recall_curve as retrieval_precision_recall_curve,
)
from torchmetrics.functional.retrieval._deprecated import _retrieval_r_precision as retrieval_r_precision
from torchmetrics.functional.retrieval._deprecated import _retrieval_recall as retrieval_recall
from torchmetrics.functional.retrieval._deprecated import _retrieval_reciprocal_rank as retrieval_reciprocal_rank
from torchmetrics.functional.text._deprecated import _bleu_score as bleu_score
from torchmetrics.functional.text._deprecated import _char_error_rate as char_error_rate
from torchmetrics.functional.text._deprecated import _chrf_score as chrf_score
from torchmetrics.functional.text._deprecated import _extended_edit_distance as extended_edit_distance
from torchmetrics.functional.text._deprecated import _match_error_rate as match_error_rate
from torchmetrics.functional.text._deprecated import _perplexity as perplexity
from torchmetrics.functional.text._deprecated import _rouge_score as rouge_score
from torchmetrics.functional.text._deprecated import _sacre_bleu_score as sacre_bleu_score
from torchmetrics.functional.text._deprecated import _squad as squad
from torchmetrics.functional.text._deprecated import _translation_edit_rate as translation_edit_rate
from torchmetrics.functional.text._deprecated import _word_error_rate as word_error_rate
from torchmetrics.functional.text._deprecated import _word_information_lost as word_information_lost
from torchmetrics.functional.text._deprecated import _word_information_preserved as word_information_preserved
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from torchmetrics.functional.text._deprecated import _bert_score as bert_score
    from torchmetrics.functional.text._deprecated import _infolm as infolm

__all__ = [
    "accuracy",
    "auroc",
    "average_precision",
    "bleu_score",
    "calibration_error",
    "char_error_rate",
    "chrf_score",
    "concordance_corrcoef",
    "cohen_kappa",
    "confusion_matrix",
    "cosine_similarity",
    "cramers_v",
    "cramers_v_matrix",
    "tweedie_deviance_score",
    "dice",
    "error_relative_global_dimensionless_synthesis",
    "exact_match",
    "explained_variance",
    "extended_edit_distance",
    "f1_score",
    "fbeta_score",
    "fleiss_kappa",
    "hamming_distance",
    "hinge_loss",
    "image_gradients",
    "jaccard_index",
    "kendall_rank_corrcoef",
    "kl_divergence",
    "log_cosh_error",
    "match_error_rate",
    "matthews_corrcoef",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "minkowski_distance",
    "multiscale_structural_similarity_index_measure",
    "pairwise_cosine_similarity",
    "pairwise_euclidean_distance",
    "pairwise_linear_similarity",
    "pairwise_manhattan_distance",
    "pairwise_minkowski_distance",
    "panoptic_quality",
    "pearson_corrcoef",
    "pearsons_contingency_coefficient",
    "pearsons_contingency_coefficient_matrix",
    "permutation_invariant_training",
    "perplexity",
    "pit_permutate",
    "precision",
    "precision_recall_curve",
    "peak_signal_noise_ratio",
    "r2_score",
    "recall",
    "relative_average_spectral_error",
    "relative_squared_error",
    "retrieval_average_precision",
    "retrieval_fall_out",
    "retrieval_hit_rate",
    "retrieval_normalized_dcg",
    "retrieval_precision",
    "retrieval_r_precision",
    "retrieval_recall",
    "retrieval_reciprocal_rank",
    "retrieval_precision_recall_curve",
    "roc",
    "root_mean_squared_error_using_sliding_window",
    "rouge_score",
    "sacre_bleu_score",
    "signal_distortion_ratio",
    "scale_invariant_signal_distortion_ratio",
    "scale_invariant_signal_noise_ratio",
    "signal_noise_ratio",
    "spearman_corrcoef",
    "specificity",
    "spectral_distortion_index",
    "squad",
    "structural_similarity_index_measure",
    "stat_scores",
    "symmetric_mean_absolute_percentage_error",
    "theils_u",
    "theils_u_matrix",
    "total_variation",
    "translation_edit_rate",
    "tschuprows_t",
    "tschuprows_t_matrix",
    "universal_image_quality_index",
    "spectral_angle_mapper",
    "weighted_mean_absolute_percentage_error",
    "word_error_rate",
    "word_information_lost",
    "word_information_preserved",
    "binary_precision_at_fixed_recall",
    "multilabel_precision_at_fixed_recall",
    "multiclass_precision_at_fixed_recall",
]
