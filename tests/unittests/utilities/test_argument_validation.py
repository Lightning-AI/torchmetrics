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
import pytest
import torch

from torchmetrics.classification import BinaryFairness
from torchmetrics.functional.classification.logauc import _validate_fpr_range
from torchmetrics.functional.classification.recall_fixed_precision import (
    _binary_recall_at_fixed_precision_arg_validation,
)
from torchmetrics.functional.classification.sensitivity_specificity import (
    _binary_sensitivity_at_specificity_arg_validation,
)
from torchmetrics.functional.classification.specificity_sensitivity import (
    _binary_specificity_at_sensitivity_arg_validation,
)
from torchmetrics.functional.classification.stat_scores import _multiclass_stat_scores_arg_validation
from torchmetrics.functional.retrieval import retrieval_average_precision, retrieval_reciprocal_rank
from torchmetrics.functional.segmentation.utils import table_contour_length, table_surface_area
from torchmetrics.image import PeakSignalNoiseRatioWithBlockedEffect
from torchmetrics.regression import LogCoshError, PearsonCorrCoef, SpearmanCorrCoef


@pytest.mark.parametrize("metric_class", [PearsonCorrCoef, SpearmanCorrCoef, LogCoshError])
@pytest.mark.parametrize("value", [0, -1, 1.5, "2", None])
def test_num_outputs_validation(metric_class, value):
    """Test that regression metrics reject a ``num_outputs`` that is not an int larger than 0."""
    with pytest.raises(ValueError, match="Expected argument `num_outputs` to be an int larger than 0"):
        metric_class(num_outputs=value)


def test_num_outputs_error_message_is_interpolated():
    """Test that the ``PearsonCorrCoef`` message interpolates the value instead of printing a literal brace."""
    with pytest.raises(ValueError, match="but got 0"):
        PearsonCorrCoef(num_outputs=0)


@pytest.mark.parametrize("value", [1, 0, -1, 2.5, "3", None])
def test_num_groups_validation(value):
    """Test that ``BinaryFairness`` rejects a ``num_groups`` that is not an int larger than 1."""
    with pytest.raises(ValueError, match="Expected argument `num_groups` to be an int larger than 1"):
        BinaryFairness(num_groups=value)


@pytest.mark.parametrize("value", [0, -1, 2.5, "8", None])
def test_psnrb_block_size_validation(value):
    """Test that ``PeakSignalNoiseRatioWithBlockedEffect`` rejects a non-positive-int ``block_size``."""
    with pytest.raises(ValueError, match="``block_size`` should be a positive integer"):
        PeakSignalNoiseRatioWithBlockedEffect(data_range=1.0, block_size=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", None])
def test_multiclass_top_k_validation(value):
    """Test that ``top_k`` is rejected unless it is an int larger than or equal to 1."""
    with pytest.raises(ValueError, match="Expected argument `top_k` to be an integer larger than or equal to 1"):
        _multiclass_stat_scores_arg_validation(num_classes=3, top_k=value)


@pytest.mark.parametrize(
    ("validation_fn", "argument"),
    [
        (_binary_recall_at_fixed_precision_arg_validation, "min_precision"),
        (_binary_sensitivity_at_specificity_arg_validation, "min_specificity"),
        (_binary_specificity_at_sensitivity_arg_validation, "min_sensitivity"),
    ],
)
@pytest.mark.parametrize("value", [-0.5, 1.5, "0.5", None])
def test_min_value_validation(validation_fn, argument, value):
    """Test that the ``min_*`` arguments are rejected unless they are floats inside the [0,1] range."""
    with pytest.raises(ValueError, match=rf"Expected argument `{argument}` to be an float in the \[0,1\] range"):
        validation_fn(**{argument: value})


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_min_value_validation_accepts_valid_floats(value):
    """Test that floats on and inside the [0,1] boundaries still pass."""
    _binary_recall_at_fixed_precision_arg_validation(min_precision=value)
    _binary_sensitivity_at_specificity_arg_validation(min_specificity=value)
    _binary_specificity_at_sensitivity_arg_validation(min_sensitivity=value)


@pytest.mark.parametrize("value", [(0.0,), (0.0, 0.1, 0.2), [0.0, 0.1], "01", 0.1, None])
def test_logauc_fpr_range_validation(value):
    """Test that ``fpr_range`` is rejected unless it is a tuple of exactly two floats."""
    with pytest.raises(ValueError, match="`fpr_range` should be a tuple of two floats"):
        _validate_fpr_range(value)


@pytest.mark.parametrize("spacing", [(1.0,), (1.0, 1.0, 1.0), [1.0, 1.0], 1.0, None])
def test_table_contour_length_spacing_validation(spacing):
    """Test that ``table_contour_length`` rejects a ``spacing`` that is not a tuple of length 2."""
    with pytest.raises(ValueError, match=r"The spacing must be a tuple of length 2\."):
        table_contour_length(spacing)


# a list is unhashable and trips the `lru_cache` on `table_surface_area` before validation runs
@pytest.mark.parametrize("spacing", [(1.0,), (1.0, 1.0), 1.0, None])
def test_table_surface_area_spacing_validation(spacing):
    """Test that ``table_surface_area`` rejects a ``spacing`` that is not a tuple of length 3."""
    with pytest.raises(ValueError, match=r"The spacing must be a tuple of length 3\."):
        table_surface_area(spacing)


@pytest.mark.parametrize("functional_metric", [retrieval_average_precision, retrieval_reciprocal_rank])
@pytest.mark.parametrize("value", [-1, 1.5, "2"])
def test_retrieval_top_k_validation(functional_metric, value):
    """Test that retrieval metrics reject a ``top_k`` that is not a positive int."""
    preds = torch.tensor([0.2, 0.3, 0.5])
    target = torch.tensor([False, False, True])
    with pytest.raises(ValueError, match="``top_k`` has to be a positive integer or None"):
        functional_metric(preds, target, top_k=value)
