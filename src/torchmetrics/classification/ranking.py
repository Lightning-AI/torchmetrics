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
from typing import Any, Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification.ranking import (
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_coverage_error_update,
    _multilabel_ranking_average_precision_update,
    _multilabel_ranking_loss_update,
    _multilabel_ranking_tensor_validation,
    _ranking_reduce,
)
from torchmetrics.metric import Metric


class MultilabelCoverageError(Metric):
    """Computes `Multilabel coverage error`_. The score measure how far we need to go through the ranked scores to
    cover all true labels. The best value is equal to the average number of labels in the target tensor per sample.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor
      containing ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlce`` (:class:`~torch.Tensor`): A tensor containing the multilabel coverage error.

    Args:
        num_labels: Integer specifing the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.classification import MultilabelCoverageError
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> mlce = MultilabelCoverageError(num_labels=5)
        >>> mlce(preds, target)
        tensor(3.9000)
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(num_labels, threshold=0.0, ignore_index=ignore_index)
        self.validate_args = validate_args
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.add_state("measure", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multilabel_ranking_tensor_validation(preds, target, self.num_labels, self.ignore_index)
        preds, target = _multilabel_confusion_matrix_format(
            preds, target, self.num_labels, threshold=0.0, ignore_index=self.ignore_index, should_threshold=False
        )
        measure, n_elements = _multilabel_coverage_error_update(preds, target)
        self.measure += measure
        self.total += n_elements

    def compute(self) -> Tensor:
        return _ranking_reduce(self.measure, self.total)


class MultilabelRankingAveragePrecision(Metric):
    """Computes label ranking average precision score for multilabel data [1]. The score is the average over each
    ground truth label assigned to each sample of the ratio of true vs. total labels with lower score. Best score
    is 1.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor
      containing ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlrap`` (:class:`~torch.Tensor`): A tensor containing the multilabel ranking average precision.

    Args:
        num_labels: Integer specifing the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.classification import MultilabelRankingAveragePrecision
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> mlrap = MultilabelRankingAveragePrecision(num_labels=5)
        >>> mlrap(preds, target)
        tensor(0.7744)
    """

    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(num_labels, threshold=0.0, ignore_index=ignore_index)
        self.validate_args = validate_args
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.add_state("measure", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multilabel_ranking_tensor_validation(preds, target, self.num_labels, self.ignore_index)
        preds, target = _multilabel_confusion_matrix_format(
            preds, target, self.num_labels, threshold=0.0, ignore_index=self.ignore_index, should_threshold=False
        )
        measure, n_elements = _multilabel_ranking_average_precision_update(preds, target)
        self.measure += measure
        self.total += n_elements

    def compute(self) -> Tensor:
        return _ranking_reduce(self.measure, self.total)


class MultilabelRankingLoss(Metric):
    """Computes the label ranking loss for multilabel data [1]. The score is corresponds to the average number of
    label pairs that are incorrectly ordered given some predictions weighted by the size of the label set and the
    number of labels not in the label set. The best score is 0.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor
      containing ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlrl`` (:class:`~torch.Tensor`): A tensor containing the multilabel ranking loss.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.classification import MultilabelRankingLoss
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> mlrl = MultilabelRankingLoss(num_labels=5)
        >>> mlrl(preds, target)
        tensor(0.4167)
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(num_labels, threshold=0.0, ignore_index=ignore_index)
        self.validate_args = validate_args
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.add_state("measure", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        if self.validate_args:
            _multilabel_ranking_tensor_validation(preds, target, self.num_labels, self.ignore_index)
        preds, target = _multilabel_confusion_matrix_format(
            preds, target, self.num_labels, threshold=0.0, ignore_index=self.ignore_index, should_threshold=False
        )
        measure, n_elements = _multilabel_ranking_loss_update(preds, target)
        self.measure += measure
        self.total += n_elements

    def compute(self) -> Tensor:
        return _ranking_reduce(self.measure, self.total)
