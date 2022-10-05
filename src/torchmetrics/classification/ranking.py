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
    _coverage_error_compute,
    _coverage_error_update,
    _label_ranking_average_precision_compute,
    _label_ranking_average_precision_update,
    _label_ranking_loss_compute,
    _label_ranking_loss_update,
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
    """Computes multilabel coverage error [1]. The score measure how far we need to go through the ranked scores to
    cover all true labels. The best value is equal to the average number of labels in the target tensor per sample.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

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
        >>> metric = MultilabelCoverageError(num_labels=5)
        >>> metric(preds, target)
        tensor(3.9000)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
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

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

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
        >>> metric = MultilabelRankingAveragePrecision(num_labels=5)
        >>> metric(preds, target)
        tensor(0.7744)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
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

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

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
        >>> metric = MultilabelRankingLoss(num_labels=5)
        >>> metric(preds, target)
        tensor(0.4167)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
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


class CoverageError(Metric):
    """Computes multilabel coverage error [1]. The score measure how far we need to go through the ranked scores to
    cover all true labels. The best value is equal to the average number of labels in the target tensor per sample.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import CoverageError
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> metric = CoverageError()
        >>> metric(preds, target)
        tensor(3.9000)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    coverage: Tensor
    numel: Tensor
    weight: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("coverage", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> None:  # type: ignore
        """
        Args:
            preds: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
                of labels. Should either be probabilities of the positive class or corresponding logits
            target: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
                of labels. Should only contain binary labels.
            sample_weight: tensor of shape ``N`` where ``N`` is the number of samples. How much each sample
                should be weighted in the final score.
        """
        coverage, numel, sample_weight = _coverage_error_update(preds, target, sample_weight)
        self.coverage += coverage
        self.numel += numel
        if sample_weight is not None:
            self.weight += sample_weight

    def compute(self) -> Tensor:
        """Computes the multilabel coverage error."""
        return _coverage_error_compute(self.coverage, self.numel, self.weight)


class LabelRankingAveragePrecision(Metric):
    """Computes label ranking average precision score for multilabel data [1].

    The score is the average over each ground truth label assigned to each sample of the ratio of true vs.
    total labels with lower score. Best score is 1.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import LabelRankingAveragePrecision
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> metric = LabelRankingAveragePrecision()
        >>> metric(preds, target)
        tensor(0.7744)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
    """

    score: Tensor
    numel: Tensor
    sample_weight: Tensor
    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sample_weight", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> None:  # type: ignore
        """
        Args:
            preds: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
                of labels. Should either be probabilities of the positive class or corresponding logits
            target: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
                of labels. Should only contain binary labels.
            sample_weight: tensor of shape ``N`` where ``N`` is the number of samples. How much each sample
                should be weighted in the final score.
        """
        score, numel, sample_weight = _label_ranking_average_precision_update(preds, target, sample_weight)
        self.score += score
        self.numel += numel
        if sample_weight is not None:
            self.sample_weight += sample_weight

    def compute(self) -> Tensor:
        """Computes the label ranking average precision score."""
        return _label_ranking_average_precision_compute(self.score, self.numel, self.sample_weight)


class LabelRankingLoss(Metric):
    """Computes the label ranking loss for multilabel data [1]. The score is corresponds to the average number of
    label pairs that are incorrectly ordered given some predictions weighted by the size of the label set and the
    number of labels not in the label set. The best score is 0.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import LabelRankingLoss
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand(10, 5)
        >>> target = torch.randint(2, (10, 5))
        >>> metric = LabelRankingLoss()
        >>> metric(preds, target)
        tensor(0.4167)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.
    """

    loss: Tensor
    numel: Tensor
    sample_weight: Tensor
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sample_weight", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, sample_weight: Optional[Tensor] = None) -> None:  # type: ignore
        """
        Args:
            preds: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
                of labels. Should either be probabilities of the positive class or corresponding logits
            target: tensor of shape ``[N,L]`` where ``N`` is the number of samples and ``L`` is the number
                of labels. Should only contain binary labels.
            sample_weight: tensor of shape ``N`` where ``N`` is the number of samples. How much each sample
                should be weighted in the final score.
        """
        loss, numel, sample_weight = _label_ranking_loss_update(preds, target, sample_weight)
        self.loss += loss
        self.numel += numel
        if sample_weight is not None:
            self.sample_weight += sample_weight

    def compute(self) -> Tensor:
        """Computes the label ranking loss."""
        return _label_ranking_loss_compute(self.loss, self.numel, self.sample_weight)
