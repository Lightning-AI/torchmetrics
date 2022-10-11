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

from typing import Any, Dict, Optional

from torch import Tensor, tensor

from torchmetrics.functional.text.perplexity import _perplexity_compute, _perplexity_update
from torchmetrics.metric import Metric


class Perplexity(Metric):
    r"""Perplexity measures how well a language model predicts a text sample. It's calculated as the average number
    of bits per word a model needs to represent the sample.

    Args:
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> metric = Perplexity(ignore_index=-100)
        >>> metric(preds, target)
        tensor(5.2545)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(**kwargs)
        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError(f"Argument `ignore_index` expected to either be `None` or an `int` but got {ignore_index}")
        self.ignore_index = ignore_index
        self.add_state("total_log_probs", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Compute and store intermediate statistics for Perplexity.

        Args:
            preds:
                Probabilities assigned to each token in a sequence with shape [batch_size, seq_len, vocab_size].
            target:
                Ground truth values with a shape [batch_size, seq_len].
        """
        total_log_probs, count = _perplexity_update(preds, target, self.ignore_index)
        self.total_log_probs += total_log_probs
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity.

        Returns:
           Perplexity
        """
        return _perplexity_compute(self.total_log_probs, self.count)
