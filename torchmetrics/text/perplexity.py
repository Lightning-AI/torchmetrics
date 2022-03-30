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

from typing import Any, Dict

from torch import Tensor, tensor

from torchmetrics.functional.text.perplexity import _perplexity_compute, _perplexity_update
from torchmetrics.metric import Metric


class Perplexity(Metric):
    r"""
    Perplexity measures how well a language model predicts a text sample. It's calculated as the average number of bits
    per word a model needs to represent the sample.

    Args:
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Perplexity

    Examples:
        >>> probs = tensor([[.2, .04, .8], [.34, .12, .56]])
        >>> mask = tensor([[True, True, False], [True, True, True]])
        >>> metric = Perplexity()
        >>> metric(probs, mask)
        tensor(4.58)
    """
    is_differentiable = False
    higher_is_better = False
    total: Tensor
    count: Tensor

    def __init__(
        self,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(**kwargs)
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, mask: Tensor) -> None:
        """Store references/predictions for computing the perplexity.

        Args:
            probs: Probabilities assigned to each token in a sequence with shape (batch_size, seq_len)
            mask: Mask for the sequence with dtype bool and shape (batch_size, seq_len)
        """
        total, count = _perplexity_update(probs, mask)
        self.total += total
        self.count += count

    def compute(self) -> Tensor:
        """Calculate the perplexity.

        Returns:
           Perplexity
        """
        return _perplexity_compute(self.total, self.count)
