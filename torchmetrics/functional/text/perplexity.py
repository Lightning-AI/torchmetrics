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

from typing import Optional, Tuple

from torch import Tensor, isnan


# From https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
# From PyTorch v1.10, this is officially supported.
def nanmean(v: Tensor, *args, inplace: bool = False, **kwargs) -> Tensor:
    if not inplace:
        v = v.clone()
    is_nan = isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def _perplexity_update(probs: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Update the perplexity score with the current probabilities.

    Args:
        probs: Probabilities assigned to each token in a sequence with shape (batch_size, seq_len)
        mask: Mask for the sequence with dtype bool and shape (batch_size, seq_len)
    Returns:
        Perplexity, summed over all samples
        Number of samples
    """
    if mask is not None:
        probs = probs.clone()
        probs[~mask] = float("NaN")

    # It doesn't matter the log and exp base, as long as they are the same, because they cancel each other out.
    total = (-nanmean(probs.log(), dim=-1)).exp().sum()
    count = len(probs)
    return total, count


def _perplexity_compute(total: Tensor, count: Tensor) -> Tensor:
    """Compute the Perplexity.

    Args:
        total: Perplexity, summed over all samples
        count: Number of samples
    Returns:
        Perplexity
    """
    return total / count


def perplexity(self, probs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Perplexity measures how well a language model predicts a text sample. It's calculated as the average number
    of bits per word a model needs to represent the sample.

    Args:
        probs: Probabilities assigned to each token in a sequence with shape (batch_size, seq_len)
        mask: Mask for the sequence with dtype bool and shape (batch_size, seq_len)

    Returns:
        Perplexity

    Examples:
        >>> from torch import tensor
        >>> probs = tensor([[.2, .04, .8], [.34, .12, .56]])
        >>> mask = tensor([[True, True, False], [True, True, True]])
        >>> perplexity(probs, mask)
        tensor(7.3522)
    """
    total, count = _perplexity_update(probs, mask)
    return _perplexity_compute(total, count)
