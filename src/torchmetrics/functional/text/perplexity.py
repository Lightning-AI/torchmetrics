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

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

_TORCH_FLOAT_OR_DOUBLE = (torch.float32, torch.float64)


def _check_shape_and_type_consistency(preds: Tensor, target: Tensor) -> None:
    """Check shape and type consistency of input vectors.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size]. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].

    Raises:
        ValueError:
            If ``preds`` tensor has no 3 dimensions.
        ValueError:
            If ``target`` tensor has no 2 dimensions.
        ValueError:
            If the first two dimensions of ``preds`` and ``target`` do not equal.
        TypeError:
            If ``preds`` dtype is not one of ``(torch.float16, torch.float32, torch.float64)``
        TypeError:
            If ``target`` is not of a type LongTensor (torch.int64)
    """
    if len(preds.shape) != 3:
        raise ValueError(
            "Input tensor `preds` is expected to have 3 dimensions, [batch_size, seq_len, vocab_size],"
            f" but got {len(preds.shape)}."
        )
    if len(target.shape) != 2:
        raise ValueError(
            "Input tensor `target` is expected to have 2 dimensions, [batch_size, seq_len],"
            f" but got {len(target.shape)}."
        )
    if preds.shape[:2] != target.shape:
        raise ValueError(
            "Input tensors `preds` and `target` are expected to have equaling first two dimensions,"
            f" [batch_size, seq_len], but got {preds.shape[:2]} and {target.shape}."
        )
    if preds.dtype not in _TORCH_FLOAT_OR_DOUBLE:
        raise TypeError(
            f"Input tensor `preds` is expected to be of a type one of {_TORCH_FLOAT_OR_DOUBLE} but got {preds.dtype}."
        )
    if target.dtype != torch.int64:
        raise TypeError(f"Input tensor `target` is expected to be of a type {torch.int64} but got {target.dtype}.")


def _perplexity_update(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Compute intermediate statistics for Perplexity.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size]. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.

    Returns:
        Log probabilities, summed over all samples
        Number of samples
    """
    _check_shape_and_type_consistency(preds, target)

    probs = F.softmax(preds.reshape(-1, preds.shape[-1]), dim=1)
    target = target.reshape(-1)

    if ignore_index is not None:
        mask = target.ne(ignore_index)
        target = target.where(target != ignore_index, torch.tensor(0, device=target.device))
    else:
        mask = torch.ones_like(target, dtype=torch.bool)

    probs = probs[:, target].diagonal()[mask]
    total_log_probs = -probs.log().sum()
    count = mask.sum()

    return total_log_probs, count


def _perplexity_compute(total: Tensor, count: Tensor) -> Tensor:
    """Compute the Perplexity.

    Args:
        total: Log probabilities, summed over all samples
        count: Number of samples
    Returns:
        Perplexity
    """
    return torch.exp(total / count)


def perplexity(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> Tensor:
    """Perplexity measures how well a language model predicts a text sample.

    This metric is calculated as the average number of bits per word a model needs to represent the sample.

    Args:
        preds:
            Logits or a unnormalized score assigned to each token in a sequence with shape [batch_size, seq_len,
            vocab_size], which is the output of a language model. Scores will be normalized internally using softmax.
        target:
            Ground truth values with a shape [batch_size, seq_len].
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score.

    Returns:
        Perplexity value

    Examples:
        >>> import torch
        >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
        >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
        >>> target[0, 6:] = -100
        >>> perplexity(preds, target, ignore_index=-100)
        tensor(5.2545)
    """
    total, count = _perplexity_update(preds, target, ignore_index)
    return _perplexity_compute(total, count)
