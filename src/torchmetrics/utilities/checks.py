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
import multiprocessing
import os
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from time import perf_counter
from typing import Any, Callable, Optional, no_type_check
from unittest.mock import Mock

import torch
from torch import Tensor

from torchmetrics.metric import Metric

_DOCTEST_DOWNLOAD_TIMEOUT = int(os.environ.get("DOCTEST_DOWNLOAD_TIMEOUT", 120))
_SKIP_SLOW_DOCTEST = bool(os.environ.get("SKIP_SLOW_DOCTEST", 0))


def _check_for_empty_tensors(preds: Tensor, target: Tensor) -> bool:
    return preds.numel() == target.numel() == 0


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )


def _check_retrieval_functional_inputs(
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
) -> tuple[Tensor, Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type.

    Args:
        preds: either tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values

    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty
            or not of the correct ``dtypes``.

    Returns:
        preds: as torch.float32
        target: as torch.long if not floating point else torch.float32

    """
    if preds.shape != target.shape:
        raise ValueError("`preds` and `target` must be of the same shape")

    if not preds.numel() or not preds.size():
        raise ValueError("`preds` and `target` must be non-empty and non-scalar tensors")

    return _check_retrieval_target_and_prediction_types(preds, target, allow_non_binary_target=allow_non_binary_target)


def _check_retrieval_inputs(
    indexes: Tensor,
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Check ``indexes``, ``preds`` and ``target`` tensors are of the same shape and of the correct data type.

    Args:
        indexes: tensor with queries indexes
        preds: tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values
        ignore_index: ignore predictions where targets are equal to this number

    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty or not of the correct ``dtypes``.

    Returns:
        indexes: as ``torch.long``
        preds: as ``torch.float32``
        target: as ``torch.long``

    """
    if indexes.shape != preds.shape or preds.shape != target.shape:
        raise ValueError("`indexes`, `preds` and `target` must be of the same shape")

    if indexes.dtype is not torch.long:
        raise ValueError("`indexes` must be a tensor of long integers")

    # remove predictions where target is equal to `ignore_index`
    if ignore_index is not None:
        valid_positions = target != ignore_index
        indexes, preds, target = indexes[valid_positions], preds[valid_positions], target[valid_positions]

    if not indexes.numel() or not indexes.size():
        raise ValueError(
            "`indexes`, `preds` and `target` must be non-empty and non-scalar tensors",
        )

    preds, target = _check_retrieval_target_and_prediction_types(
        preds, target, allow_non_binary_target=allow_non_binary_target
    )

    return indexes.long().flatten(), preds, target


def _check_retrieval_target_and_prediction_types(
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
) -> tuple[Tensor, Tensor]:
    """Check ``preds`` and ``target`` tensors are of the same shape and of the correct data type.

    Args:
        preds: either tensor with scores/logits
        target: tensor with ground true labels
        allow_non_binary_target: whether to allow target to contain non-binary values

    Raises:
        ValueError:
            If ``preds`` and ``target`` don't have the same shape, if they are empty or not of the correct ``dtypes``.

    """
    if target.dtype not in (torch.bool, torch.long, torch.int) and not torch.is_floating_point(target):
        raise ValueError("`target` must be a tensor of booleans, integers or floats")

    if not preds.is_floating_point():
        raise ValueError("`preds` must be a tensor of floats")

    if not allow_non_binary_target and (target.max() > 1 or target.min() < 0):
        raise ValueError("`target` must contain `binary` values")

    target = target.float() if target.is_floating_point() else target.long()
    preds = preds.float()

    return preds.flatten(), target.flatten()


def _allclose_recursive(res1: Any, res2: Any, atol: float = 1e-6) -> bool:
    """Recursively asserting that two results are within a certain tolerance."""
    # single output compare
    if isinstance(res1, Tensor):
        return torch.allclose(res1, res2, atol=atol)
    if isinstance(res1, str):
        return res1 == res2
    if isinstance(res1, Sequence):
        return all(_allclose_recursive(r1, r2) for r1, r2 in zip(res1, res2))
    if isinstance(res1, Mapping):
        return all(_allclose_recursive(res1[k], res2[k]) for k in res1)
    return res1 == res2


@no_type_check
def check_forward_full_state_property(
    metric_class: Metric,
    init_args: Optional[dict[str, Any]] = None,
    input_args: Optional[dict[str, Any]] = None,
    num_update_to_compare: Sequence[int] = [10, 100, 1000],
    reps: int = 5,
) -> None:
    """Check if the new ``full_state_update`` property works as intended.

    This function checks if the property can safely be set to ``False`` which will for most metrics results in a
    speedup when using ``forward``.

    Args:
        metric_class: metric class object that should be checked
        init_args: dict containing arguments for initializing the metric class
        input_args: dict containing arguments to pass to ``forward``
        num_update_to_compare: if we successfully detect that the flag is safe to set to ``False``
            we will run some speedup test. This arg should be a list of integers for how many
            steps to compare over.
        reps: number of repetitions of speedup test

    Example (states in ``update`` are independent, save to set ``full_state_update=False``)
        >>> from torchmetrics.classification import MulticlassConfusionMatrix
        >>> check_forward_full_state_property(  # doctest: +SKIP
        ...     MulticlassConfusionMatrix,
        ...     init_args = {'num_classes': 3},
        ...     input_args = {'preds': torch.randint(3, (100,)), 'target': torch.randint(3, (100,))},
        ... )
        Full state for 10 steps took: ...
        Partial state for 10 steps took: ...
        Full state for 100 steps took: ...
        Partial state for 100 steps took: ...
        Full state for 1000 steps took: ...
        Partial state for 1000 steps took: ...
        Recommended setting `full_state_update=False`

    Example (states in ``update`` are dependent meaning that ``full_state_update=True``):
        >>> from torchmetrics.classification import MulticlassConfusionMatrix
        >>> class MyMetric(MulticlassConfusionMatrix):
        ...     def update(self, preds, target):
        ...         super().update(preds, target)
        ...         # by construction make future states dependent on prior states
        ...         if self.confmat.sum() > 20:
        ...             self.reset()
        >>> check_forward_full_state_property(
        ...     MyMetric,
        ...     init_args = {'num_classes': 3},
        ...     input_args = {'preds': torch.randint(3, (10,)), 'target': torch.randint(3, (10,))},
        ... )
        Recommended setting `full_state_update=True`

    """
    init_args = init_args or {}
    input_args = input_args or {}

    class FullState(metric_class):
        full_state_update = True

    class PartState(metric_class):
        full_state_update = False

    fullstate = FullState(**init_args)
    partstate = PartState(**init_args)

    equal = True
    try:  # if it fails, the code most likely need access to the full state
        for _ in range(num_update_to_compare[0]):
            equal = equal & _allclose_recursive(fullstate(**input_args), partstate(**input_args))
    except RuntimeError:
        equal = False
    res1 = fullstate.compute()
    try:  # if it fails, the code most likely need access to the full state
        res2 = partstate.compute()
    except RuntimeError:
        equal = False
    equal = equal & _allclose_recursive(res1, res2)

    if not equal:  # we can stop early because the results did not match
        print("Recommended setting `full_state_update=True`")
        return

    # Do timings
    res = torch.zeros(2, len(num_update_to_compare), reps)
    for i, metric in enumerate([fullstate, partstate]):
        for j, t in enumerate(num_update_to_compare):
            for r in range(reps):
                start = perf_counter()
                for _ in range(t):
                    _ = metric(**input_args)
                end = perf_counter()
                res[i, j, r] = end - start
                metric.reset()

    mean = torch.mean(res, -1)
    std = torch.std(res, -1)

    for t in range(len(num_update_to_compare)):
        print(f"Full state for {num_update_to_compare[t]} steps took: {mean[0, t]}+-{std[0, t]:0.3f}")
        print(f"Partial state for {num_update_to_compare[t]} steps took: {mean[1, t]:0.3f}+-{std[1, t]:0.3f}")

    faster = (mean[1, -1] < mean[0, -1]).item()  # if faster on average, we recommend upgrading
    print(f"Recommended setting `full_state_update={not faster}`")
    return


def is_overridden(method_name: str, instance: object, parent: object) -> bool:
    """Check if a method has been overridden by an instance compared to its parent class."""
    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    # `functools.wraps()` support
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    # `Mock(wraps=...)` support
    if isinstance(instance_attr, Mock):
        # access the wrapped function
        instance_attr = instance_attr._mock_wraps
    # `partial` support
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")

    return instance_attr.__code__ != parent_attr.__code__


def _try_proceed_with_timeout(fn: Callable, timeout: int = _DOCTEST_DOWNLOAD_TIMEOUT) -> bool:
    """Check if a certain function is taking too long to execute.

    Function will only be executed if running inside a doctest context. Currently, does not support Windows.

    Args:
        fn: function to check
        timeout: timeout for function

    Returns:
        Bool indicating if the function finished within the specified timeout

    """
    # source: https://stackoverflow.com/a/14924210/4521646
    proc = multiprocessing.Process(target=fn)

    print(f"Trying to run `{fn.__name__}` for {timeout}s...", file=sys.stderr)
    proc.start()
    # Wait for N seconds or until process finishes
    proc.join(timeout)
    # If thread is still active
    if not proc.is_alive():
        return True

    print(f"`{fn.__name__}` did not complete with {timeout}, killing process and returning False", file=sys.stderr)
    # Terminate - may not work if process is stuck for good
    # proc.terminate()
    # proc.join()
    # OR Kill - will work for sure, no chance for process to finish nicely however
    proc.kill()
    return False
