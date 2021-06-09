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
import functools
import inspect
import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, List, Optional, Union

import torch
from torch import Tensor, nn

from torchmetrics.utilities import apply_to_collection, rank_zero_warn
from torchmetrics.utilities.data import _flatten, dim_zero_cat, dim_zero_mean, dim_zero_sum
from torchmetrics.utilities.distributed import gather_all_tensors
from torchmetrics.utilities.imports import _LIGHTNING_AVAILABLE, _compare_version


class Metric(nn.Module, ABC):
    """
    Base class for all metrics present in the Metrics API.

    Implements ``add_state()``, ``forward()``, ``reset()`` and a few other things to
    handle distributed synchronization and per-step metric computation.

    Override ``update()`` and ``compute()`` functions to implement your own metric. Use
    ``add_state()`` to register metric state variables which keep track of state on each
    call of ``update()`` and are synchronized across processes when ``compute()`` is called.

    Note:
        Metric state variables can either be ``torch.Tensors`` or an empty list which can we used
        to store `torch.Tensors``.

    Note:
        Different metrics only override ``update()`` and not ``forward()``. A call to ``update()``
        is valid, but it won't return the metric value at the current step. A call to ``forward()``
        automatically calls ``update()`` and also returns the metric value at the current step.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.
    """

    __jit_ignored_attributes__ = ["is_differentiable"]

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__()

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"torchmetrics.metric.{self.__class__.__name__}")

        self._LIGHTNING_GREATER_EQUAL_1_3 = _compare_version("pytorch_lightning", operator.ge, "1.3.0")

        self.dist_sync_on_step = dist_sync_on_step
        self.compute_on_step = compute_on_step
        self.process_group = process_group
        self.dist_sync_fn = dist_sync_fn
        self._to_sync = True

        self._update_signature = inspect.signature(self.update)
        self.update = self._wrap_update(self.update)
        self.compute = self._wrap_compute(self.compute)
        self._computed = None
        self._forward_cache = None
        self._update_called = False

        # initialize state
        self._defaults = {}
        self._persistent = {}
        self._reductions = {}

    def add_state(
        self, name: str, default, dist_reduce_fx: Optional[Union[str, Callable]] = None, persistent: bool = False
    ):
        """
        Adds metric state variable. Only used by subclasses.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a ``torch.Tensor`` or an empty list. The state will be
                reset to this value when ``self.reset()`` is called.
            dist_reduce_fx (Optional): Function to reduce state across multiple processes in distributed mode.
                If value is ``"sum"``, ``"mean"``, or ``"cat"``, we will use ``torch.sum``, ``torch.mean``,
                and ``torch.cat`` respectively, each with argument ``dim=0``. Note that the ``"cat"`` reduction
                only makes sense if the state is a list, and not a tensor. The user can also pass a custom
                function in this parameter.
            persistent (Optional): whether the state will be saved as part of the modules ``state_dict``.
                Default is ``False``.

        Note:
            Setting ``dist_reduce_fx`` to None will return the metric state synchronized across different processes.
            However, there won't be any reduction function applied to the synchronized metric state.

            The metric states would be synced as follows

            - If the metric state is ``torch.Tensor``, the synced value will be a stacked ``torch.Tensor`` across
              the process dimension if the metric state was a ``torch.Tensor``. The original ``torch.Tensor`` metric
              state retains dimension and hence the synchronized output will be of shape ``(num_process, ...)``.

            - If the metric state is a ``list``, the synced value will be a ``list`` containing the
              combined elements from all processes.

        Note:
            When passing a custom function to ``dist_reduce_fx``, expect the synchronized metric state to follow
            the format discussed in the above note.

        Raises:
            ValueError:
                If ``default`` is not a ``tensor`` or an ``empty list``.
            ValueError:
                If ``dist_reduce_fx`` is not callable or one of ``"mean"``, ``"sum"``, ``"cat"``, ``None``.
        """
        if not isinstance(default, (Tensor, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not isinstance(dist_reduce_fx, Callable):
            raise ValueError("`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', None]")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fx

    @torch.jit.unused
    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        with torch.no_grad():
            self.update(*args, **kwargs)

        if self.compute_on_step:
            self._to_sync = self.dist_sync_on_step

            # save context before switch
            cache = {attr: getattr(self, attr) for attr in self._defaults}

            # call reset, update, compute, on single batch
            self.reset()
            self.update(*args, **kwargs)
            self._forward_cache = self.compute()

            # restore context
            for attr, val in cache.items():
                setattr(self, attr, val)
            self._to_sync = True
            self._computed = None

            return self._forward_cache

    def _sync_dist(self, dist_sync_fn=gather_all_tensors):
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}
        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]
        output_dict = apply_to_collection(
            input_dict,
            Tensor,
            dist_sync_fn,
            group=self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)
            if isinstance(output_dict[attr][0], Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError('reduction_fn must be callable or None')
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update):

        @functools.wraps(update)
        def wrapped_func(*args, **kwargs):
            self._computed = None
            self._update_called = True
            return update(*args, **kwargs)

        return wrapped_func

    def _wrap_compute(self, compute):

        @functools.wraps(compute)
        def wrapped_func(*args, **kwargs):
            if not self._update_called:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.", UserWarning
                )

            # return cached value
            if self._computed is not None:
                return self._computed

            dist_sync_fn = self.dist_sync_fn
            if dist_sync_fn is None and torch.distributed.is_available() and torch.distributed.is_initialized():
                # User provided a bool, so we assume DDP if available
                dist_sync_fn = gather_all_tensors

            synced = False
            cache = []
            if self._to_sync and dist_sync_fn is not None:
                # cache prior to syncing
                cache = {attr: getattr(self, attr) for attr in self._defaults}

                # sync
                self._sync_dist(dist_sync_fn)
                synced = True

            self._computed = compute(*args, **kwargs)
            if synced:
                # if we synced, restore to cache so that we can continue to accumulate un-synced state
                for attr, val in cache.items():
                    setattr(self, attr, val)

            return self._computed

        return wrapped_func

    @abstractmethod
    def update(self) -> None:  # pylint: disable=E0202
        """
        Override this method to update the state variables of your metric class.
        """
        pass

    @abstractmethod
    def compute(self):  # pylint: disable=E0202
        """
        Override this method to compute the final metric value from state variables
        synchronized across the distributed backend.
        """
        pass

    def reset(self):
        """
        This method automatically resets the metric state variables to their default value.
        """
        self._update_called = False
        self._forward_cache = None
        # lower lightning versions requires this implicitly to log metric objects correctly in self.log
        if not _LIGHTNING_AVAILABLE or self._LIGHTNING_GREATER_EQUAL_1_3:
            self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

    def clone(self):
        """ Make a copy of the metric """
        return deepcopy(self)

    def __getstate__(self):
        # ignore update and compute functions for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ["update", "compute"]}

    def __setstate__(self, state):
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self.update = self._wrap_update(self.update)
        self.compute = self._wrap_compute(self.compute)

    def _apply(self, fn):
        """Overwrite _apply function such that we can also move metric states
        to the correct device when `.to`, `.cuda`, etc methods are called
        """
        this = super()._apply(fn)
        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if isinstance(value, Tensor):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]

            current_val = getattr(this, key)
            if isinstance(current_val, Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    "Expected metric state to be either a Tensor"
                    f"or a list of Tensor, but encountered {current_val}"
                )
        return this

    def persistent(self, mode: bool = False):
        """Method for post-init to change if metric states should be saved to
        its state_dict
        """
        for key in self._persistent:
            self._persistent[key] = mode

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Register metric states to be part of the state_dict
        for key in self._defaults:
            if self._persistent[key]:
                current_val = getattr(self, key)
                if not keep_vars:
                    if torch.is_tensor(current_val):
                        current_val = current_val.detach()
                    elif isinstance(current_val, list):
                        current_val = [cur_v.detach() if torch.is_tensor(cur_v) else cur_v for cur_v in current_val]
                destination[prefix + key] = current_val
        return destination

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """ Loads metric states from state_dict """
        for key in self._defaults:
            name = prefix + key
            if name in state_dict:
                setattr(self, key, state_dict.pop(name))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )

    def _filter_kwargs(self, **kwargs):
        """ filter kwargs such that they match the update signature of the metric """

        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = self._update_signature.parameters
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items() if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
        }

        # if no kwargs filtered, return al kwargs as default
        if not filtered_kwargs:
            filtered_kwargs = kwargs
        return filtered_kwargs

    def __hash__(self):
        hash_vals = [self.__class__.__name__]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))

    def __add__(self, other: Any):
        return CompositionalMetric(torch.add, self, other)

    def __and__(self, other: Any):
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __eq__(self, other: Any):
        return CompositionalMetric(torch.eq, self, other)

    def __floordiv__(self, other: Any):
        return CompositionalMetric(torch.floor_divide, self, other)

    def __ge__(self, other: Any):
        return CompositionalMetric(torch.ge, self, other)

    def __gt__(self, other: Any):
        return CompositionalMetric(torch.gt, self, other)

    def __le__(self, other: Any):
        return CompositionalMetric(torch.le, self, other)

    def __lt__(self, other: Any):
        return CompositionalMetric(torch.lt, self, other)

    def __matmul__(self, other: Any):
        return CompositionalMetric(torch.matmul, self, other)

    def __mod__(self, other: Any):
        return CompositionalMetric(torch.fmod, self, other)

    def __mul__(self, other: Any):
        return CompositionalMetric(torch.mul, self, other)

    def __ne__(self, other: Any):
        return CompositionalMetric(torch.ne, self, other)

    def __or__(self, other: Any):
        return CompositionalMetric(torch.bitwise_or, self, other)

    def __pow__(self, other: Any):
        return CompositionalMetric(torch.pow, self, other)

    def __radd__(self, other: Any):
        return CompositionalMetric(torch.add, other, self)

    def __rand__(self, other: Any):
        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __rfloordiv__(self, other: Any):
        return CompositionalMetric(torch.floor_divide, other, self)

    def __rmatmul__(self, other: Any):
        return CompositionalMetric(torch.matmul, other, self)

    def __rmod__(self, other: Any):
        return CompositionalMetric(torch.fmod, other, self)

    def __rmul__(self, other: Any):
        return CompositionalMetric(torch.mul, other, self)

    def __ror__(self, other: Any):
        return CompositionalMetric(torch.bitwise_or, other, self)

    def __rpow__(self, other: Any):
        return CompositionalMetric(torch.pow, other, self)

    def __rsub__(self, other: Any):
        return CompositionalMetric(torch.sub, other, self)

    def __rtruediv__(self, other: Any):
        return CompositionalMetric(torch.true_divide, other, self)

    def __rxor__(self, other: Any):
        return CompositionalMetric(torch.bitwise_xor, other, self)

    def __sub__(self, other: Any):
        return CompositionalMetric(torch.sub, self, other)

    def __truediv__(self, other: Any):
        return CompositionalMetric(torch.true_divide, self, other)

    def __xor__(self, other: Any):
        return CompositionalMetric(torch.bitwise_xor, self, other)

    def __abs__(self):
        return CompositionalMetric(torch.abs, self, None)

    def __inv__(self):
        return CompositionalMetric(torch.bitwise_not, self, None)

    def __invert__(self):
        return self.__inv__()

    def __neg__(self):
        return CompositionalMetric(_neg, self, None)

    def __pos__(self):
        return CompositionalMetric(torch.abs, self, None)

    def __getitem__(self, idx):
        return CompositionalMetric(lambda x: x[idx], self, None)

    @property
    def is_differentiable(self):
        # There is a bug in PyTorch that leads to properties being executed during scripting
        # To make the metric scriptable, we add property to ignore list and switch to return None here
        return None


def _neg(tensor: Tensor):
    return -torch.abs(tensor)


class CompositionalMetric(Metric):
    """Composition of two metrics with a specific operator which will be executed upon metrics compute """

    def __init__(
        self,
        operator: Callable,
        metric_a: Union[Metric, int, float, Tensor],
        metric_b: Union[Metric, int, float, Tensor, None],
    ):
        """
        Args:
            operator: the operator taking in one (if metric_b is None)
                or two arguments. Will be applied to outputs of metric_a.compute()
                and (optionally if metric_b is not None) metric_b.compute()
            metric_a: first metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None
        """
        super().__init__()

        self.op = operator

        if isinstance(metric_a, Tensor):
            self.register_buffer("metric_a", metric_a)
        else:
            self.metric_a = metric_a

        if isinstance(metric_b, Tensor):
            self.register_buffer("metric_b", metric_b)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn: Callable = None) -> None:
        # No syncing required here. syncing will be done in metric_a and metric_b
        pass

    def update(self, *args, **kwargs) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **self.metric_a._filter_kwargs(**kwargs))

        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **self.metric_b._filter_kwargs(**kwargs))

    def compute(self) -> Any:

        # also some parsing for kwargs?
        if isinstance(self.metric_a, Metric):
            val_a = self.metric_a.compute()
        else:
            val_a = self.metric_a

        if isinstance(self.metric_b, Metric):
            val_b = self.metric_b.compute()
        else:
            val_b = self.metric_b

        if val_b is None:
            return self.op(val_a)

        return self.op(val_a, val_b)

    def reset(self) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        _op_metrics = f"(\n  {self.op.__name__}(\n    {repr(self.metric_a)},\n    {repr(self.metric_b)}\n  )\n)"
        repr_str = self.__class__.__name__ + _op_metrics

        return repr_str
