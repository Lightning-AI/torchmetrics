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

# It is needed to distinguish between native float and Metric's' function called float.
# later, this function was used instead of the built-in float type...
import builtins
import functools
import inspect
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, ClassVar, Dict, Generator, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.utilities.data import (
    _flatten,
    _squeeze_if_scalar,
    apply_to_collection,
    dim_zero_cat,
    dim_zero_max,
    dim_zero_mean,
    dim_zero_min,
    dim_zero_sum,
)
from torchmetrics.utilities.distributed import gather_all_tensors
from torchmetrics.utilities.exceptions import TorchMetricsUserError
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val
from torchmetrics.utilities.prints import rank_zero_warn


def jit_distributed_available() -> bool:
    """Determine if distributed mode is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class Metric(Module, ABC):
    """Base class for all metrics present in the Metrics API.

    This class is inherited by all metrics and implements the following functionality:
    1. Handles the transfer of metric states to correct device
    2. Handles the synchronization of metric states across processes

    The three core methods of the base class are
    * ``add_state()``
    * ``forward()``
    * ``reset()``

    which should almost never be overwritten by child classes. Instead, the following methods should be overwritten
    * ``update()``
    * ``compute()``


    Args:
        kwargs: additional keyword arguments, see :ref:`Metric kwargs` for more info.

            - compute_on_cpu: If metric state should be stored on CPU during computations. Only works for list states.
            - dist_sync_on_step: If metric state should synchronize on ``forward()``. Default is ``False``
            - process_group: The process group on which the synchronization is called. Default is the world.
            - dist_sync_fn: Function that performs the allgather option on the metric state. Default is an custom
              implementation that calls ``torch.distributed.all_gather`` internally.
            - distributed_available_fn: Function that checks if the distributed backend is available. Defaults to a
              check of ``torch.distributed.is_available()`` and ``torch.distributed.is_initialized()``.
            - sync_on_compute: If metric state should synchronize when ``compute`` is called. Default is ``True``
            - compute_with_cache: If results from ``compute`` should be cached. Default is ``False``
    """

    __jit_ignored_attributes__: ClassVar[List[str]] = ["device"]
    __jit_unused_properties__: ClassVar[List[str]] = [
        "is_differentiable",
        "higher_is_better",
        "plot_lower_bound",
        "plot_upper_bound",
        "plot_legend_name",
    ]
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = None
    full_state_update: Optional[bool] = None

    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None
    plot_legend_name: Optional[str] = None

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"torchmetrics.metric.{self.__class__.__name__}")

        self._device = torch.device("cpu")

        self.compute_on_cpu = kwargs.pop("compute_on_cpu", False)
        if not isinstance(self.compute_on_cpu, bool):
            raise ValueError(
                f"Expected keyword argument `compute_on_cpu` to be an `bool` but got {self.compute_on_cpu}"
            )

        self.dist_sync_on_step = kwargs.pop("dist_sync_on_step", False)
        if not isinstance(self.dist_sync_on_step, bool):
            raise ValueError(
                f"Expected keyword argument `dist_sync_on_step` to be an `bool` but got {self.dist_sync_on_step}"
            )

        self.process_group = kwargs.pop("process_group", None)

        self.dist_sync_fn = kwargs.pop("dist_sync_fn", None)
        if self.dist_sync_fn is not None and not callable(self.dist_sync_fn):
            raise ValueError(
                f"Expected keyword argument `dist_sync_fn` to be an callable function but got {self.dist_sync_fn}"
            )

        self.distributed_available_fn = kwargs.pop("distributed_available_fn", None) or jit_distributed_available

        self.sync_on_compute = kwargs.pop("sync_on_compute", True)
        if not isinstance(self.sync_on_compute, bool):
            raise ValueError(
                f"Expected keyword argument `sync_on_compute` to be a `bool` but got {self.sync_on_compute}"
            )
        self.compute_with_cache = kwargs.pop("compute_with_cache", True)
        if not isinstance(self.compute_with_cache, bool):
            raise ValueError(
                f"Expected keyword argument `compute_with_cache` to be a `bool` but got {self.compute_with_cache}"
            )

        if kwargs:
            kwargs_ = [f"`{a}`" for a in sorted(kwargs)]
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs_)}")

        # initialize
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore[method-assign]
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore[method-assign]
        self._computed = None
        self._forward_cache = None
        self._update_count = 0
        self._to_sync = self.sync_on_compute
        self._should_unsync = True
        self._enable_grad = False
        self._dtype_convert = False

        # initialize state
        self._defaults: Dict[str, Union[List, Tensor]] = {}
        self._persistent: Dict[str, bool] = {}
        self._reductions: Dict[str, Union[str, Callable[..., Any], None]] = {}

        # state management
        self._is_synced = False
        self._cache: Optional[Dict[str, Union[List[Tensor], Tensor]]] = None

    @property
    def _update_called(self) -> bool:
        # TODO: this is needed for internal lightning, remove after v0.12 and update on lightning side
        return self._update_count > 0

    @property
    def update_called(self) -> bool:
        """Returns `True` if `update` or `forward` has been called initialization or last `reset`."""
        return self._update_count > 0

    @property
    def update_count(self) -> int:
        """Get the number of times `update` and/or `forward` has been called since initialization or last `reset`."""
        return self._update_count

    def add_state(
        self,
        name: str,
        default: Union[list, Tensor],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
    ) -> None:
        """Add metric state variable. Only used by subclasses.

        Metric state variables are either `:class:`~torch.Tensor` or an empty list, which can be appended to by the
        metric. Each state variable must have a unique name associated with it. State variables are accessible as
        attributes of the metric i.e, if ``name`` is ``"my_state"`` then its value can be accessed from an instance
        ``metric`` as ``metric.my_state``. Metric states behave like buffers and parameters of :class:`~torch.nn.Module`
        as they are also updated when ``.to()`` is called. Unlike parameters and buffers, metric states are not by
        default saved in the modules :attr:`~torch.nn.Module.state_dict`.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a :class:`~torch.Tensor` or an empty list.
                The state will be reset to this value when ``self.reset()`` is called.
            dist_reduce_fx (Optional): Function to reduce state across multiple processes in distributed mode.
                If value is ``"sum"``, ``"mean"``, ``"cat"``, ``"min"`` or ``"max"`` we will use ``torch.sum``,
                ``torch.mean``, ``torch.cat``, ``torch.min`` and ``torch.max``` respectively, each with argument
                ``dim=0``. Note that the ``"cat"`` reduction only makes sense if the state is a list, and not
                a tensor. The user can also pass a custom function in this parameter.
            persistent (Optional): whether the state will be saved as part of the modules ``state_dict``.
                Default is ``False``.

        Note:
            Setting ``dist_reduce_fx`` to None will return the metric state synchronized across different processes.
            However, there won't be any reduction function applied to the synchronized metric state.

            The metric states would be synced as follows

            - If the metric state is :class:`~torch.Tensor`, the synced value will be a stacked :class:`~torch.Tensor`
              across the process dimension if the metric state was a :class:`~torch.Tensor`. The original
              :class:`~torch.Tensor` metric state retains dimension and hence the synchronized output will be of shape
              ``(num_process, ...)``.

            - If the metric state is a ``list``, the synced value will be a ``list`` containing the
              combined elements from all processes.

        Note:
            When passing a custom function to ``dist_reduce_fx``, expect the synchronized metric state to follow
            the format discussed in the above note.

        Raises:
            ValueError:
                If ``default`` is not a ``tensor`` or an ``empty list``.
            ValueError:
                If ``dist_reduce_fx`` is not callable or one of ``"mean"``, ``"sum"``, ``"cat"``, ``"min"``,
                ``"max"`` or ``None``.
        """
        if not isinstance(default, (Tensor, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "max":
            dist_reduce_fx = dim_zero_max
        elif dist_reduce_fx == "min":
            dist_reduce_fx = dim_zero_min
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not callable(dist_reduce_fx):
            raise ValueError("`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', 'min', 'max', None]")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fx

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Aggregate and evaluate batch input directly.

        Serves the dual purpose of both computing the metric on the current batch of inputs but also add the batch
        statistics to the overall accumululating metric state. Input arguments are the exact same as corresponding
        ``update`` method. The returned output is the exact same as the output of ``compute``.

        Args:
            args: Any arguments as required by the metric ``update`` method.
            kwargs: Any keyword arguments as required by the metric ``update`` method.

        Returns:
            The output of the ``compute`` method evaluated on the current batch.

        Raises:
            TorchMetricsUserError:
                If the metric is already synced and ``forward`` is called again.
        """
        # check if states are already synced
        if self._is_synced:
            raise TorchMetricsUserError(
                "The Metric shouldn't be synced when performing ``forward``. "
                "HINT: Did you forget to call ``unsync`` ?."
            )

        if self.full_state_update or self.full_state_update is None or self.dist_sync_on_step:
            self._forward_cache = self._forward_full_state_update(*args, **kwargs)
        else:
            self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)

        return self._forward_cache

    def _forward_full_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """Forward computation using two calls to `update`.

        Doing this secures that metrics that need access to the full metric state during `update` works as expected.
        This is the most safe method to use for any metric but also the slower version of the two forward
        implementations.
        """
        # global accumulation
        self.update(*args, **kwargs)
        _update_count = self._update_count

        self._to_sync = self.dist_sync_on_step
        # skip restore cache operation from compute as cache is stored below.
        self._should_unsync = False
        # skip computing on cpu for the batch
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False

        # save context before switch
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # call reset, update, compute, on single batch
        self._enable_grad = True  # allow grads for batch computation
        self.reset()
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # restore context
        for attr, val in cache.items():
            setattr(self, attr, val)
        self._update_count = _update_count

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _forward_reduce_state_update(self, *args: Any, **kwargs: Any) -> Any:
        """Forward computation using single call to `update`.

        This can be done when the global metric state is a sinple reduction of batch states. This can be unsafe for
        certain metric cases but is also the fastest way to both accumulate globally and compute locally.
        """
        # store global state and reset to default
        global_state = {attr: getattr(self, attr) for attr in self._defaults}
        _update_count = self._update_count
        self.reset()

        # local syncronization settings
        self._to_sync = self.dist_sync_on_step
        self._should_unsync = False
        _temp_compute_on_cpu = self.compute_on_cpu
        self.compute_on_cpu = False
        self._enable_grad = True  # allow grads for batch computation

        # calculate batch state and compute batch value
        self.update(*args, **kwargs)
        batch_val = self.compute()

        # reduce batch and global state
        self._update_count = _update_count + 1
        with torch.no_grad():
            self._reduce_states(global_state)

        # restore context
        self._is_synced = False
        self._should_unsync = True
        self._to_sync = self.sync_on_compute
        self._computed = None
        self._enable_grad = False
        self.compute_on_cpu = _temp_compute_on_cpu
        if self.compute_on_cpu:
            self._move_list_states_to_cpu()

        return batch_val

    def _reduce_states(self, incoming_state: Dict[str, Any]) -> None:
        """Add an incoming metric state to the current state of the metric.

        Args:
            incoming_state: a dict containing a metric state similar metric itself
        """
        for attr in self._defaults:
            local_state = getattr(self, attr)
            global_state = incoming_state[attr]
            reduce_fn = self._reductions[attr]
            if reduce_fn == dim_zero_sum:
                reduced = global_state + local_state
            elif reduce_fn == dim_zero_mean:
                reduced = ((self._update_count - 1) * global_state + local_state).float() / self._update_count
            elif reduce_fn == dim_zero_max:
                reduced = torch.max(global_state, local_state)
            elif reduce_fn == dim_zero_min:
                reduced = torch.min(global_state, local_state)
            elif reduce_fn == dim_zero_cat:
                reduced = global_state + local_state
            elif reduce_fn is None and isinstance(global_state, Tensor):
                reduced = torch.stack([global_state, local_state])
            elif reduce_fn is None and isinstance(global_state, list):
                reduced = _flatten([global_state, local_state])
            elif reduce_fn and callable(reduce_fn):
                reduced = reduce_fn(torch.stack([global_state, local_state]))
            else:
                raise TypeError(f"Unsupported reduce_fn: {reduce_fn}")
            setattr(self, attr, reduced)

    def _sync_dist(self, dist_sync_fn: Callable = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

        output_dict = apply_to_collection(
            input_dict,
            Tensor,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)

            if isinstance(output_dict[attr], list) and len(output_dict[attr]) == 0:
                setattr(self, attr, [])
                continue

            if isinstance(output_dict[attr][0], Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("reduction_fn must be callable or None")
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update: Callable) -> Callable:
        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1
            with torch.set_grad_enabled(self._enable_grad):
                try:
                    update(*args, **kwargs)
                except RuntimeError as err:
                    if "Expected all tensors to be on" in str(err):
                        raise RuntimeError(
                            "Encountered different devices in metric calculation (see stacktrace for details)."
                            " This could be due to the metric class not being on the same device as input."
                            f" Instead of `metric={self.__class__.__name__}(...)` try to do"
                            f" `metric={self.__class__.__name__}(...).to(device)` where"
                            " device corresponds to the device of the input."
                        ) from err
                    raise err

            if self.compute_on_cpu:
                self._move_list_states_to_cpu()

        return wrapped_func

    def _move_list_states_to_cpu(self) -> None:
        """Move list states to cpu to save GPU memory."""
        for key in self._defaults:
            current_val = getattr(self, key)
            if isinstance(current_val, Sequence):
                setattr(self, key, [cur_v.to("cpu") for cur_v in current_val])

    def sync(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> None:
        """Sync function for manually controlling when metrics states should be synced across processes.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            distributed_available: Function to determine if we are running inside a distributed setting

        Raises:
            TorchMetricsUserError:
                If the metric is already synced and ``sync`` is called again.
        """
        if self._is_synced and should_sync:
            raise TorchMetricsUserError("The Metric has already been synced.")

        if distributed_available is None and self.distributed_available_fn is not None:
            distributed_available = self.distributed_available_fn

        is_distributed = distributed_available() if callable(distributed_available) else None

        if not should_sync or not is_distributed:
            return

        if dist_sync_fn is None:
            dist_sync_fn = gather_all_tensors

        # cache prior to syncing
        self._cache = {attr: getattr(self, attr) for attr in self._defaults}

        # sync
        self._sync_dist(dist_sync_fn, process_group=process_group)
        self._is_synced = True

    def unsync(self, should_unsync: bool = True) -> None:
        """Unsync function for manually controlling when metrics states should be reverted back to their local states.

        Args:
            should_unsync: Whether to perform unsync
        """
        if not should_unsync:
            return

        if not self._is_synced:
            raise TorchMetricsUserError("The Metric has already been un-synced.")

        if self._cache is None:
            raise TorchMetricsUserError("The internal cache should exist to unsync the Metric.")

        # if we synced, restore to cache so that we can continue to accumulate un-synced state
        for attr, val in self._cache.items():
            setattr(self, attr, val)
        self._is_synced = False
        self._cache = None

    @contextmanager
    def sync_context(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        should_unsync: bool = True,
        distributed_available: Optional[Callable] = None,
    ) -> Generator:
        """Context manager to synchronize states.

        This context manager is used in distributed setting and makes sure that the local cache states are restored
        after yielding the syncronized state.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: `None` (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            should_unsync: Whether to restore the cache state so that the metrics can
                continue to be accumulated.
            distributed_available: Function to determine if we are running inside a distributed setting
        """
        self.sync(
            dist_sync_fn=dist_sync_fn,
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=distributed_available,
        )

        yield

        self.unsync(should_unsync=self._is_synced and should_unsync)

    def _wrap_compute(self, compute: Callable) -> Callable:
        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if self._update_count == 0:
                rank_zero_warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                    UserWarning,
                )

            # return cached value
            if self._computed is not None:
                return self._computed

            # compute relies on the sync context manager to gather the states across processes and apply reduction
            # if synchronization happened, the current rank accumulated states will be restored to keep
            # accumulation going if ``should_unsync=True``,
            with self.sync_context(
                dist_sync_fn=self.dist_sync_fn,
                should_sync=self._to_sync,
                should_unsync=self._should_unsync,
            ):
                value = _squeeze_if_scalar(compute(*args, **kwargs))

            if self.compute_with_cache:
                self._computed = value

            return value

        return wrapped_func

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method to update the state variables of your metric class."""

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value.

        This method will automatically synchronize state variables when running in distributed backend.
        """

    def plot(self, *_: Any, **__: Any) -> Any:
        """Override this method plot the metric value."""
        raise NotImplementedError

    def _plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor], Dict[str, Tensor], Sequence[Dict[str, Tensor]]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed
        """
        val = val if val is not None else self.compute()
        fig, ax = plot_single_or_multi_val(
            val,
            ax=ax,
            higher_is_better=self.higher_is_better,
            name=self.__class__.__name__,
            lower_bound=self.plot_lower_bound,
            upper_bound=self.plot_upper_bound,
            legend_name=self.plot_legend_name,
        )
        return fig, ax

    def reset(self) -> None:
        """Reset metric state variables to their default value."""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

        # reset internal states
        self._cache = None
        self._is_synced = False

    def clone(self) -> "Metric":
        """Make a copy of the metric."""
        return deepcopy(self)

    def __getstate__(self) -> Dict[str, Any]:
        """Get the current state, including all metric states, for the metric. Used for loading and saving a metric."""
        # ignore update and compute functions for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ["update", "compute", "_update_signature"]}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the metric, based on a input state. Used for loading and saving a metric."""
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore[method-assign]
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore[method-assign]

    def __setattr__(self, name: str, value: Any) -> None:
        """Overwrite default method to prevent specific attributes from being set by user."""
        if name in (
            "higher_is_better",
            "is_differentiable",
            "full_state_update",
            "plot_lower_bound",
            "plot_upper_bound",
            "plot_legend_name",
        ):
            raise RuntimeError(f"Can't change const `{name}`.")
        super().__setattr__(name, value)

    @property
    def device(self) -> "torch.device":
        """Return the device of the metric."""
        return self._device

    def type(self, dst_type: Union[str, torch.dtype]) -> "Metric":  # noqa: A003
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.
        """
        return self

    def float(self) -> "Metric":  # noqa: A003
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.
        """
        return self

    def double(self) -> "Metric":
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.
        """
        return self

    def half(self) -> "Metric":
        """Override default and prevent dtype casting.

        Please use :meth:`Metric.set_dtype` instead.
        """
        return self

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type (type or string): the desired type.
        """
        self._dtype_convert = True
        out = super().type(dst_type)
        out._dtype_convert = False
        return out

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> Module:
        """Overwrite `_apply` function such that we can also move metric states to the correct device.

        This method is called by the base ``nn.Module`` class whenever `.to`, `.cuda`, `.float`, `.half` etc. methods
        are called. Dtype conversion is garded and will only happen through the special `set_dtype` method.

        Args:
            fn: the function to apply
            exclude_state: list of state variables to exclude from applying the function, that then needs to be handled
                by the metric class itself.
        """
        this = super()._apply(fn)
        fs = str(fn)
        cond = any(f in fs for f in ["Module.type", "Module.half", "Module.float", "Module.double", "Module.bfloat16"])
        if not self._dtype_convert and cond:
            return this

        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if key in exclude_state:
                continue

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
                    f"Expected metric state to be either a Tensor or a list of Tensor, but encountered {current_val}"
                )

        # make sure to update the device attribute
        # if the dummy tensor moves device by fn function we should also update the attribute
        self._device = fn(torch.zeros(1, device=self.device)).device

        # Additional apply to forward cache and computed attributes (may be nested)
        if this._computed is not None:
            this._computed = apply_to_collection(this._computed, Tensor, fn)
        if this._forward_cache is not None:
            this._forward_cache = apply_to_collection(this._forward_cache, Tensor, fn)

        return this

    def persistent(self, mode: bool = False) -> None:
        """Change post-init if metric states should be saved to its state_dict."""
        for key in self._persistent:
            self._persistent[key] = mode

    def state_dict(  # type: ignore[override]  # todo
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        """Get the current state of metric as an dictionary.

        Args:
            destination: Optional dictionary, that if provided, the state of module will be updated into the dict and
                the same object is returned. Otherwise, an ``OrderedDict`` will be created and returned.
            prefix: optional string, a prefix added to parameter and buffer names to compose the keys in state_dict.
            keep_vars: by default the :class:`~torch.Tensor`s returned in the state dict are detached from autograd.
                If set to ``True``, detaching will not be performed.
        """
        destination: Dict[str, Union[torch.Tensor, List, Any]] = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars  # type: ignore[arg-type]
        )
        # Register metric states to be part of the state_dict
        for key in self._defaults:
            if not self._persistent[key]:
                continue
            current_val = getattr(self, key)
            if not keep_vars:
                if isinstance(current_val, Tensor):
                    current_val = current_val.detach()
                elif isinstance(current_val, list):
                    current_val = [cur_v.detach() if isinstance(cur_v, Tensor) else cur_v for cur_v in current_val]
            destination[prefix + key] = deepcopy(current_val)
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
        """Load metric states from state_dict."""
        for key in self._defaults:
            name = prefix + key
            if name in state_dict:
                setattr(self, key, state_dict.pop(name))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs such that they match the update signature of the metric."""
        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = self._update_signature.parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params and _sign_params[k].kind not in _params)
        }

        exists_var_keyword = any(v.kind == inspect.Parameter.VAR_KEYWORD for v in _sign_params.values())
        # if no kwargs filtered, return all kwargs as default
        if not filtered_kwargs and not exists_var_keyword:
            # no kwargs in update signature -> don't return any kwargs
            return {}
        if exists_var_keyword:
            # kwargs found in update signature -> return all kwargs to be sure to not omit any.
            # filtering logic is likely implemented within the update call.
            return kwargs
        return filtered_kwargs

    def __hash__(self) -> int:
        """Return an unique hash of the metric.

        The hash depends on both the class itself but also the current metric state, which therefore enforces that two
        instances of the same metrics never have the same hash even if they have been updated on the same data.
        """
        # we need to add the id here, since PyTorch requires a module hash to be unique.
        # Internally, PyTorch nn.Module relies on that for children discovery
        # (see https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/module.py#L1544)
        # For metrics that include tensors it is not a problem,
        # since their hash is unique based on the memory location but we cannot rely on that for every metric.
        hash_vals = [self.__class__.__name__, id(self)]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))

    def __add__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the addition operator."""
        return CompositionalMetric(torch.add, self, other)

    def __and__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical and operator."""
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __eq__(  # type: ignore[override]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the equal operator."""
        return CompositionalMetric(torch.eq, self, other)

    def __floordiv__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the floor division operator."""
        return CompositionalMetric(torch.floor_divide, self, other)

    def __ge__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the greater than or equal operator."""
        return CompositionalMetric(torch.ge, self, other)

    def __gt__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the greater than operator."""
        return CompositionalMetric(torch.gt, self, other)

    def __le__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the less than or equal operator."""
        return CompositionalMetric(torch.le, self, other)

    def __lt__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the less than operator."""
        return CompositionalMetric(torch.lt, self, other)

    def __matmul__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the matrix multiplication operator."""
        return CompositionalMetric(torch.matmul, self, other)

    def __mod__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the remainder operator."""
        return CompositionalMetric(torch.fmod, self, other)

    def __mul__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the multiplication operator."""
        return CompositionalMetric(torch.mul, self, other)

    def __ne__(  # type: ignore[override]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the not equal operator."""
        return CompositionalMetric(torch.ne, self, other)

    def __or__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical or operator."""
        return CompositionalMetric(torch.bitwise_or, self, other)

    def __pow__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the exponential/power operator."""
        return CompositionalMetric(torch.pow, self, other)

    def __radd__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the addition operator."""
        return CompositionalMetric(torch.add, other, self)

    def __rand__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical and operator."""
        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __rfloordiv__(self, other: "CompositionalMetric") -> "Metric":
        """Construct compositional metric using the floor division operator."""
        return CompositionalMetric(torch.floor_divide, other, self)

    def __rmatmul__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the matrix multiplication operator."""
        return CompositionalMetric(torch.matmul, other, self)

    def __rmod__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the remainder operator."""
        return CompositionalMetric(torch.fmod, other, self)

    def __rmul__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the multiplication operator."""
        return CompositionalMetric(torch.mul, other, self)

    def __ror__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical or operator."""
        return CompositionalMetric(torch.bitwise_or, other, self)

    def __rpow__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the exponential/power operator."""
        return CompositionalMetric(torch.pow, other, self)

    def __rsub__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the subtraction operator."""
        return CompositionalMetric(torch.sub, other, self)

    def __rtruediv__(  # type: ignore[misc]
        self, other: Union["Metric", int, builtins.float, Tensor]
    ) -> "CompositionalMetric":
        """Construct compositional metric using the true divide operator."""
        return CompositionalMetric(torch.true_divide, other, self)

    def __rxor__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical xor operator."""
        return CompositionalMetric(torch.bitwise_xor, other, self)

    def __sub__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the subtraction operator."""
        return CompositionalMetric(torch.sub, self, other)

    def __truediv__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the true divide operator."""
        return CompositionalMetric(torch.true_divide, self, other)

    def __xor__(self, other: Union["Metric", int, builtins.float, Tensor]) -> "CompositionalMetric":
        """Construct compositional metric using the logical xor operator."""
        return CompositionalMetric(torch.bitwise_xor, self, other)

    def __abs__(self) -> "CompositionalMetric":
        """Construct compositional metric using the absolute operator."""
        return CompositionalMetric(torch.abs, self, None)

    def __inv__(self) -> "CompositionalMetric":
        """Construct compositional metric using the not operator."""
        return CompositionalMetric(torch.bitwise_not, self, None)

    def __invert__(self) -> "CompositionalMetric":
        """Construct compositional metric using the not operator."""
        return self.__inv__()

    def __neg__(self) -> "CompositionalMetric":
        """Construct compositional metric using absolute negative operator."""
        return CompositionalMetric(_neg, self, None)

    def __pos__(self) -> "CompositionalMetric":
        """Construct compositional metric using absolute operator."""
        return CompositionalMetric(torch.abs, self, None)

    def __getitem__(self, idx: int) -> "CompositionalMetric":
        """Construct compositional metric using the get item operator."""
        return CompositionalMetric(lambda x: x[idx], self, None)

    def __getnewargs__(self) -> Tuple:
        """Needed method for construction of new metrics __new__ method."""
        return tuple(
            Metric.__str__(self),
        )

    __iter__ = None


def _neg(x: Tensor) -> Tensor:
    return -torch.abs(x)


class CompositionalMetric(Metric):
    """Composition of two metrics with a specific operator which will be executed upon metrics compute."""

    def __init__(
        self,
        operator: Callable,
        metric_a: Union[Metric, int, float, Tensor],
        metric_b: Union[Metric, int, float, Tensor, None],
    ) -> None:
        """Class for creating compositions of metrics.

        This metric class is the output of adding, multiplying etc. any other metric. The metric re-implements the
        standard ``update``, ``forward``, ``reset`` and ``compute`` methods to redirect the arguments to the metrics
        that formed this composition.

        Args:
            operator:
                The operator taking in one (if metric_b is None) or two arguments. Will be applied to outputs of
                metric_a.compute() and (optionally if metric_b is not None) metric_b.compute()
            metric_a:
                First metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None.
        """
        super().__init__()

        self.op = operator

        if isinstance(metric_a, Tensor):
            self.register_buffer("metric_a", metric_a, persistent=False)
        else:
            self.metric_a = metric_a

        if isinstance(metric_b, Tensor):
            self.register_buffer("metric_b", metric_b, persistent=False)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None) -> None:
        """No syncing required here. syncing will be done in metric_a and metric_b."""

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Redirect the call to the input which the conposition was formed from."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **self.metric_a._filter_kwargs(**kwargs))

        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **self.metric_b._filter_kwargs(**kwargs))

    def compute(self) -> Any:
        """Redirect the call to the input which the conposition was formed from."""
        # also some parsing for kwargs?
        val_a = self.metric_a.compute() if isinstance(self.metric_a, Metric) else self.metric_a
        val_b = self.metric_b.compute() if isinstance(self.metric_b, Metric) else self.metric_b

        if val_b is None:
            return self.op(val_a)

        return self.op(val_a, val_b)

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate metric on current batch and accumulate to global state."""
        val_a = (
            self.metric_a(*args, **self.metric_a._filter_kwargs(**kwargs))
            if isinstance(self.metric_a, Metric)
            else self.metric_a
        )
        val_b = (
            self.metric_b(*args, **self.metric_b._filter_kwargs(**kwargs))
            if isinstance(self.metric_b, Metric)
            else self.metric_b
        )

        if val_a is None:
            self._forward_cache = None
            return self._forward_cache

        if val_b is None:
            if isinstance(self.metric_b, Metric):
                self._forward_cache = None
                return self._forward_cache

            # Unary op
            self._forward_cache = self.op(val_a)
            return self._forward_cache

        # Binary op
        self._forward_cache = self.op(val_a, val_b)
        return self._forward_cache

    def reset(self) -> None:
        """Redirect the call to the input which the conposition was formed from."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False) -> None:
        """Change if metric state is persistent (save as part of state_dict) or not.

        Args:
            mode: bool indicating if all states should be persistent or not

        """
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        """Return a representation of the compositional metric, including the two inputs it was formed from."""
        _op_metrics = f"(\n  {self.op.__name__}(\n    {self.metric_a!r},\n    {self.metric_b!r}\n  )\n)"
        return self.__class__.__name__ + _op_metrics

    def _wrap_compute(self, compute: Callable) -> Callable:
        """No wrapping nessesary for compositional metrics."""
        return compute
