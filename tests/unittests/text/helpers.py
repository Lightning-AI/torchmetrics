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
import pickle
import sys
from functools import partial, wraps
from typing import Any, Callable, Optional, Sequence, Union

import pytest
import torch
from torch import Tensor
from torch.multiprocessing import set_start_method

from torchmetrics import Metric
from unittests.helpers.testers import MetricTester, _assert_allclose, _assert_requires_grad, _assert_tensor

try:
    set_start_method("spawn")
except RuntimeError:
    pass


TEXT_METRIC_INPUT = Union[Sequence[str], Sequence[Sequence[str]], Sequence[Sequence[Sequence[str]]]]
NUM_BATCHES = 2


def _class_test(
    rank: int,
    worldsize: int,
    preds: TEXT_METRIC_INPUT,
    targets: TEXT_METRIC_INPUT,
    metric_class: Metric,
    sk_metric: Callable,
    dist_sync_on_step: bool,
    metric_args: dict = None,
    check_dist_sync_on_step: bool = True,
    check_batch: bool = True,
    atol: float = 1e-8,
    device: str = "cpu",
    fragment_kwargs: bool = False,
    check_scriptable: bool = True,
    key: str = None,
    **kwargs_update: Any,
):
    """Utility function doing the actual comparison between class metric and reference metric.

    Args:
        rank: rank of current process
        worldsize: number of processes
        preds: Sequence of predicted tokens or predicted sentences
        targets: Sequence of target tokens or target sentences
        metric_class: metric class that should be tested
        sk_metric: callable function that is used for comparison
        dist_sync_on_step: bool, if true will synchronize metric state across
            processes at each ``forward()``
        metric_args: dict with additional arguments used for class initialization
        check_dist_sync_on_step: bool, if true will check if the metric is also correctly
            calculated per batch per device (and not just at the end)
        check_batch: bool, if true will check if the metric is also correctly
            calculated across devices for each batch (and not just at the end)
        device: determine which device to run on, either 'cuda' or 'cpu'
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `targets` among processes
        key: The key passed onto the `_assert_allclose` to compare the respective metric from the Dict output against
            the sk_metric.
        kwargs_update: Additional keyword arguments that will be passed with preds and
            targets when running update on the metric.
    """
    if not metric_args:
        metric_args = {}

    # Instanciate metric
    metric = metric_class(dist_sync_on_step=dist_sync_on_step, **metric_args)

    # check that the metric is scriptable
    if check_scriptable:
        torch.jit.script(metric)

    # move to device
    metric = metric.to(device)
    kwargs_update = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)

    for i in range(rank, NUM_BATCHES, worldsize):
        batch_kwargs_update = {k: v[i] if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}
        batch_result = metric(preds[i], targets[i], **batch_kwargs_update)

        if metric.dist_sync_on_step and check_dist_sync_on_step and rank == 0:
            # Concatenation of Sequence of strings
            ddp_preds = type(preds)()
            ddp_targets = type(targets)()
            for r in range(worldsize):
                ddp_preds = ddp_preds + preds[i + r]
                ddp_targets = ddp_targets + targets[i + r]
            ddp_kwargs_upd = {
                k: torch.cat([v[i + r] for r in range(worldsize)]).cpu() if isinstance(v, Tensor) else v
                for k, v in (kwargs_update if fragment_kwargs else batch_kwargs_update).items()
            }

            sk_batch_result = sk_metric(ddp_preds, ddp_targets, **ddp_kwargs_upd)
            _assert_allclose(batch_result, sk_batch_result, atol=atol, key=key)

        elif check_batch and not metric.dist_sync_on_step:
            batch_kwargs_update = {
                k: v.cpu() if isinstance(v, Tensor) else v
                for k, v in (batch_kwargs_update if fragment_kwargs else kwargs_update).items()
            }
            sk_batch_result = sk_metric(preds[i], targets[i], **batch_kwargs_update)
            _assert_allclose(batch_result, sk_batch_result, atol=atol, key=key)

    # check that metrics are hashable
    assert hash(metric)

    # check on all batches on all ranks
    result = metric.compute()
    _assert_tensor(result, key=key)

    # Concatenation of Sequence of strings
    total_preds = type(preds)()
    total_targets = type(targets)()
    for i in range(NUM_BATCHES):
        total_preds = total_preds + preds[i]
        total_targets = total_targets + targets[i]
    total_kwargs_update = {
        k: torch.cat([v[i] for i in range(NUM_BATCHES)]).cpu() if isinstance(v, Tensor) else v
        for k, v in kwargs_update.items()
    }
    sk_result = sk_metric(total_preds, total_targets, **total_kwargs_update)
    # assert after aggregation
    _assert_allclose(result, sk_result, atol=atol, key=key)


def _functional_test(
    preds: TEXT_METRIC_INPUT,
    targets: TEXT_METRIC_INPUT,
    metric_functional: Callable,
    sk_metric: Callable,
    metric_args: dict = None,
    atol: float = 1e-8,
    device: str = "cpu",
    fragment_kwargs: bool = False,
    key: str = None,
    **kwargs_update,
):
    """Utility function doing the actual comparison between functional metric and reference metric.

    Args:
        preds: torch tensor with predictions
        targets: torch tensor with targets
        metric_functional: metric functional that should be tested
        sk_metric: callable function that is used for comparison
        metric_args: dict with additional arguments used for class initialization
        device: determine which device to run on, either 'cuda' or 'cpu'
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `targets` among processes
        key: The key passed onto the `_assert_allclose` to compare the respective metric from the Dict output against
            the sk_metric.
        kwargs_update: Additional keyword arguments that will be passed with preds and
            targets when running update on the metric.
    """
    if not metric_args:
        metric_args = {}

    metric = partial(metric_functional, **metric_args)

    # Move to device
    kwargs_update = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}

    for i in range(NUM_BATCHES):
        extra_kwargs = {k: v[i] if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}
        tm_result = metric(preds[i], targets[i], **extra_kwargs)

        extra_kwargs = {
            k: v.cpu() if isinstance(v, Tensor) else v
            for k, v in (extra_kwargs if fragment_kwargs else kwargs_update).items()
        }
        sk_result = sk_metric(preds[i], targets[i], **extra_kwargs)

        # assert its the same
        _assert_allclose(tm_result, sk_result, atol=atol, key=key)


def _assert_half_support(
    metric_module: Metric,
    metric_functional: Callable,
    preds: TEXT_METRIC_INPUT,
    targets: TEXT_METRIC_INPUT,
    device: str = "cpu",
    **kwargs_update,
):
    """Test if an metric can be used with half precision tensors.

    Args:
        metric_module: the metric module to test
        metric_functional: the metric functional to test
        preds: torch tensor with predictions
        targets: torch tensor with targets
        device: determine device, either "cpu" or "cuda"
        kwargs_update: Additional keyword arguments that will be passed with preds and
                targets when running update on the metric.
    """
    y_hat = preds[0]
    y = targets[0]
    kwargs_update = {
        k: (v[0].half() if v.is_floating_point() else v[0]).to(device) if isinstance(v, Tensor) else v
        for k, v in kwargs_update.items()
    }
    metric_module = metric_module.to(device)
    _assert_tensor(metric_module(y_hat, y, **kwargs_update))
    _assert_tensor(metric_functional(y_hat, y, **kwargs_update))


class TextTester(MetricTester):
    """Class used for efficiently run alot of parametrized tests in ddp mode. Makes sure that ddp is only setup
    once and that pool of processes are used for all tests.

    All tests for text metrics should subclass from this and implement a new method called `test_metric_name` where the
    method `self.run_metric_test` is called inside.
    """

    def run_functional_metric_test(
        self,
        preds: TEXT_METRIC_INPUT,
        targets: TEXT_METRIC_INPUT,
        metric_functional: Callable,
        sk_metric: Callable,
        metric_args: dict = None,
        fragment_kwargs: bool = False,
        key: str = None,
        **kwargs_update,
    ):
        """Main method that should be used for testing functions. Call this inside testing method.

        Args:
            preds: torch tensor with predictions
            targets: torch tensor with targets
            metric_functional: metric class that should be tested
            sk_metric: callable function that is used for comparison
            metric_args: dict with additional arguments used for class initialization
            fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `targets` among processes
            key: The key passed onto the `_assert_allclose` to compare the respective metric from the Dict output
                against the sk_metric.
            kwargs_update: Additional keyword arguments that will be passed with preds and
                targets when running update on the metric.
        """
        device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

        _functional_test(
            preds=preds,
            targets=targets,
            metric_functional=metric_functional,
            sk_metric=sk_metric,
            metric_args=metric_args,
            atol=self.atol,
            device=device,
            fragment_kwargs=fragment_kwargs,
            key=key,
            **kwargs_update,
        )

    def run_class_metric_test(
        self,
        ddp: bool,
        preds: TEXT_METRIC_INPUT,
        targets: TEXT_METRIC_INPUT,
        metric_class: Metric,
        sk_metric: Callable,
        dist_sync_on_step: bool,
        metric_args: dict = None,
        check_dist_sync_on_step: bool = True,
        check_batch: bool = True,
        fragment_kwargs: bool = False,
        check_scriptable: bool = True,
        key: str = None,
        **kwargs_update,
    ):
        """Main method that should be used for testing class. Call this inside testing methods.

        Args:
            ddp: bool, if running in ddp mode or not
            preds: torch tensor with predictions
            targets: torch tensor with targets
            metric_class: metric class that should be tested
            sk_metric: callable function that is used for comparison
            dist_sync_on_step: bool, if true will synchronize metric state across
                processes at each ``forward()``
            metric_args: dict with additional arguments used for class initialization
            check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                calculated per batch per device (and not just at the end)
            check_batch: bool, if true will check if the metric is also correctly
                calculated across devices for each batch (and not just at the end)
            fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `targets` among processes
            check_scriptable:
            key: The key passed onto the `_assert_allclose` to compare the respective metric from the Dict output
                against the sk_metric.
            kwargs_update: Additional keyword arguments that will be passed with preds and
                targets when running update on the metric.
        """
        if not metric_args:
            metric_args = {}
        if ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")

            self.pool.starmap(
                partial(
                    _class_test,
                    preds=preds,
                    targets=targets,
                    metric_class=metric_class,
                    sk_metric=sk_metric,
                    dist_sync_on_step=dist_sync_on_step,
                    metric_args=metric_args,
                    check_dist_sync_on_step=check_dist_sync_on_step,
                    check_batch=check_batch,
                    atol=self.atol,
                    fragment_kwargs=fragment_kwargs,
                    check_scriptable=check_scriptable,
                    key=key,
                    **kwargs_update,
                ),
                [(rank, self.poolSize) for rank in range(self.poolSize)],
            )
        else:
            device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

            _class_test(
                rank=0,
                worldsize=1,
                preds=preds,
                targets=targets,
                metric_class=metric_class,
                sk_metric=sk_metric,
                dist_sync_on_step=dist_sync_on_step,
                metric_args=metric_args,
                check_dist_sync_on_step=check_dist_sync_on_step,
                check_batch=check_batch,
                atol=self.atol,
                device=device,
                fragment_kwargs=fragment_kwargs,
                check_scriptable=check_scriptable,
                key=key,
                **kwargs_update,
            )

    @staticmethod
    def run_precision_test_cpu(
        preds: TEXT_METRIC_INPUT,
        targets: TEXT_METRIC_INPUT,
        metric_module: Metric,
        metric_functional: Callable,
        metric_args: dict = None,
        **kwargs_update,
    ):
        """Test if a metric can be used with half precision tensors on cpu
        Args:
            preds: torch tensor with predictions
            targets: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            kwargs_update: Additional keyword arguments that will be passed with preds and
                targets when running update on the metric.
        """
        metric_args = metric_args or {}
        _assert_half_support(
            metric_module(**metric_args), metric_functional, preds, targets, device="cpu", **kwargs_update
        )

    @staticmethod
    def run_precision_test_gpu(
        preds: TEXT_METRIC_INPUT,
        targets: TEXT_METRIC_INPUT,
        metric_module: Metric,
        metric_functional: Callable,
        metric_args: dict = None,
        **kwargs_update,
    ):
        """Test if a metric can be used with half precision tensors on gpu
        Args:
            preds: torch tensor with predictions
            targets: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            kwargs_update: Additional keyword arguments that will be passed with preds and
                targets when running update on the metric.
        """
        metric_args = metric_args or {}
        _assert_half_support(
            metric_module(**metric_args), metric_functional, preds, targets, device="cuda", **kwargs_update
        )

    @staticmethod
    def run_differentiability_test(
        preds: TEXT_METRIC_INPUT,
        targets: TEXT_METRIC_INPUT,
        metric_module: Metric,
        metric_functional: Callable,
        metric_args: dict = None,
        key: str = None,
    ):
        """Test if a metric is differentiable or not.

        Args:
            preds: torch tensor with predictions
            targets: torch tensor with targets
            metric_module: the metric module to test
            metric_functional:
            metric_args: dict with additional arguments used for class initialization
            key: The key passed onto the `_assert_allclose` to compare the respective metric from the Dict output
                against the sk_metric.
        """
        metric_args = metric_args or {}
        # only floating point tensors can require grad
        metric = metric_module(**metric_args)
        out = metric(preds[0], targets[0])

        # Check if requires_grad matches is_differentiable attribute
        _assert_requires_grad(metric, out, key=key)

        if metric.is_differentiable:
            # check for numerical correctness
            assert torch.autograd.gradcheck(partial(metric_functional, **metric_args), (preds[0], targets[0]))


def skip_on_connection_issues(reason: str = "Unable to load checkpoints from HuggingFace `transformers`."):
    """Wrapper which handles HF-related tests if they fail due to connection issues.

    The tests run normally if no connection issue arises, and they're marked as skipped otherwise.
    """
    _error_msg_start = "We couldn't connect to"

    def test_decorator(function: Callable, *args: Any, **kwargs: Any) -> Optional[Callable]:
        @wraps(function)
        def run_test(*args: Any, **kwargs: Any) -> Optional[Any]:
            try:
                return function(*args, **kwargs)
            except OSError as ex:
                if _error_msg_start not in str(ex):
                    raise ex
                pytest.skip(reason)

        return run_test

    return test_decorator
