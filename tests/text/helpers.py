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
from functools import partial
from typing import Any, Callable, Sequence, Union

import pytest
import torch
from torch import Tensor
from torch.multiprocessing import Pool, set_start_method

from tests.helpers.testers import (
    NUM_BATCHES,
    NUM_PROCESSES,
    _assert_allclose,
    _assert_requires_grad,
    _assert_tensor,
    setup_ddp,
)
from torchmetrics import Metric

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def _class_test(
    rank: int,
    worldsize: int,
    preds: Union[Sequence[str], Sequence[Sequence[str]]],
    target: Union[Sequence[str], Sequence[Sequence[str]]],
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
    **kwargs_update: Any,
):
    """Utility function doing the actual comparison between lightning class metric and reference metric.

    Args:
        rank: rank of current process
        worldsize: number of processes
        preds: Sequence of predictions tokens or predicted sentences
        target: Sequence of target tokens or target sentences
        metric_class: lightning metric class that should be tested
        sk_metric: callable function that is used for comparison
        dist_sync_on_step: bool, if true will synchronize metric state across
            processes at each ``forward()``
        metric_args: dict with additional arguments used for class initialization
        check_dist_sync_on_step: bool, if true will check if the metric is also correctly
            calculated per batch per device (and not just at the end)
        check_batch: bool, if true will check if the metric is also correctly
            calculated across devices for each batch (and not just at the end)
        device: determine which device to run on, either 'cuda' or 'cpu'
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
        kwargs_update: Additional keyword arguments that will be passed with preds and
            target when running update on the metric.
    """
    if not metric_args:
        metric_args = {}

    # Instanciate lightning metric
    metric = metric_class(
        compute_on_step=check_dist_sync_on_step or check_batch, dist_sync_on_step=dist_sync_on_step, **metric_args
    )

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

        batch_result = metric(preds[i], target[i], **batch_kwargs_update)

        if metric.dist_sync_on_step and check_dist_sync_on_step and rank == 0:
            ddp_preds = [preds[i + r] for r in range(worldsize)]
            ddp_target = [target[i + r] for r in range(worldsize)]
            ddp_kwargs_upd = {
                k: torch.cat([v[i + r] for r in range(worldsize)]).cpu() if isinstance(v, Tensor) else v
                for k, v in (kwargs_update if fragment_kwargs else batch_kwargs_update).items()
            }

            sk_batch_result = sk_metric(ddp_preds, ddp_target, **ddp_kwargs_upd)
            _assert_allclose(batch_result, sk_batch_result, atol=atol)

        elif check_batch and not metric.dist_sync_on_step:
            batch_kwargs_update = {
                k: v.cpu() if isinstance(v, Tensor) else v
                for k, v in (batch_kwargs_update if fragment_kwargs else kwargs_update).items()
            }
            sk_batch_result = sk_metric(preds[i], target[i], **batch_kwargs_update)
            _assert_allclose(batch_result, sk_batch_result, atol=atol)

    # check on all batches on all ranks
    result = metric.compute()
    _assert_tensor(result)

    total_preds = torch.cat([preds[i] for i in range(NUM_BATCHES)]).cpu()
    total_target = torch.cat([target[i] for i in range(NUM_BATCHES)]).cpu()
    total_kwargs_update = {
        k: torch.cat([v[i] for i in range(NUM_BATCHES)]).cpu() if isinstance(v, Tensor) else v
        for k, v in kwargs_update.items()
    }
    sk_result = sk_metric(total_preds, total_target, **total_kwargs_update)

    # assert after aggregation
    _assert_allclose(result, sk_result, atol=atol)


def _functional_test(
    preds: Union[Sequence[str], Sequence[Sequence[str]]],
    target: Union[Sequence[str], Sequence[Sequence[str]]],
    metric_functional: Callable,
    sk_metric: Callable,
    metric_args: dict = None,
    atol: float = 1e-8,
    device: str = "cpu",
    fragment_kwargs: bool = False,
    **kwargs_update,
):
    """Utility function doing the actual comparison between lightning functional metric and reference metric.

    Args:
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_functional: lightning metric functional that should be tested
        sk_metric: callable function that is used for comparison
        metric_args: dict with additional arguments used for class initialization
        device: determine which device to run on, either 'cuda' or 'cpu'
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
        kwargs_update: Additional keyword arguments that will be passed with preds and
            target when running update on the metric.
    """
    if not metric_args:
        metric_args = {}

    metric = partial(metric_functional, **metric_args)

    # Move to device
    kwargs_update = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}

    for i in range(NUM_BATCHES):
        extra_kwargs = {k: v[i] if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}
        lightning_result = metric(preds[i], target[i], **extra_kwargs)
        extra_kwargs = {
            k: v.cpu() if isinstance(v, Tensor) else v
            for k, v in (extra_kwargs if fragment_kwargs else kwargs_update).items()
        }
        sk_result = sk_metric(preds[i], target[i], **extra_kwargs)

        # assert its the same
        _assert_allclose(lightning_result, sk_result, atol=atol)


def _assert_half_support(
    metric_module: Metric,
    metric_functional: Callable,
    preds: Union[Sequence[str], Sequence[Sequence[str]]],
    target: Union[Sequence[str], Sequence[Sequence[str]]],
    device: str = "cpu",
    **kwargs_update,
):
    """Test if an metric can be used with half precision tensors.

    Args:
        metric_module: the metric module to test
        metric_functional: the metric functional to test
        preds: torch tensor with predictions
        target: torch tensor with targets
        device: determine device, either "cpu" or "cuda"
        kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.
    """
    y_hat = preds[0]
    y = target[0]
    kwargs_update = {
        k: (v[0].half() if v.is_floating_point() else v[0]).to(device) if isinstance(v, Tensor) else v
        for k, v in kwargs_update.items()
    }
    metric_module = metric_module.to(device)
    _assert_tensor(metric_module(y_hat, y, **kwargs_update))
    _assert_tensor(metric_functional(y_hat, y, **kwargs_update))


class TextTester:
    """Class used for efficiently run alot of parametrized tests in ddp mode. Makes sure that ddp is only setup
    once and that pool of processes are used for all tests.

    All tests should subclass from this and implement a new method called     `test_metric_name` where the method
    `self.run_metric_test` is called inside.
    """

    atol = 1e-8

    def setup_class(self):
        """Setup the metric class.

        This will spawn the pool of workers that are used for metric testing and setup_ddp
        """

        self.poolSize = NUM_PROCESSES
        self.pool = Pool(processes=self.poolSize)
        self.pool.starmap(setup_ddp, [(rank, self.poolSize) for rank in range(self.poolSize)])

    def teardown_class(self):
        """Close pool of workers."""
        self.pool.close()
        self.pool.join()

    def run_functional_metric_test(
        self,
        preds: Union[Sequence[str], Sequence[Sequence[str]]],
        target: Union[Sequence[str], Sequence[Sequence[str]]],
        metric_functional: Callable,
        sk_metric: Callable,
        metric_args: dict = None,
        fragment_kwargs: bool = False,
        **kwargs_update,
    ):
        """Main method that should be used for testing functions. Call this inside testing method.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_functional: lightning metric class that should be tested
            sk_metric: callable function that is used for comparison
            metric_args: dict with additional arguments used for class initialization
            fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.
        """
        device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

        _functional_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            sk_metric=sk_metric,
            metric_args=metric_args,
            atol=self.atol,
            device=device,
            fragment_kwargs=fragment_kwargs,
            **kwargs_update,
        )

    def run_class_metric_test(
        self,
        ddp: bool,
        preds: Union[Sequence[str], Sequence[Sequence[str]]],
        target: Union[Sequence[str], Sequence[Sequence[str]]],
        metric_class: Metric,
        sk_metric: Callable,
        dist_sync_on_step: bool,
        metric_args: dict = None,
        check_dist_sync_on_step: bool = True,
        check_batch: bool = True,
        fragment_kwargs: bool = False,
        check_scriptable: bool = True,
        **kwargs_update,
    ):
        """Main method that should be used for testing class. Call this inside testing methods.

        Args:
            ddp: bool, if running in ddp mode or not
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_class: lightning metric class that should be tested
            sk_metric: callable function that is used for comparison
            dist_sync_on_step: bool, if true will synchronize metric state across
                processes at each ``forward()``
            metric_args: dict with additional arguments used for class initialization
            check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                calculated per batch per device (and not just at the end)
            check_batch: bool, if true will check if the metric is also correctly
                calculated across devices for each batch (and not just at the end)
            fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.
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
                    target=target,
                    metric_class=metric_class,
                    sk_metric=sk_metric,
                    dist_sync_on_step=dist_sync_on_step,
                    metric_args=metric_args,
                    check_dist_sync_on_step=check_dist_sync_on_step,
                    check_batch=check_batch,
                    atol=self.atol,
                    fragment_kwargs=fragment_kwargs,
                    check_scriptable=check_scriptable,
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
                target=target,
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
                **kwargs_update,
            )

    @staticmethod
    def run_precision_test_cpu(
        preds: Union[Sequence[str], Sequence[Sequence[str]]],
        target: Union[Sequence[str], Sequence[Sequence[str]]],
        metric_module: Metric,
        metric_functional: Callable,
        metric_args: dict = None,
        **kwargs_update,
    ):
        """Test if a metric can be used with half precision tensors on cpu
        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.
        """
        metric_args = metric_args or {}
        _assert_half_support(
            metric_module(**metric_args), metric_functional, preds, target, device="cpu", **kwargs_update
        )

    @staticmethod
    def run_precision_test_gpu(
        preds: Union[Sequence[str], Sequence[Sequence[str]]],
        target: Union[Sequence[str], Sequence[Sequence[str]]],
        metric_module: Metric,
        metric_functional: Callable,
        metric_args: dict = None,
        **kwargs_update,
    ):
        """Test if a metric can be used with half precision tensors on gpu
        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.
        """
        metric_args = metric_args or {}
        _assert_half_support(
            metric_module(**metric_args), metric_functional, preds, target, device="cuda", **kwargs_update
        )

    @staticmethod
    def run_differentiability_test(
        preds: Union[Sequence[str], Sequence[Sequence[str]]],
        target: Union[Sequence[str], Sequence[Sequence[str]]],
        metric_module: Metric,
        metric_functional: Callable,
        metric_args: dict = None,
    ):
        """Test if a metric is differentiable or not.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_args: dict with additional arguments used for class initialization
        """
        metric_args = metric_args or {}
        # only floating point tensors can require grad
        metric = metric_module(**metric_args)
        out = metric(preds[0], target[0])

        # Check if requires_grad matches is_differentiable attribute
        _assert_requires_grad(metric, out)

        if metric.is_differentiable:
            # check for numerical correctness
            assert torch.autograd.gradcheck(partial(metric_functional, **metric_args), (preds[0], target[0]))
