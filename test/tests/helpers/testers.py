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
import os
import pickle
import sys
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pytest
import torch
from torch import Tensor, tensor
from torch.multiprocessing import Pool, set_start_method

from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MAPMetricResults
from torchmetrics.utilities.data import apply_to_collection

try:
    set_start_method("spawn")
except RuntimeError:
    pass

NUM_PROCESSES = 2
NUM_BATCHES = 4  # Need to be divisible with the number of processes
BATCH_SIZE = 32
# NUM_BATCHES = 10 if torch.cuda.is_available() else 4
# BATCH_SIZE = 64 if torch.cuda.is_available() else 32
NUM_CLASSES = 5
EXTRA_DIM = 3
THRESHOLD = 0.5

MAX_PORT = 8100
START_PORT = 8088
CURRENT_PORT = START_PORT


def setup_ddp(rank, world_size):
    """Setup ddp environment."""
    global CURRENT_PORT

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(CURRENT_PORT)

    CURRENT_PORT += 1
    if CURRENT_PORT > MAX_PORT:
        CURRENT_PORT = START_PORT

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def _assert_allclose(pl_result: Any, sk_result: Any, atol: float = 1e-8, key: Optional[str] = None) -> None:
    """Utility function for recursively asserting that two results are within a certain tolerance."""
    # single output compare
    if isinstance(pl_result, Tensor):
        assert np.allclose(pl_result.detach().cpu().numpy(), sk_result, atol=atol, equal_nan=True)
    # multi output compare
    elif isinstance(pl_result, Sequence):
        for pl_res, sk_res in zip(pl_result, sk_result):
            _assert_allclose(pl_res, sk_res, atol=atol)
    elif isinstance(pl_result, Dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        assert np.allclose(pl_result[key].detach().cpu().numpy(), sk_result, atol=atol, equal_nan=True)
    else:
        raise ValueError("Unknown format for comparison")


def _assert_tensor(pl_result: Any, key: Optional[str] = None) -> None:
    """Utility function for recursively checking that some input only consists of torch tensors."""
    if isinstance(pl_result, Sequence):
        for plr in pl_result:
            _assert_tensor(plr)
    elif isinstance(pl_result, Dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        assert isinstance(pl_result[key], Tensor)
    elif isinstance(pl_result, MAPMetricResults):
        for val_index in [a for a in dir(pl_result) if not a.startswith("__")]:
            assert isinstance(pl_result[val_index], Tensor)
    else:
        assert isinstance(pl_result, Tensor)


def _assert_requires_grad(metric: Metric, pl_result: Any, key: Optional[str] = None) -> None:
    """Utility function for recursively asserting that metric output is consistent with the `is_differentiable`
    attribute."""
    if isinstance(pl_result, Sequence):
        for plr in pl_result:
            _assert_requires_grad(metric, plr, key=key)
    elif isinstance(pl_result, Dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        assert metric.is_differentiable == pl_result[key].requires_grad
    else:
        assert metric.is_differentiable == pl_result.requires_grad


def _class_test(
    rank: int,
    worldsize: int,
    preds: Union[Tensor, List[Dict[str, Tensor]]],
    target: Union[Tensor, List[Dict[str, Tensor]]],
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
    check_state_dict: bool = True,
    **kwargs_update: Any,
):
    """Utility function doing the actual comparison between class metric and reference metric.

    Args:
        rank: rank of current process
        worldsize: number of processes
        preds: torch tensor with predictions
        target: torch tensor with targets
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
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
        kwargs_update: Additional keyword arguments that will be passed with preds and
            target when running update on the metric.
    """
    assert len(preds) == len(target)
    num_batches = len(preds)

    if not metric_args:
        metric_args = {}

    # Instantiate metric
    metric = metric_class(dist_sync_on_step=dist_sync_on_step, **metric_args)
    with pytest.raises(RuntimeError):
        metric.is_differentiable = not metric.is_differentiable
    with pytest.raises(RuntimeError):
        metric.higher_is_better = not metric.higher_is_better

    # check that the metric is scriptable
    if check_scriptable:
        torch.jit.script(metric)

    # move to device
    metric = metric.to(device)
    preds = apply_to_collection(preds, Tensor, lambda x: x.to(device))
    target = apply_to_collection(target, Tensor, lambda x: x.to(device))

    kwargs_update = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)

    for i in range(rank, num_batches, worldsize):
        batch_kwargs_update = {k: v[i] if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}

        batch_result = metric(preds[i], target[i], **batch_kwargs_update)

        if metric.dist_sync_on_step and check_dist_sync_on_step and rank == 0:
            if isinstance(preds, Tensor):
                ddp_preds = torch.cat([preds[i + r] for r in range(worldsize)]).cpu()
                ddp_target = torch.cat([target[i + r] for r in range(worldsize)]).cpu()
            else:
                ddp_preds = [preds[i + r] for r in range(worldsize)]
                ddp_target = [target[i + r] for r in range(worldsize)]
            ddp_kwargs_upd = {
                k: torch.cat([v[i + r] for r in range(worldsize)]).cpu() if isinstance(v, Tensor) else v
                for k, v in (kwargs_update if fragment_kwargs else batch_kwargs_update).items()
            }

            sk_batch_result = sk_metric(ddp_preds, ddp_target, **ddp_kwargs_upd)
            if isinstance(batch_result, dict):
                for key in batch_result:
                    _assert_allclose(batch_result, sk_batch_result[key].numpy(), atol=atol, key=key)
            else:
                _assert_allclose(batch_result, sk_batch_result, atol=atol)

        elif check_batch and not metric.dist_sync_on_step:
            batch_kwargs_update = {
                k: v.cpu() if isinstance(v, Tensor) else v
                for k, v in (batch_kwargs_update if fragment_kwargs else kwargs_update).items()
            }
            preds_ = preds[i].cpu() if isinstance(preds, Tensor) else preds[i]
            target_ = target[i].cpu() if isinstance(target, Tensor) else target[i]
            sk_batch_result = sk_metric(preds_, target_, **batch_kwargs_update)
            if isinstance(batch_result, dict):
                for key in batch_result.keys():
                    _assert_allclose(batch_result, sk_batch_result[key].numpy(), atol=atol, key=key)
            else:
                _assert_allclose(batch_result, sk_batch_result, atol=atol)

    # check that metrics are hashable

    assert hash(metric)

    # assert that state dict is empty
    if check_state_dict:
        assert metric.state_dict() == {}

    # check on all batches on all ranks
    result = metric.compute()
    if isinstance(result, dict):
        for key in result.keys():
            _assert_tensor(result, key=key)
    else:
        _assert_tensor(result)

    if isinstance(preds, Tensor):
        total_preds = torch.cat([preds[i] for i in range(num_batches)]).cpu()
        total_target = torch.cat([target[i] for i in range(num_batches)]).cpu()
    else:
        total_preds = [item for sublist in preds for item in sublist]
        total_target = [item for sublist in target for item in sublist]

    total_kwargs_update = {
        k: torch.cat([v[i] for i in range(num_batches)]).cpu() if isinstance(v, Tensor) else v
        for k, v in kwargs_update.items()
    }
    sk_result = sk_metric(total_preds, total_target, **total_kwargs_update)

    # assert after aggregation
    if isinstance(sk_result, dict):
        for key in sk_result.keys():
            _assert_allclose(result, sk_result[key].numpy(), atol=atol, key=key)
    else:
        _assert_allclose(result, sk_result, atol=atol)


def _functional_test(
    preds: Tensor,
    target: Tensor,
    metric_functional: Callable,
    sk_metric: Callable,
    metric_args: dict = None,
    atol: float = 1e-8,
    device: str = "cpu",
    fragment_kwargs: bool = False,
    **kwargs_update,
):
    """Utility function doing the actual comparison between functional metric and reference metric.

    Args:
        preds: torch tensor with predictions
        target: torch tensor with targets
        metric_functional: metric functional that should be tested
        sk_metric: callable function that is used for comparison
        metric_args: dict with additional arguments used for class initialization
        device: determine which device to run on, either 'cuda' or 'cpu'
        fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
        kwargs_update: Additional keyword arguments that will be passed with preds and
            target when running update on the metric.
    """
    assert preds.shape[0] == target.shape[0]
    num_batches = preds.shape[0]

    if not metric_args:
        metric_args = {}

    metric = partial(metric_functional, **metric_args)

    # move to device
    preds = preds.to(device)
    target = target.to(device)
    kwargs_update = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}

    for i in range(num_batches):
        extra_kwargs = {k: v[i] if isinstance(v, Tensor) else v for k, v in kwargs_update.items()}
        tm_result = metric(preds[i], target[i], **extra_kwargs)
        extra_kwargs = {
            k: v.cpu() if isinstance(v, Tensor) else v
            for k, v in (extra_kwargs if fragment_kwargs else kwargs_update).items()
        }
        sk_result = sk_metric(preds[i].cpu(), target[i].cpu(), **extra_kwargs)

        # assert its the same
        _assert_allclose(tm_result, sk_result, atol=atol)


def _assert_half_support(
    metric_module: Optional[Metric],
    metric_functional: Optional[Callable],
    preds: Tensor,
    target: Tensor,
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
    y_hat = preds[0].half().to(device) if preds[0].is_floating_point() else preds[0].to(device)
    y = target[0].half().to(device) if target[0].is_floating_point() else target[0].to(device)
    kwargs_update = {
        k: (v[0].half() if v.is_floating_point() else v[0]).to(device) if isinstance(v, Tensor) else v
        for k, v in kwargs_update.items()
    }
    if metric_module is not None:
        metric_module = metric_module.to(device)
        _assert_tensor(metric_module(y_hat, y, **kwargs_update))
    if metric_functional is not None:
        _assert_tensor(metric_functional(y_hat, y, **kwargs_update))


class MetricTester:
    """Class used for efficiently run alot of parametrized tests in ddp mode. Makes sure that ddp is only setup
    once and that pool of processes are used for all tests.

    All tests should subclass from this and implement a new method called     `test_metric_name` where the method
    `self.run_metric_test` is called inside.
    """

    atol: float = 1e-8
    poolSize: int
    pool: Pool

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
        preds: Tensor,
        target: Tensor,
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
            metric_functional: metric class that should be tested
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
        preds: Union[Tensor, List[Dict]],
        target: Union[Tensor, List[Dict]],
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
            metric_class: metric class that should be tested
            sk_metric: callable function that is used for comparison
            dist_sync_on_step: bool, if true will synchronize metric state across
                processes at each ``forward()``
            metric_args: dict with additional arguments used for class initialization
            check_dist_sync_on_step: bool, if true will check if the metric is also correctly
                calculated per batch per device (and not just at the end)
            check_batch: bool, if true will check if the metric is also correctly
                calculated across devices for each batch (and not just at the end)
            fragment_kwargs: whether tensors in kwargs should be divided as `preds` and `target` among processes
            check_scriptable:
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
        preds: Tensor,
        target: Tensor,
        metric_module: Optional[Metric] = None,
        metric_functional: Optional[Callable] = None,
        metric_args: Optional[dict] = None,
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
            metric_module(**metric_args) if metric_module is not None else None,
            metric_functional,
            preds,
            target,
            device="cpu",
            **kwargs_update,
        )

    @staticmethod
    def run_precision_test_gpu(
        preds: Tensor,
        target: Tensor,
        metric_module: Optional[Metric] = None,
        metric_functional: Optional[Callable] = None,
        metric_args: Optional[dict] = None,
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
            metric_module(**metric_args) if metric_module is not None else None,
            metric_functional,
            preds,
            target,
            device="cuda",
            **kwargs_update,
        )

    @staticmethod
    def run_differentiability_test(
        preds: Tensor,
        target: Tensor,
        metric_module: Metric,
        metric_functional: Optional[Callable] = None,
        metric_args: Optional[dict] = None,
    ):
        """Test if a metric is differentiable or not.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional:
            metric_args: dict with additional arguments used for class initialization
        """
        metric_args = metric_args or {}
        # only floating point tensors can require grad
        metric = metric_module(**metric_args)
        if preds.is_floating_point():
            preds.requires_grad = True
            out = metric(preds[0, :2], target[0, :2])

            # Check if requires_grad matches is_differentiable attribute
            _assert_requires_grad(metric, out)

            if metric.is_differentiable and metric_functional is not None:
                # check for numerical correctness
                assert torch.autograd.gradcheck(
                    partial(metric_functional, **metric_args), (preds[0, :2].double(), target[0, :2])
                )

            # reset as else it will carry over to other tests
            preds.requires_grad = False


class DummyMetric(Metric):
    name = "Dummy"
    full_state_update: Optional[bool] = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("x", tensor(0.0), dist_reduce_fx="sum")

    def update(self):
        pass

    def compute(self):
        pass


class DummyListMetric(Metric):
    name = "DummyList"
    full_state_update: Optional[bool] = True

    def __init__(self):
        super().__init__()
        self.add_state("x", [], dist_reduce_fx="cat")

    def update(self):
        pass

    def compute(self):
        pass


class DummyMetricSum(DummyMetric):
    def update(self, x):
        self.x += x

    def compute(self):
        return self.x


class DummyMetricDiff(DummyMetric):
    def update(self, y):
        self.x -= y

    def compute(self):
        return self.x


class DummyMetricMultiOutput(DummyMetricSum):
    def compute(self):
        return [self.x, self.x]
