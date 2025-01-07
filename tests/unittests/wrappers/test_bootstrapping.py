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
import operator
from functools import partial
from typing import Any

import numpy as np
import pytest
import torch
from lightning_utilities import apply_to_collection
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from torch import Tensor

from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.wrappers.bootstrapping import BootStrapper, _bootstrap_sampler
from unittests._helpers import seed_all

seed_all(42)

_preds = torch.randint(10, (10, 32))
_target = torch.randint(10, (10, 32))


class TestBootStrapper(BootStrapper):
    """Subclass of Bootstrapper class.

    For testing purpose, we subclass the bootstrapper class so we can get the exact permutation the class is creating.
    This is necessary such that the reference we are comparing to returns the exact same result for a given permutation.

    """

    def update(self, *args: Any) -> None:
        """Update input where the permutation is also saved."""
        self.out = []
        for idx in range(self.num_bootstraps):
            size = len(args[0])
            sample_idx = _bootstrap_sampler(size, sampling_strategy=self.sampling_strategy).to(self.device)
            new_args = apply_to_collection(args, Tensor, torch.index_select, dim=0, index=sample_idx)
            self.metrics[idx].update(*new_args)
            self.out.append(new_args)


def _sample_checker(old_samples, new_samples, op: operator, threshold: int):
    found_one = False
    for os in old_samples:
        cond = op(os, new_samples)
        if cond.sum() > threshold:
            found_one = True
            break
    return found_one


@pytest.mark.parametrize("sampling_strategy", ["poisson", "multinomial"])
def test_bootstrap_sampler(sampling_strategy):
    """Make sure that the bootstrap sampler works as intended."""
    old_samples = torch.randn(20, 2)

    # make sure that the new samples are only made up of old samples
    idx = _bootstrap_sampler(20, sampling_strategy=sampling_strategy)
    new_samples = old_samples[idx]
    for ns in new_samples:
        assert ns in old_samples

    found_one = _sample_checker(old_samples, new_samples, operator.eq, 2)
    assert found_one, "resampling did not work because no samples were sampled twice"

    found_zero = _sample_checker(old_samples, new_samples, operator.ne, 0)
    assert found_zero, "resampling did not work because all samples were at least sampled once"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sampling_strategy", ["poisson", "multinomial"])
@pytest.mark.parametrize(
    ("metric", "ref_metric"),
    [
        (MulticlassPrecision(num_classes=10, average="micro"), partial(precision_score, average="micro")),
        (MulticlassRecall(num_classes=10, average="micro"), partial(recall_score, average="micro")),
        (MeanSquaredError(), mean_squared_error),
    ],
)
def test_bootstrap(device, sampling_strategy, metric, ref_metric):
    """Test that the different bootstraps gets updated as we expected and that the compute method works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Test with device='cuda' requires gpu")

    _kwargs = {"base_metric": metric, "mean": True, "std": True, "raw": True, "sampling_strategy": sampling_strategy}
    _kwargs.update({"quantile": torch.tensor([0.05, 0.95], device=device)})

    bootstrapper = TestBootStrapper(**_kwargs)
    bootstrapper.to(device)

    collected_preds = [[] for _ in range(10)]
    collected_target = [[] for _ in range(10)]
    for p, t in zip(_preds, _target):
        p, t = p.to(device), t.to(device)
        bootstrapper.update(p, t)

        for i, o in enumerate(bootstrapper.out):
            collected_preds[i].append(o[0])
            collected_target[i].append(o[1])

    collected_preds = [torch.cat(cp).cpu() for cp in collected_preds]
    collected_target = [torch.cat(ct).cpu() for ct in collected_target]

    sk_scores = [ref_metric(ct, cp) for ct, cp in zip(collected_target, collected_preds)]

    output = bootstrapper.compute()
    assert np.allclose(output["quantile"][0].cpu(), np.quantile(sk_scores, 0.05))
    assert np.allclose(output["quantile"][1].cpu(), np.quantile(sk_scores, 0.95))

    assert np.allclose(output["mean"].cpu(), np.mean(sk_scores))
    assert np.allclose(output["std"].cpu(), np.std(sk_scores, ddof=1))
    assert np.allclose(output["raw"].cpu(), sk_scores)

    # check that resetting works
    bootstrapper.reset()

    assert bootstrapper.update_count == 0
    assert all(m.update_count == 0 for m in bootstrapper.metrics)
    output = bootstrapper.compute()
    if not isinstance(metric, MeanSquaredError):
        assert output["mean"] == 0
        assert output["std"] == 0
        assert (output["raw"] == torch.zeros(10, device=device)).all()


@pytest.mark.parametrize("sampling_strategy", ["poisson", "multinomial"])
def test_low_sample_amount(sampling_strategy):
    """Test that the metric works with very little data.

    In this case it is very likely that no samples from a current batch should be included in one of the bootstraps,
    but this should still not crash the metric.
    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2048

    """
    preds = torch.randn(3, 3).softmax(dim=-1)
    target = torch.LongTensor([0, 0, 0])
    bootstrap_f1 = BootStrapper(
        MulticlassF1Score(num_classes=3, average=None), num_bootstraps=20, sampling_strategy=sampling_strategy
    )
    assert bootstrap_f1(preds, target)  # does not work


def test_args_and_kwargs_works():
    """Test that metric works with both args and kwargs and mix.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2450

    """
    x = torch.rand(100)
    y = x + torch.randn_like(x)
    ae = MeanAbsoluteError()
    assert ae(x, y) == ae(preds=x, target=y)

    bootstrapped_ae = BootStrapper(ae)
    res1 = bootstrapped_ae(x, y)
    res2 = bootstrapped_ae(x, target=y)
    res3 = bootstrapped_ae(preds=x, target=y)

    assert (res1["mean"].shape == res2["mean"].shape) & (res2["mean"].shape == res3["mean"].shape)
    assert (res1["std"].shape == res2["std"].shape) & (res2["mean"].shape == res3["std"].shape)
