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
import numpy as np
import pytest
import torch
from sklearn.metrics import precision_score, recall_score

from torchmetrics.classification import Precision, Recall
from torchmetrics.utilities import apply_to_collection
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_7
from torchmetrics.wrappers.bootstrapping import BootStrapper, _bootstrap_sampler

_preds = torch.randint(10, (10, 32))
_target = torch.randint(10, (10, 32))


class TestBootStrapper(BootStrapper):
    """ For testing purpose, we subclass the bootstrapper class so we can get the exact permutation
        the class is creating
    """

    def update(self, *args, **kwargs):
        self.out = []
        for idx in range(self.num_bootstraps):
            new_args = apply_to_collection(args, torch.Tensor, _bootstrap_sampler, generator=self.generator)
            new_kwargs = apply_to_collection(kwargs, torch.Tensor, _bootstrap_sampler, generator=self.generator)
            self.metrics[idx].update(*new_args, **new_kwargs)
            self.out.append(new_args)


@pytest.mark.parametrize("sampling_strategy", ['poisson', 'multinomial'])
def test_bootstrap_sampler(sampling_strategy):
    """ make sure that the bootstrap sampler works as intended """
    old_samples = torch.randn(5, 2)

    # make sure that the new samples are only made up of old samples
    new_samples = _bootstrap_sampler(old_samples, sampling_strategy=sampling_strategy)
    for ns in new_samples:
        assert ns in old_samples

    # make sure some samples are also sampled twice
    found_one = False
    for os in old_samples:
        cond = os == new_samples
        if cond.sum() > 2:
            found_one = True
            break
    
    assert found_one, "resampling did not work because no samples were sampled twice"
        
    # make sure some samples are never sampled
    found_zero = False
    for os in old_samples:
        cond = os != new_samples
        if cond.sum() > 0:
            found_zero = True
            break
    
    assert found_zero, "resampling did not work because all samples were atleast sampled once"
    

@pytest.mark.parametrize(
    "metric, sk_metric", [[Precision(average='micro'), precision_score], [Recall(average='micro'), recall_score]]
)
def test_bootstrap(metric, sk_metric):
    """ Test that the different bootstraps gets updated as we expected and that the compute method works """
    if _TORCH_GREATER_EQUAL_1_7:
        bootstrapper = TestBootStrapper(metric, mean=True, std=True, quantile=torch.tensor([0.05, 0.95]), raw=True)
    else:
        bootstrapper = TestBootStrapper(metric, mean=True, std=True, raw=True)

    collected_preds = [[] for _ in range(10)]
    collected_target = [[] for _ in range(10)]
    for p, t in zip(_preds, _target):
        bootstrapper.update(p, t)

        for i, o in enumerate(bootstrapper.out):

            collected_preds[i].append(o[0])
            collected_target[i].append(o[1])

    collected_preds = [torch.cat(cp) for cp in collected_preds]
    collected_target = [torch.cat(ct) for ct in collected_target]

    sk_scores = [sk_metric(ct, cp, average='micro') for ct, cp in zip(collected_target, collected_preds)]

    output = bootstrapper.compute()
    # quantile only avaible for pytorch v1.7 and forward
    if _TORCH_GREATER_EQUAL_1_7:
        pl_mean, pl_std, pl_quantile, pl_raw = output
        assert np.allclose(pl_quantile[0], np.quantile(sk_scores, 0.05))
        assert np.allclose(pl_quantile[1], np.quantile(sk_scores, 0.95))
    else:
        pl_mean, pl_std, pl_raw = output

    assert np.allclose(pl_mean, np.mean(sk_scores))
    assert np.allclose(pl_std, np.std(sk_scores, ddof=1))
    assert np.allclose(pl_raw, sk_scores)
