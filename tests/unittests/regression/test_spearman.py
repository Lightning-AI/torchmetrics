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
from collections import namedtuple
from functools import partial

import pytest
import torch
from scipy.stats import rankdata, spearmanr
from torchmetrics.functional.regression.spearman import _rank_data, spearman_corrcoef
from torchmetrics.regression.spearman import SpearmanCorrCoef
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1

from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

_single_target_inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_single_target_inputs2 = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE),
    target=torch.randn(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs1 = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)

_multi_target_inputs2 = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    target=torch.randn(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)

_specific_input = Input(
    preds=torch.stack([torch.tensor([1.0, 0.0, 4.0, 1.0, 0.0, 3.0, 0.0]) for _ in range(NUM_BATCHES)]),
    target=torch.stack([torch.tensor([4.0, 0.0, 3.0, 3.0, 3.0, 1.0, 1.0]) for _ in range(NUM_BATCHES)]),
)


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
        (_specific_input.preds, _specific_input.target),
    ],
)
def test_ranking(preds, target):
    """Test that ranking function works as expected."""
    for p, t in zip(preds, target):
        scipy_ranking = [rankdata(p.numpy()), rankdata(t.numpy())]
        tm_ranking = [_rank_data(p), _rank_data(t)]
        assert (torch.tensor(scipy_ranking[0]) == tm_ranking[0]).all()
        assert (torch.tensor(scipy_ranking[1]) == tm_ranking[1]).all()


def _scipy_spearman(preds, target):
    if preds.ndim == 2:
        return [spearmanr(t.numpy(), p.numpy())[0] for t, p in zip(target.T, preds.T)]
    return spearmanr(target.numpy(), preds.numpy())[0]


@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_inputs1.preds, _single_target_inputs1.target),
        (_single_target_inputs2.preds, _single_target_inputs2.target),
        (_multi_target_inputs1.preds, _multi_target_inputs1.target),
        (_multi_target_inputs2.preds, _multi_target_inputs2.target),
        (_specific_input.preds, _specific_input.target),
    ],
)
class TestSpearmanCorrCoef(MetricTester):
    """Test class for `SpearmanCorrCoef` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    def test_spearman_corrcoef(self, preds, target, ddp):
        """Test class implementation of metric."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SpearmanCorrCoef,
            _scipy_spearman,
            metric_args={"num_outputs": num_outputs},
        )

    def test_spearman_corrcoef_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(preds, target, spearman_corrcoef, _scipy_spearman)

    def test_spearman_corrcoef_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=partial(SpearmanCorrCoef, num_outputs=num_outputs),
            metric_functional=spearman_corrcoef,
        )

    # Spearman half + cpu does not work due to missing support in torch.arange
    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_2_1,
        reason="Pytoch below 2.1 does not support cpu + half precision used in Spearman metric",
    )
    def test_spearman_corrcoef_half_cpu(self, preds, target):
        """Test dtype support of the metric on CPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_cpu(
            preds, target, partial(SpearmanCorrCoef, num_outputs=num_outputs), spearman_corrcoef
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_spearman_corrcoef_half_gpu(self, preds, target):
        """Test dtype support of the metric on GPU."""
        num_outputs = EXTRA_DIM if preds.ndim == 3 else 1
        self.run_precision_test_gpu(
            preds, target, partial(SpearmanCorrCoef, num_outputs=num_outputs), spearman_corrcoef
        )


def test_error_on_different_shape():
    """Test that error is raised when the preds and target shapes are not what is expected of the metric."""
    metric = SpearmanCorrCoef(num_outputs=1)
    with pytest.raises(TypeError, match="Expected `preds` and `target` both to be floating point tensors.*"):
        metric(torch.randint(5, (100,)), torch.randn(100))

    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))

    metric = SpearmanCorrCoef(num_outputs=5)
    with pytest.raises(ValueError, match="Expected both predictions and target to be either 1- or 2-dimensional.*"):
        metric(torch.randn(100, 2, 5), torch.randn(100, 2, 5))

    metric = SpearmanCorrCoef(num_outputs=2)
    with pytest.raises(ValueError, match="Expected argument `num_outputs` to match the second dimension of input.*"):
        metric(torch.randn(100, 5), torch.randn(100, 5))
