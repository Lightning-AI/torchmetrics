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

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_peak_signal_noise_ratio
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

_input_size = (NUM_BATCHES, BATCH_SIZE, 32, 32)
_inputs = [
    Input(
        preds=torch.randint(n_cls_pred, _input_size, dtype=torch.float),
        target=torch.randint(n_cls_target, _input_size, dtype=torch.float),
    )
    for n_cls_pred, n_cls_target in [(10, 10), (5, 10), (10, 5)]
]


def _to_sk_peak_signal_noise_ratio_inputs(value, dim):
    value = value.numpy()
    batches = value[None] if value.ndim == len(_input_size) - 1 else value

    if dim is None:
        return [batches]

    num_dims = np.size(dim)
    if not num_dims:
        return batches

    inputs = []
    for batch in batches:
        batch = np.moveaxis(batch, dim, np.arange(-num_dims, 0))
        psnr_input_shape = batch.shape[-num_dims:]
        inputs.extend(batch.reshape(-1, *psnr_input_shape))
    return inputs


def _skimage_psnr(preds, target, data_range, reduction, dim):
    if isinstance(data_range, tuple):
        preds = preds.clamp(min=data_range[0], max=data_range[1])
        target = target.clamp(min=data_range[0], max=data_range[1])
        data_range = data_range[1] - data_range[0]
    sk_preds_lists = _to_sk_peak_signal_noise_ratio_inputs(preds, dim=dim)
    sk_target_lists = _to_sk_peak_signal_noise_ratio_inputs(target, dim=dim)
    np_reduce_map = {"elementwise_mean": np.mean, "none": np.array, "sum": np.sum}
    return np_reduce_map[reduction](
        [
            skimage_peak_signal_noise_ratio(sk_target, sk_preds, data_range=data_range)
            for sk_target, sk_preds in zip(sk_target_lists, sk_preds_lists)
        ]
    )


def _base_e_sk_psnr(preds, target, data_range, reduction, dim):
    return _skimage_psnr(preds, target, data_range, reduction, dim) * np.log(10)


@pytest.mark.parametrize(
    "preds, target, data_range, reduction, dim",
    [
        (_inputs[0].preds, _inputs[0].target, 10, "elementwise_mean", None),
        (_inputs[1].preds, _inputs[1].target, 10, "elementwise_mean", None),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", None),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", 1),
        (_inputs[2].preds, _inputs[2].target, 5, "elementwise_mean", (1, 2)),
        (_inputs[2].preds, _inputs[2].target, 5, "sum", (1, 2)),
        (_inputs[0].preds, _inputs[0].target, (0.0, 1.0), "elementwise_mean", None),
    ],
)
@pytest.mark.parametrize(
    "base, ref_metric",
    [
        (10.0, _skimage_psnr),
        (2.718281828459045, _base_e_sk_psnr),
    ],
)
class TestPSNR(MetricTester):
    """Test class for `PeakSignalNoiseRatio` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
    def test_psnr(self, preds, target, data_range, base, reduction, dim, ref_metric, ddp):
        """Test class implementation of metric."""
        _args = {"data_range": data_range, "base": base, "reduction": reduction, "dim": dim}
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            PeakSignalNoiseRatio,
            partial(ref_metric, data_range=data_range, reduction=reduction, dim=dim),
            metric_args=_args,
        )

    def test_psnr_functional(self, preds, target, ref_metric, data_range, base, reduction, dim):
        """Test functional implementation of metric."""
        _args = {"data_range": data_range, "base": base, "reduction": reduction, "dim": dim}
        self.run_functional_metric_test(
            preds,
            target,
            peak_signal_noise_ratio,
            partial(ref_metric, data_range=data_range, reduction=reduction, dim=dim),
            metric_args=_args,
        )

    # PSNR half + cpu does not work due to missing support in torch.log
    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_2_1,
        reason="Pytoch below 2.1 does not support cpu + half precision used in PSNR metric",
    )
    def test_psnr_half_cpu(self, preds, target, data_range, reduction, dim, base, ref_metric):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            PeakSignalNoiseRatio,
            peak_signal_noise_ratio,
            {"data_range": data_range, "base": base, "reduction": reduction, "dim": dim},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_psnr_half_gpu(self, preds, target, data_range, reduction, dim, base, ref_metric):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            PeakSignalNoiseRatio,
            peak_signal_noise_ratio,
            {"data_range": data_range, "base": base, "reduction": reduction, "dim": dim},
        )


@pytest.mark.parametrize("reduction", ["none", "sum"])
def test_reduction_for_dim_none(reduction):
    """Test that warnings are raised when then reduction parameter is combined with no dim provided arg."""
    match = f"The `reduction={reduction}` will not have any effect when `dim` is None."
    with pytest.warns(UserWarning, match=match):
        PeakSignalNoiseRatio(reduction=reduction, dim=None)

    with pytest.warns(UserWarning, match=match):
        peak_signal_noise_ratio(_inputs[0].preds, _inputs[0].target, reduction=reduction, dim=None)


def test_missing_data_range():
    """Check that error is raised if data range is not provided."""
    with pytest.raises(ValueError, match="The `data_range` must be given when `dim` is not None."):
        PeakSignalNoiseRatio(data_range=None, dim=0)

    with pytest.raises(ValueError, match="The `data_range` must be given when `dim` is not None."):
        peak_signal_noise_ratio(_inputs[0].preds, _inputs[0].target, data_range=None, dim=0)
