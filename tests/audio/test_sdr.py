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
from collections import namedtuple
from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile
from torch import Tensor

from tests.audio import _SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB, _SAMPLE_NUMPY_ISSUE_895
from tests.helpers import seed_all
from tests.helpers.testers import MetricTester
from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.functional import signal_distortion_ratio
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

inputs_1spk = Input(
    preds=torch.rand(4, 1, 1, 1000),
    target=torch.rand(4, 1, 1, 1000),
)
inputs_2spk = Input(
    preds=torch.rand(4, 1, 2, 1000),
    target=torch.rand(4, 1, 2, 1000),
)


def sdr_original_batch(preds: Tensor, target: Tensor, compute_permutation: bool = False) -> Tensor:
    # shape: preds [BATCH_SIZE, spk, Time] , target [BATCH_SIZE, spk, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, spk, Time] , target [NUM_BATCHES*BATCH_SIZE, spk, Time]
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for b in range(preds.shape[0]):
        sdr_val_np, _, _, _ = bss_eval_sources(target[b], preds[b], compute_permutation)
        mss.append(sdr_val_np)
    return torch.tensor(mss)


def average_metric(preds: Tensor, target: Tensor, metric_func: Callable) -> Tensor:
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds, target).mean()


original_impl_compute_permutation = partial(sdr_original_batch)
# original_impl_no_compute_permutation = partial(sdr_original_batch)


@pytest.mark.parametrize(
    "preds, target, sk_metric",
    [
        (inputs_1spk.preds, inputs_1spk.target, original_impl_compute_permutation),
        # (inputs_1spk.preds, inputs_1spk.target, original_impl_no_compute_permutation, False),
        (inputs_2spk.preds, inputs_2spk.target, original_impl_compute_permutation),
        # (inputs_2spk.preds, inputs_2spk.target, original_impl_no_compute_permutation, False),
    ],
)
class TestSDR(MetricTester):
    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_sdr(self, preds, target, sk_metric, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SignalDistortionRatio,
            sk_metric=partial(average_metric, metric_func=sk_metric),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={},
        )

    def test_sdr_functional(self, preds, target, sk_metric):
        self.run_functional_metric_test(
            preds,
            target,
            signal_distortion_ratio,
            sk_metric,
            metric_args={},
        )

    def test_sdr_differentiability(self, preds, target, sk_metric):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_args={},
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_sdr_half_cpu(self, preds, target, sk_metric):
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_functional=signal_distortion_ratio,
            metric_args={},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_sdr_half_gpu(self, preds, target, sk_metric):
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_functional=signal_distortion_ratio,
            metric_args={},
        )


def test_error_on_different_shape(metric_class=SignalDistortionRatio):
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_on_real_audio():
    rate, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    rate, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    assert torch.allclose(
        signal_distortion_ratio(torch.from_numpy(deg), torch.from_numpy(ref)).float(),
        torch.tensor(0.2211),
        rtol=0.0001,
        atol=1e-4,
    )


def test_too_low_precision():
    """Corner case where the precision of the input is important."""
    data = np.load(_SAMPLE_NUMPY_ISSUE_895)
    preds = torch.tensor(data["preds"])
    target = torch.tensor(data["target"])

    sdr_tm = signal_distortion_ratio(preds, target).double()

    # check equality with bss_eval_sources in every pytorch version
    sdr_bss, _, _, _ = bss_eval_sources(target.numpy(), preds.numpy(), False)
    assert torch.allclose(
        sdr_tm.mean(),
        torch.tensor(sdr_bss).mean(),
        rtol=0.0001,
        atol=1e-2,
    )
