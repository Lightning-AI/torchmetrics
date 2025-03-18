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
from functools import partial

import numpy as np
import pytest
import torch
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile
from torch import Tensor

from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.functional import signal_distortion_ratio
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.audio import _SAMPLE_AUDIO_SPEECH, _SAMPLE_AUDIO_SPEECH_BAB_DB, _SAMPLE_NUMPY_ISSUE_895

seed_all(42)


inputs_1spk = _Input(
    preds=torch.rand(2, 1, 1, 500),
    target=torch.rand(2, 1, 1, 500),
)

inputs_2spk = _Input(
    preds=torch.rand(2, 1, 2, 500),
    target=torch.rand(2, 1, 2, 500),
)


def _reference_sdr_batch(
    preds: Tensor, target: Tensor, compute_permutation: bool = False, reduce_mean: bool = False
) -> Tensor:
    # shape: preds [BATCH_SIZE, spk, Time] , target [BATCH_SIZE, spk, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, spk, Time] , target [NUM_BATCHES*BATCH_SIZE, spk, Time]
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    mss = []
    for b in range(preds.shape[0]):
        sdr_val_np, _, _, _ = bss_eval_sources(target[b], preds[b], compute_permutation)
        mss.append(sdr_val_np)
    sdr = torch.tensor(np.array(mss))
    if reduce_mean:
        # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
        # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
        return sdr.mean()
    return sdr


@pytest.mark.parametrize(
    "preds, target",
    [(inputs_1spk.preds, inputs_1spk.target), (inputs_2spk.preds, inputs_2spk.target)],
)
class TestSDR(MetricTester):
    """Test class for `SignalDistortionRatio` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_sdr(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            SignalDistortionRatio,
            reference_metric=partial(_reference_sdr_batch, reduce_mean=True),
            metric_args={},
        )

    def test_sdr_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            signal_distortion_ratio,
            _reference_sdr_batch,
            metric_args={},
        )

    def test_sdr_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_args={},
        )

    def test_sdr_half_cpu(self, preds, target):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_functional=signal_distortion_ratio,
            metric_args={},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_sdr_half_gpu(self, preds, target):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=SignalDistortionRatio,
            metric_functional=signal_distortion_ratio,
            metric_args={},
        )


def test_error_on_different_shape(metric_class=SignalDistortionRatio):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_on_real_audio():
    """Test that metric works on real audio signal."""
    _, ref = wavfile.read(_SAMPLE_AUDIO_SPEECH)
    _, deg = wavfile.read(_SAMPLE_AUDIO_SPEECH_BAB_DB)
    sdr = signal_distortion_ratio(torch.from_numpy(deg), torch.from_numpy(ref))
    assert torch.allclose(sdr.float(), torch.tensor(0.2211), rtol=0.0001, atol=1e-4)


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
