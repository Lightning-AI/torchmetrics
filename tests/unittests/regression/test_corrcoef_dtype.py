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
"""The rank correlation metrics must not silently change the dtype of their input.

`spearman_corrcoef` and `kendall_rank_corrcoef` build their statistics out of integer intermediates -- ranks, and
concordant/discordant pair counts. Dividing two integer tensors in PyTorch yields the *global default* dtype rather
than anything derived from the input, so a ``float64`` input used to come back as ``float32``. See
https://github.com/Lightning-AI/torchmetrics/issues/3465.

"""

import pytest
import torch

from torchmetrics.functional import kendall_rank_corrcoef, pearson_corrcoef, spearman_corrcoef
from torchmetrics.functional.regression.spearman import _rank_data
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1

NUM_SAMPLES = 100
KENDALL_VARIANTS = ["a", "b", "c"]

# torch below 2.1 cannot run the half-precision path on CPU at all -- `torch.arange` lacks the support that
# `_rank_data` relies on -- which is why the existing `test_spearman_corrcoef_half_cpu` carries the same guard
_half_on_cpu = pytest.mark.skipif(
    not _TORCH_GREATER_EQUAL_2_1, reason="torch below 2.1 does not support cpu + half precision in these metrics"
)

# float16/bfloat16 are promoted rather than preserved: ranks are integers and need more mantissa than they can hold
PROMOTED = [
    pytest.param(torch.float16, torch.float32, id="float16->float32", marks=_half_on_cpu),
    pytest.param(torch.bfloat16, torch.float32, id="bfloat16->float32", marks=_half_on_cpu),
    pytest.param(torch.float32, torch.float32, id="float32"),
    pytest.param(torch.float64, torch.float64, id="float64"),
]


def _inputs(dtype: torch.dtype, num_outputs: int = 1):
    """Reproducible inputs cast to ``dtype``."""
    generator = torch.Generator().manual_seed(42)
    shape = (NUM_SAMPLES,) if num_outputs == 1 else (NUM_SAMPLES, num_outputs)
    preds = torch.rand(shape, generator=generator, dtype=torch.float64)
    target = torch.rand(shape, generator=generator, dtype=torch.float64)
    return preds.to(dtype), target.to(dtype)


@pytest.mark.parametrize(("in_dtype", "out_dtype"), PROMOTED)
def test_spearman_corrcoef_dtype(in_dtype, out_dtype):
    """Spearman must return the promoted input dtype, not the global default."""
    preds, target = _inputs(in_dtype)
    assert spearman_corrcoef(preds, target).dtype == out_dtype


@pytest.mark.parametrize(("in_dtype", "out_dtype"), PROMOTED)
@pytest.mark.parametrize("variant", KENDALL_VARIANTS)
def test_kendall_rank_corrcoef_dtype(in_dtype, out_dtype, variant):
    """Every Kendall variant must agree on the output dtype.

    Before the fix, ``"c"`` preserved ``float64`` while ``"a"`` and ``"b"`` did not.

    """
    preds, target = _inputs(in_dtype)
    assert kendall_rank_corrcoef(preds, target, variant=variant).dtype == out_dtype


@pytest.mark.parametrize(("in_dtype", "out_dtype"), PROMOTED)
def test_rank_data_dtype(in_dtype, out_dtype):
    """``_rank_data`` is the shared root cause for Spearman -- pin it directly."""
    data, _ = _inputs(in_dtype)
    assert _rank_data(data).dtype == out_dtype


@pytest.mark.parametrize(("in_dtype", "out_dtype"), PROMOTED)
def test_spearman_multioutput_dtype(in_dtype, out_dtype):
    """The multioutput path ranks each column separately and must behave the same."""
    preds, target = _inputs(in_dtype, num_outputs=3)
    assert spearman_corrcoef(preds, target).dtype == out_dtype


def test_spearman_matches_float32_baseline():
    """Widening the accumulation must not move the value for inputs float32 already handled."""
    preds, target = _inputs(torch.float32)
    baseline = spearman_corrcoef(preds, target)
    widened = spearman_corrcoef(preds.double(), target.double())
    assert torch.allclose(widened.float(), baseline, atol=1e-6)


@pytest.mark.parametrize("variant", KENDALL_VARIANTS)
def test_kendall_matches_float32_baseline(variant):
    """Same for Kendall, across every variant."""
    preds, target = _inputs(torch.float32)
    baseline = kendall_rank_corrcoef(preds, target, variant=variant)
    widened = kendall_rank_corrcoef(preds.double(), target.double(), variant=variant)
    assert torch.allclose(widened.float(), baseline, atol=1e-6)


def test_rank_correlations_agree_with_pearson_on_dtype():
    """The three correlation metrics should not disagree about what dtype a float64 input deserves."""
    preds, target = _inputs(torch.float64)
    dtypes = {
        "pearson": pearson_corrcoef(preds, target).dtype,
        "spearman": spearman_corrcoef(preds, target).dtype,
        "kendall": kendall_rank_corrcoef(preds, target).dtype,
    }
    assert set(dtypes.values()) == {torch.float64}, dtypes


def test_spearman_ties_dtype():
    """The tie-averaging branch divides summed ranks by their counts -- the exact spot that used to downcast."""
    preds = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float64)
    assert _rank_data(preds).dtype == torch.float64
    assert spearman_corrcoef(preds, target).dtype == torch.float64
