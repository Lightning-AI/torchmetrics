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
import torch

from torchmetrics.functional.segmentation.generalized_dice import generalized_dice_score


def _channel(num_ones: int, length: int = 10) -> torch.Tensor:
    """Build a length-``length`` 0/1 channel with ``num_ones`` leading ones."""
    v = torch.zeros(length, dtype=torch.int64)
    v[:num_ones] = 1
    return v


def test_generalized_dice_absent_class_weight_is_per_class() -> None:
    """Absent-class inf weights must be replaced per class, independent of batch size."""
    # N=2 samples, C=3 classes (N != C, so a class-major/sample-major mix-up is observable).
    # Sample 0 has every class present, fixing the per-class max square weights:
    #   class 0 -> 1/1^2 = 1.00, class 1 -> 1/2^2 = 0.25, class 2 -> 1/2^2 = 0.25.
    # Sample 1 has class 0 absent from the target (square weight = inf) but predicted
    # (3 px), so the inf must be replaced by class 0's own max weight (1.00).
    target = torch.stack([
        torch.stack([_channel(1), _channel(2), _channel(2)]),  # sample 0
        torch.stack([_channel(0), _channel(5), _channel(2)]),  # sample 1: class 0 absent
    ])
    preds = torch.stack([
        torch.stack([_channel(1), _channel(2), _channel(2)]),  # sample 0: perfect
        torch.stack([_channel(3), _channel(5), _channel(2)]),  # sample 1: class 0 predicted, no overlap
    ])

    score = generalized_dice_score(preds, target, num_classes=3)

    # Sample 1, per_class=False (square weights):
    #   numerator   = 2*(0*w0 + 5*0.04 + 2*0.25) = 1.4
    #   denominator = (0+3)*w0 + (5+5)*0.04 + (2+2)*0.25 = 3*w0 + 1.4
    # correct w0 is class 0's max weight (1.00): 1.4 / 4.40 = 0.31818...
    # the `.T` reindexing hands class 1's max (0.25): 1.4 / 2.15 = 0.65116... (wrong)
    expected = torch.tensor([1.0, 1.4 / 4.4])
    assert torch.allclose(score, expected, atol=1e-4), score
