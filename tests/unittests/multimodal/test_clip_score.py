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
import pytest
import torch
from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor

from torchmetrics.multimodal.clip_score import CLIPScore
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

from collections import namedtuple


Input = namedtuple("Input", ["images", "captions"])


captions = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was "
    "found dead in the staircase of a local shopping center.",
    "The victim's brother said he cannot imagine anyone who would want to harm him,\"Finally, it went uphill again at "
    'him."',
]

_random_input = Input(
    images=torch.randint(255, (2, 2, 3, 224, 224)),
    captions=[captions[0:2], captions[2:]]
)


def _compare_fn(preds, target):
    processor = _CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = _CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=target, images=[p.cpu() for p in preds], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    print(logits_per_image)
    return logits_per_image.diag().mean().detach()


@pytest.mark.parametrize("input", [_random_input,])
class TestCLIPScore(MetricTester):
    atol = 1e-5

    @pytest.mark.parametrize("ddp", [True, False])
    def test_clip_score(self, input, ddp):
        # images are preds and targets are captions
        preds, target = input
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CLIPScore,
            sk_metric=_compare_fn,
            check_scriptable=False,
        )


def test_error_on_not_same_amount_of_input():
    """Test that an error is raised if the number of images and text examples does not match."""
    metric = CLIPScore()
    with pytest.raises(ValueError):
        metric(torch.randint(255, (2, 3, 224, 224)), "28-year-old chef found dead in San Francisco mall")


def test_error_on_wrong_image_format():
    """Test that an error is raised if not all images are [c, h, w] format."""
    metric = CLIPScore()
    with pytest.raises(ValueError):
        metric(torch.randint(255, (224, 224)), "28-year-old chef found dead in San Francisco mall")
