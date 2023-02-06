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

import pytest
import torch
from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor

from torchmetrics.functional.multimodal.clip_score import clip_score
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester
from unittests.text.helpers import skip_on_connection_issues

seed_all(42)


Input = namedtuple("Input", ["images", "captions"])


captions = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was "
    "found dead in the staircase of a local shopping center.",
    "The victim's brother said he cannot imagine anyone who would want to harm him,\"Finally, it went uphill again at "
    'him."',
    "A lawyer says him .\nMoschetto, 54 and prosecutors say .\nAuthority abc Moschetto  .",
]

_random_input = Input(images=torch.randint(255, (2, 2, 3, 224, 224)), captions=[captions[0:2], captions[2:]])


def _compare_fn(preds, target, model_name_or_path):
    processor = _CLIPProcessor.from_pretrained(model_name_or_path)
    model = _CLIPModel.from_pretrained(model_name_or_path)
    inputs = processor(text=target, images=[p.cpu() for p in preds], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.diag().mean().detach()


@pytest.mark.parametrize("model_name_or_path", ["openai/clip-vit-base-patch32"])
@pytest.mark.parametrize("input", [_random_input])
@pytest.mark.skipif(not _TRANSFORMERS_AVAILABLE, reason="test requires bert_score")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
class TestCLIPScore(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    @skip_on_connection_issues()
    def test_clip_score(self, input, model_name_or_path, ddp, dist_sync_on_step):
        # images are preds and targets are captions
        preds, target = input
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CLIPScore,
            reference_metric=partial(_compare_fn, model_name_or_path=model_name_or_path),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"model_name_or_path": model_name_or_path},
            check_scriptable=False,
            check_state_dict=False,
        )

    @skip_on_connection_issues()
    def test_clip_score_functional(self, input, model_name_or_path):
        preds, target = input
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=clip_score,
            reference_metric=partial(_compare_fn, model_name_or_path=model_name_or_path),
            metric_args={"model_name_or_path": model_name_or_path},
        )

    @skip_on_connection_issues()
    def test_clip_score_differentiability(self, input, model_name_or_path):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=CLIPScore,
            metric_functional=clip_score,
            metric_args={"model_name_or_path": model_name_or_path},
        )

    @skip_on_connection_issues()
    def test_error_on_not_same_amount_of_input(self, input, model_name_or_path):
        """Test that an error is raised if the number of images and text examples does not match."""
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        with pytest.raises(ValueError):  # noqa: PT011  # todo
            metric(torch.randint(255, (2, 3, 224, 224)), "28-year-old chef found dead in San Francisco mall")

    @skip_on_connection_issues()
    def test_error_on_wrong_image_format(self, input, model_name_or_path):
        """Test that an error is raised if not all images are [c, h, w] format."""
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        with pytest.raises(ValueError):  # noqa: PT011  # todo
            metric(torch.randint(255, (224, 224)), "28-year-old chef found dead in San Francisco mall")
