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
import os
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import piq
import pytest
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import PILToTensor

from torchmetrics.functional.multimodal.clip_iqa import clip_image_quality_assessment
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
from torchmetrics.utilities.imports import _PIQ_GREATER_EQUAL_0_8, _TRANSFORMERS_GREATER_EQUAL_4_10
from unittests._helpers import skip_on_connection_issues
from unittests._helpers.testers import MetricTester
from unittests.image import _SAMPLE_IMAGE


@pytest.mark.parametrize(
    ("prompts", "match"),
    [
        ("quality", "Argument `prompts` must be a tuple containing strings or tuples of strings"),
        (("quality", 1), "Argument `prompts` must be a tuple containing strings or tuples of strings"),
        ((("quality", "quality", "quality"),), "If a tuple is provided in argument `prompts`, it must be of length 2"),
        (("quality", "something"), "All elements of `prompts` must be one of.*"),
    ],
)
def test_raises_error_on_wrong_prompts(prompts, match):
    """Test that the function raises an error if the prompts argument are not valid."""
    img = torch.rand(1, 3, 256, 256)

    with pytest.raises(ValueError, match=match):
        clip_image_quality_assessment(img, prompts=prompts)


class CLIPTesterClass(CLIPImageQualityAssessment):
    """Tester class for `CLIPImageQualityAssessment` metric overriding its update method."""

    def update(self, preds, target):
        """Override the update method to support two input arguments."""
        super().update(preds)

    def compute(self):
        """Override the compute method."""
        return super().compute().sum()


def _clip_iqa_wrapped(preds, target):
    """Tester function for `clip_image_quality_assessment` that supports two input arguments."""
    return clip_image_quality_assessment(preds)


def _reference_clip_iqa(preds, target, reduce=False):
    """Reference implementation of `CLIPImageQualityAssessment` metric."""
    res = piq.CLIPIQA()(preds).squeeze()
    return res.sum() if reduce else res


@pytest.mark.skipif(not _PIQ_GREATER_EQUAL_0_8, reason="metric requires piq>=0.8")
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_10, reason="test requires transformers>=4.10")
class TestCLIPIQA(MetricTester):
    """Test clip iqa metric."""

    @skip_on_connection_issues()
    @pytest.mark.parametrize("ddp", [False])
    def test_clip_iqa(self, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=torch.rand(2, 1, 3, 128, 128),
            target=torch.rand(2, 1, 3, 128, 128),
            metric_class=CLIPTesterClass,
            reference_metric=partial(_reference_clip_iqa, reduce=True),
            check_scriptable=False,
            check_state_dict=False,
        )

    @skip_on_connection_issues()
    @pytest.mark.parametrize("shapes", [(2, 1, 3, 256, 256), (2, 2, 3, 256, 256), (2, 2, 3, 128, 128)])
    def test_clip_iqa_functional(self, shapes):
        """Test functional implementation of metric."""
        img = torch.rand(shapes)
        self.run_functional_metric_test(
            preds=img,
            target=img,
            metric_functional=_clip_iqa_wrapped,
            reference_metric=_reference_clip_iqa,
        )


@skip_on_connection_issues()
@pytest.mark.skipif(not _PIQ_GREATER_EQUAL_0_8, reason="metric requires piq>=0.8")
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_10, reason="test requires transformers>=4.10")
@pytest.mark.skipif(not os.path.isfile(_SAMPLE_IMAGE), reason="test image not found")
def test_for_correctness_sample_images():
    """Compare the output of the function with the output of the reference implementation."""
    img = Image.open(_SAMPLE_IMAGE)
    img = PILToTensor()(img)
    img = img.float()[None]

    reference = piq.CLIPIQA(data_range=255)
    reference_score = reference(img)

    result = clip_image_quality_assessment(img, data_range=255)
    assert torch.allclose(reference_score, result)


@skip_on_connection_issues()
@pytest.mark.skipif(not _PIQ_GREATER_EQUAL_0_8, reason="metric requires piq>=0.8")
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_10, reason="test requires transformers>=4.10")
@pytest.mark.parametrize(
    "model",
    [
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ],
)
@pytest.mark.skipif(not os.path.isfile(_SAMPLE_IMAGE), reason="test image not found")
def test_other_models(model):
    """Test that the function works with other models."""
    img = Image.open(_SAMPLE_IMAGE)
    img = PILToTensor()(img)
    img = img.float()[None]

    reference = piq.CLIPIQA(data_range=255)
    reference_score = reference(img)

    result = clip_image_quality_assessment(img, data_range=255, model_name_or_path=model)
    # allow large difference between scores due to different models, but still in the same ballpark
    assert reference_score - 0.2 < result < reference_score + 0.2


@skip_on_connection_issues()
@pytest.mark.skipif(not _PIQ_GREATER_EQUAL_0_8, reason="metric requires piq>=0.8")
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_10, reason="test requires transformers>=4.10")
@pytest.mark.parametrize(
    "prompts",
    [
        ("quality",),
        ("brightness",),
        ("noisiness",),
        ("colorfullness",),
        ("sharpness",),
        ("contrast",),
        ("complexity",),
        ("natural",),
        ("happy",),
        ("scary",),
        ("new",),
        ("warm",),
        ("real",),
        ("beautiful",),
        ("lonely",),
        ("relaxing",),
        # some random combinations
        ("quality", "brightness"),
        ("quality", "brightness", "noisiness"),
        ("quality", "brightness", "noisiness", "colorfullness"),
        # custom prompts
        (("Photo of a cat", "Photo of a dog"),),
        (("Photo of a cat", "Photo of a dog"), "quality"),
        (("Photo of a cat", "Photo of a dog"), "quality", ("Colorful photo", "Black and white photo")),
    ],
)
@pytest.mark.skipif(not os.path.isfile(_SAMPLE_IMAGE), reason="test image not found")
def test_prompt(prompts):
    """Test that the function works with other prompts, and that output is as expected."""
    img = Image.open(_SAMPLE_IMAGE)
    img = PILToTensor()(img)
    img = img.float()[None]

    result = clip_image_quality_assessment(img, data_range=255, prompts=prompts)
    if len(prompts) == 1:
        assert isinstance(result, Tensor)
        assert 0 < result < 1
    else:
        assert isinstance(result, dict)
        for i, (k, v) in enumerate(result.items()):
            assert isinstance(k, str)
            assert k == prompts[i] if isinstance(prompts[i], str) else "user_defined_" in k
            assert isinstance(v, Tensor)
            assert 0 < v < 1


@skip_on_connection_issues()
@pytest.mark.skipif(not _PIQ_GREATER_EQUAL_0_8, reason="metric requires piq>=0.8")
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_10, reason="test requires transformers>=4.10")
def test_plot_method():
    """Test the plot method of CLIPScore separately in this file due to the skipping conditions."""
    metric = CLIPImageQualityAssessment()
    metric.update(torch.rand(1, 3, 256, 256))
    fig, ax = metric.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
