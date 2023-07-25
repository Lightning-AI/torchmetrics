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
import piq
import pytest
import torch
from torchmetrics.functional.image.clip_iqa import clip_image_quality_assessment
from unittests.image import _SAMPLE_IMAGE, _SAMPLE_IMAGE2
from PIL import Image
from torchvision.transforms import PILToTensor
from torch import Tensor

@pytest.mark.parametrize("prompts, match",
    [
        ("quality", "Argument `prompts` must be a list containing strings or tuples of strings"),
        (["quality", 1], "Argument `prompts` must be a list containing strings or tuples of strings"),
        ([("quality", "quality", "quality")], "If a tuple is provided in argument `prompts`, it must be of length 2"),
        (["quality", "something"], "All elements of `prompts` must be one of.*"),
    ]
)
def test_raises_error_on_wrong_prompts(prompts, match):
    """Test that the function raises an error if the prompts argument are not valid."""
    img = torch.rand(1, 3, 256, 256)

    with pytest.raises(ValueError, match=match):
        clip_image_quality_assessment(img, prompts=prompts)


@pytest.mark.parametrize("shapes", [(1, 3, 256, 256), (2, 3, 256, 256), (2, 3, 128, 128)])
def test_for_correctness_random_images(shapes):
    """Compare the output of the function with the output of the reference implementation."""
    img = torch.rand(shapes)

    reference = piq.CLIPIQA()
    reference_score = reference(img)

    result = clip_image_quality_assessment(img)
    assert torch.allclose(result, reference_score)

@pytest.mark.parametrize("path", [_SAMPLE_IMAGE])
def test_for_correctness_sample_images(path):
    img = Image.open(path)
    img = PILToTensor()(img)
    img = img.float()[None]

    reference = piq.CLIPIQA(data_range=255)
    reference_score = reference(img)

    result = clip_image_quality_assessment(img, data_range=255)
    assert torch.allclose(reference_score, result)

@pytest.mark.parametrize("model", [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14-336",
    "openai/clip-vit-large-patch14"
])
def test_other_models(model):
    """Test that the function works with other models and prompts."""
    img = Image.open(_SAMPLE_IMAGE)
    img = PILToTensor()(img)
    img = img.float()[None]

    reference = piq.CLIPIQA(data_range=255)
    reference_score = reference(img)

    result = clip_image_quality_assessment(img, data_range=255, model_name_or_path=model)
    assert reference_score - 0.2 < result < reference_score + 0.2

@pytest.mark.parametrize("prompts",
    [
        ["quality"],
        ["brightness"],
        ["noisiness"],
        ["colorfullness"],
        ["sharpness"],
        ["contrast"],
        ["complexity"],
        ["natural"],
        ["happy"],
        ["scary"],
        ["new"],
        ["warm"],
        ["real"],
        ["beutiful"],
        ["lonely"],
        ["relaxing"],
        # some random combinations
        ["quality", "brightness"],
        ["quality", "brightness", "noisiness"],
        ["quality", "brightness", "noisiness", "colorfullness"],
        # custom prompts
        [("Photo of a cat", "Photo of a dog")],
        [("Photo of a cat", "Photo of a dog"), "quality"],
        [("Photo of a cat", "Photo of a dog"), "quality", ("Colorful photo", "Black and white photo")],
    ]
)
def test_prompt(prompts):
    """Test that the function works with other models and prompts."""
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
            assert k == prompts[i] if isinstance(prompts[i], str) else "user_defined" in prompts[i]
            assert isinstance(v, Tensor)
            assert 0 < v < 1
