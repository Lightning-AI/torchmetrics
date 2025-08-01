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
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor

from torchmetrics.functional.multimodal.clip_score import (
    _detect_modality,
    _get_clip_model_and_processor,
    _process_image_data,
    _process_text_data,
    clip_score,
)
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10
from unittests._helpers import (
    _TORCH_LESS_THAN_2_1,
    _TRANSFORMERS_RANGE_LT_4_50_LE_4_53,
    seed_all,
    skip_on_connection_issues,
    skip_on_cuda_oom,
)
from unittests._helpers.testers import MetricTester

seed_all(42)


class _InputImagesCaptions(NamedTuple):
    images: Tensor
    captions: list[list[str]]


captions = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was found dead.",
    "The victim's brother said he cannot imagine anyone who would want to harm him",
    "A lawyer says him .\nMoschetto, 54 and prosecutors say .\nAuthority abc Moschetto.",
]

_random_input = _InputImagesCaptions(
    images=torch.randint(255, (2, 2, 3, 64, 64)), captions=[captions[0:2], captions[2:]]
)


def _reference_clip_score(preds, target, model_name_or_path):
    """Reference implementation for CLIP score.

    This uses the forward methods instead of the individual get_image_features and get_text_features methods.

    """
    model, processor = _get_clip_model_and_processor(model_name_or_path)
    inputs = processor(text=target, images=[p.cpu() for p in preds], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.diag().mean().detach()


def _custom_clip_processor_model():
    """Simulate the user providing a custom CLIP processor and model."""
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

    processor = _CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = _CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "openai/clip-vit-base-patch32",
        # "jinaai/jina-clip-v2",  # TODO: this can be enabled on CI with large GPU
        "zer0int/LongCLIP-L-Diffusers",
        _custom_clip_processor_model,
    ],
)
@pytest.mark.parametrize("inputs", [_random_input])
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_10, reason="test requires transformers>=4.10")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
@pytest.mark.skipif(
    _TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_LT_4_50_LE_4_53,
    # todo: if the transformers compatibility issue present in next feature release,
    #  consider bumping also torch min versions in the metrics implementations
    reason="could be due to torch compatibility issues with transformers",
)
class TestCLIPScore(MetricTester):
    """Test class for `CLIPScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @skip_on_connection_issues()
    def test_clip_score(self, inputs, model_name_or_path, ddp):
        """Test class implementation of metric."""
        # images are preds and targets are captions
        preds, target = inputs
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CLIPScore,
            reference_metric=partial(_reference_clip_score, model_name_or_path=model_name_or_path),
            metric_args={"model_name_or_path": model_name_or_path},
            check_scriptable=False,
            check_state_dict=False,
            check_batch=False,
            check_picklable=model_name_or_path != "jinaai/jina-clip-v2",
        )

    @skip_on_connection_issues()
    def test_clip_score_functional(self, inputs, model_name_or_path):
        """Test functional implementation of metric."""
        preds, target = inputs
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=clip_score,
            reference_metric=partial(_reference_clip_score, model_name_or_path=model_name_or_path),
            metric_args={"model_name_or_path": model_name_or_path},
        )

    @skip_on_connection_issues()
    def test_clip_score_differentiability(self, inputs, model_name_or_path):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=CLIPScore,
            metric_functional=clip_score,
            metric_args={"model_name_or_path": model_name_or_path},
        )

    @skip_on_connection_issues()
    def test_error_on_not_same_amount_of_input(self, inputs, model_name_or_path):
        """Test that an error is raised if the number of images and text examples does not match."""
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        with pytest.raises(ValueError, match="Expected the number of source and target examples to be the same.*"):
            metric(torch.randint(255, (2, 3, 64, 64)), "28-year-old chef found dead in San Francisco mall")

    @skip_on_connection_issues()
    def test_error_on_wrong_image_format(self, inputs, model_name_or_path):
        """Test that an error is raised if not all images are [c, h, w] format."""
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        with pytest.raises(
            ValueError, match="Expected all images to be 3d but found image that has either more or less"
        ):
            metric(torch.randint(255, (64, 64)), "28-year-old chef found dead in San Francisco mall")

    @skip_on_connection_issues()
    def test_plot_method(self, inputs, model_name_or_path):
        """Test the plot method of CLIPScore separately in this file due to the skipping conditions."""
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        preds, target = inputs
        metric.update(preds[0], target[0])
        fig, ax = metric.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @skip_on_connection_issues()
    def test_warning_on_long_caption(self, inputs, model_name_or_path):
        """Test that warning is given on long captions but metric still works."""
        if model_name_or_path != "openai/clip-vit-base-patch32":
            pytest.skip("This test is only relevant for the default model.")
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        preds, target = inputs
        target[0] = [target[0][0], "A 28-year-old chef who recently moved to San Francisco was found dead. " * 100]
        with pytest.warns(
            UserWarning,
            match="Encountered caption longer than max_position_embeddings=77. Will truncate captions to this.*",
        ):
            metric.update(preds[0], target[0])

    @skip_on_connection_issues()
    def test_clip_score_image_to_image(self, inputs, model_name_or_path):
        """Test CLIP score for image-to-image comparison."""
        expected = {
            "openai/clip-vit-base-patch32": 96.0,
            "jinaai/jina-clip-v2": 88.0,
            "zer0int/LongCLIP-L-Diffusers": 98.0,
            "_custom_clip_processor_model": 96.0,
        }
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        preds, _ = inputs
        score = metric(preds[0][0], preds[0][1])
        assert score.detach().round() == torch.tensor(
            expected[model_name_or_path if not callable(model_name_or_path) else "_custom_clip_processor_model"]
        )

    @skip_on_connection_issues()
    def test_clip_score_text_to_text(self, inputs, model_name_or_path):
        """Test CLIP score for text-to-text comparison."""
        expected = {
            "openai/clip-vit-base-patch32": 65.0,
            "jinaai/jina-clip-v2": 50.0,
            "zer0int/LongCLIP-L-Diffusers": 44.0,
            "_custom_clip_processor_model": 65.0,
        }
        metric = CLIPScore(model_name_or_path=model_name_or_path)
        _, target = inputs
        score = metric(target[0][0], target[0][1])
        assert score.detach().round() == torch.tensor(
            expected[model_name_or_path if not callable(model_name_or_path) else "_custom_clip_processor_model"]
        )

    @skip_on_connection_issues()
    def test_clip_score_functional_image_to_image(self, inputs, model_name_or_path):
        """Test functional implementation of image-to-image CLIP score."""
        expected = {
            "openai/clip-vit-base-patch32": 96.0,
            "jinaai/jina-clip-v2": 88.0,
            "zer0int/LongCLIP-L-Diffusers": 98.0,
            "_custom_clip_processor_model": 96.0,
        }
        preds, _ = inputs
        score = clip_score(preds[0][0], preds[0][1], model_name_or_path=model_name_or_path)
        assert score.detach().round() == torch.tensor(
            expected[model_name_or_path if not callable(model_name_or_path) else "_custom_clip_processor_model"]
        )

    @skip_on_cuda_oom()
    @skip_on_connection_issues()
    def test_clip_score_functional_text_to_text(self, inputs, model_name_or_path):
        """Test functional implementation of text-to-text CLIP score."""
        expected = {
            "openai/clip-vit-base-patch32": 65.0,
            "jinaai/jina-clip-v2": 49.0,
            "zer0int/LongCLIP-L-Diffusers": 44.0,
            "_custom_clip_processor_model": 65.0,
        }
        _, target = inputs
        score = clip_score(target[0][0], target[0][1], model_name_or_path=model_name_or_path)
        assert score.detach().round() == torch.tensor(
            expected[model_name_or_path if not callable(model_name_or_path) else "_custom_clip_processor_model"]
        )


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (torch.randn(3, 64, 64), "image"),
        ([torch.randn(3, 64, 64)], "image"),
        ("some text", "text"),
        (["text1", "text2"], "text"),
    ],
)
def test_detect_modality(input_data, expected):
    """Test that modality detection works correctly."""
    assert _detect_modality(input_data) == expected

    with pytest.raises(ValueError, match="Empty input list"):
        _detect_modality([])

    with pytest.raises(ValueError, match="Could not automatically determine modality"):
        _detect_modality(123)


@pytest.mark.parametrize(
    ("images", "expected_len", "should_raise"),
    [
        (torch.randn(3, 64, 64), 1, False),
        (torch.randn(2, 3, 64, 64), 2, False),
        ([torch.randn(3, 64, 64)], 1, False),
        ([torch.randn(3, 64, 64), torch.randn(3, 64, 64)], 2, False),
        (torch.randn(64, 64), 0, True),
        ([torch.randn(64, 64)], 0, True),
    ],
)
def test_process_image_data(images, expected_len, should_raise):
    """Test that image processing works correctly."""
    if should_raise:
        with pytest.raises(ValueError, match="Expected all images to be 3d"):
            _process_image_data(images)
    else:
        processed = _process_image_data(images)
        assert isinstance(processed, list)
        assert len(processed) == expected_len
        assert all(isinstance(img, Tensor) and img.ndim == 3 for img in processed)


@pytest.mark.parametrize(
    ("texts", "expected_len"),
    [
        ("single text", 1),
        (["text1", "text2"], 2),
        ([""], 1),
        ([], 0),
    ],
)
def test_process_text_data(texts, expected_len):
    """Test that text processing works correctly."""
    processed = _process_text_data(texts)
    assert isinstance(processed, list)
    assert len(processed) == expected_len
    assert all(isinstance(text, str) for text in processed)
