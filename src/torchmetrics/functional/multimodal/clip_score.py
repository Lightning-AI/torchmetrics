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
from typing import List, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

    def _download_clip() -> None:
        _CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        _CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_clip):
        __doctest_skip__ = ["clip_score"]

else:
    __doctest_skip__ = ["clip_score"]
    _CLIPModel = None
    _CLIPProcessor = None


def _clip_score_update(
    images: Union[Tensor, List[Tensor]],
    text: Union[str, List[str]],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> Tuple[Tensor, int]:
    if not isinstance(images, list):
        if images.ndim == 3:
            images = [images]
    else:  # unwrap into list
        images = list(images)

    if not all(i.ndim == 3 for i in images):
        raise ValueError("Expected all images to be 3d but found image that has either more or less")

    if not isinstance(text, list):
        text = [text]

    if len(text) != len(images):
        raise ValueError(
            f"Expected the number of images and text examples to be the same but got {len(images)} and {len(text)}"
        )
    device = images[0].device
    processed_input = processor(text=text, images=[i.cpu() for i in images], return_tensors="pt", padding=True)

    img_features = model.get_image_features(processed_input["pixel_values"].to(device))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    txt_features = model.get_text_features(
        processed_input["input_ids"].to(device), processed_input["attention_mask"].to(device)
    )
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity between feature vectors
    score = 100 * (img_features * txt_features).sum(axis=-1)
    return score, len(text)


def _get_model_and_processor(
    model_name_or_path: Literal[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ] = "openai/clip-vit-large-patch14",
) -> Tuple[_CLIPModel, _CLIPProcessor]:
    if _TRANSFORMERS_AVAILABLE:
        model = _CLIPModel.from_pretrained(model_name_or_path)
        processor = _CLIPProcessor.from_pretrained(model_name_or_path)
        return model, processor

    raise ModuleNotFoundError(
        "`clip_score` metric requires `transformers` package be installed."
        " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[multimodal]`."
    )


def clip_score(
    images: Union[Tensor, List[Tensor]],
    text: Union[str, List[str]],
    model_name_or_path: Literal[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ] = "openai/clip-vit-large-patch14",
) -> Tensor:
    r"""Calculate `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP is a reference free metric that can be used to evaluate the correlation between a generated caption for an
    image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual CLIP embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    Args:
        images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
        text: Either a single caption or a list of captions
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are
            `"openai/clip-vit-base-patch16"`, `"openai/clip-vit-base-patch32"`, `"openai/clip-vit-large-patch14-336"`
            and `"openai/clip-vit-large-patch14"`,

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
        ValueError:
            If not all images have format [C, H, W]
        ValueError:
            If the number of images and captions do not match

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.functional.multimodal import clip_score
        >>> score = clip_score(torch.randint(255, (3, 224, 224)), "a photo of a cat", "openai/clip-vit-base-patch16")
        >>> print(score.detach())
        tensor(24.4255)
    """
    model, processor = _get_model_and_processor(model_name_or_path)
    device = images.device if isinstance(images, Tensor) else images[0].device
    score, _ = _clip_score_update(images, text, model.to(device), processor)
    score = score.mean(0)
    return torch.max(score, torch.zeros_like(score))
