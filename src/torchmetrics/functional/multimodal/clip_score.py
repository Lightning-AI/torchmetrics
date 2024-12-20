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
from typing import TYPE_CHECKING, List, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10

if TYPE_CHECKING and _TRANSFORMERS_GREATER_EQUAL_4_10:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_10:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

    def _download_clip_for_clip_score() -> None:
        _CLIPModel.from_pretrained("openai/clip-vit-large-patch14", resume_download=True)
        _CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", resume_download=True)

    if not _try_proceed_with_timeout(_download_clip_for_clip_score):
        __doctest_skip__ = ["clip_score"]
else:
    __doctest_skip__ = ["clip_score"]
    _CLIPModel = None
    _CLIPProcessor = None


def _detect_modality(input_data: Union[Tensor, List[Tensor], List[str], str]) -> Literal["image", "text"]:
    """Automatically detect the modality of the input data.

    Args:
        input_data: Input data that can be either image tensors or text strings

    Returns:
        str: Either "image" or "text"

    Raises:
        ValueError: If the modality cannot be determined

    """
    if isinstance(input_data, Tensor):
        if input_data.ndim == 3 or input_data.ndim == 4:  # Single image: [C, H, W]
            return "image"
    elif isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Empty input list")
        # Check first element
        if isinstance(input_data[0], Tensor):
            if input_data[0].ndim == 3:  # [C, H, W]
                return "image"
        elif isinstance(input_data[0], str):
            return "text"
    elif isinstance(input_data, str):
        return "text"

    raise ValueError("Could not automatically determine modality for input_data")


# def _process_data(
#     data: Union[Tensor, List[Tensor], List[str], str], modality: Literal["image", "text"]
# ) -> List[Union[Tensor, str]]:
#     """Helper function to process both source and target data."""
#     if modality == "image":
#         if not isinstance(data, list) and isinstance(data, Tensor) and data.ndim == 3:
#             data = [data]
#         elif isinstance(data, list):
#             data = list(data)
#         if not all(isinstance(i, Tensor) and i.ndim == 3 for i in data):
#             raise ValueError("Expected all images to be 3d but found image that has either more or less")
#     else:  # text
#         if not isinstance(data, list):
#             data = [data]
#     return data

def _process_image_data(images: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    """Helper function to process image data."""
    if isinstance(images, Tensor):
        if images.ndim == 3:
           return [images]

    if not all(i.ndim == 3 for i in images):
        raise ValueError("Expected all images to be 3d but found image that has either more or less")
    return images

def _process_text_data(texts: Union[str, List[str]]) -> List[str]:
    """Helper function to process text data."""
    if not isinstance(texts, list):
        texts = [texts]
    return texts

def _get_features(
    data: List[Union[Tensor, str]],
    modality: Literal["image", "text"],
    device: torch.device,
    model: "_CLIPModel",
    processor: "_CLIPProcessor",
) -> Tensor:
    """Get features from the CLIP model for either images or text.

    Args:
       data: List of input data (images or text)
       modality: Type of input data ("image" or "text")
       device: Device to run the model on
       model: CLIP model instance
       processor: CLIP processor instance
    Returns:
       Tensor of features from the CLIP model

    """
    if modality == "image":
        # Add type checking for images
        image_data = [i for i in data if isinstance(i, Tensor)]
        processed = processor(images=[i.cpu() for i in image_data], return_tensors="pt", padding=True)
        features = model.get_image_features(processed["pixel_values"].to(device))
    else:
        processed = processor(text=data, return_tensors="pt", padding=True)
        max_position_embeddings = model.config.text_config.max_position_embeddings
        if processed["attention_mask"].shape[-1] > max_position_embeddings:
            rank_zero_warn(
                f"Encountered caption longer than {max_position_embeddings=}. Will truncate captions to this length."
                "If longer captions are needed, initialize argument `model_name_or_path` with a model that supports"
                "longer sequences",
                UserWarning,
            )
            processed["attention_mask"] = processed["attention_mask"][..., :max_position_embeddings]
            processed["input_ids"] = processed["input_ids"][..., :max_position_embeddings]
        features = model.get_text_features(processed["input_ids"].to(device), processed["attention_mask"].to(device))

    return features


def _clip_score_update(
    source: Union[Tensor, List[Tensor], List[str], str],
    target: Union[Tensor, List[Tensor], List[str], str],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> tuple[Tensor, int]:
    source_modality = _detect_modality(source)
    target_modality = _detect_modality(target)

    processor_map = {
        "image": _process_image_data,
        "text": _process_text_data,
    }
    source_data = processor_map[source_modality](source)
    target_data = processor_map[target_modality](target)

    # Verify matching lengths
    if len(source_data) != len(target_data):
        raise ValueError(
            "Expected the number of source and target examples to be the same but got "
            f"{len(source_data)} and {len(target_data)}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if source_modality == "image" and isinstance(source_data[0], Tensor):
        device = source_data[0].device
    elif target_modality == "image" and isinstance(target_data[0], Tensor):
        device = target_data[0].device
    model = model.to(device)

    source_features = _get_features(source_data, source_modality, device, model, processor)
    target_features = _get_features(target_data, target_modality, device, model, processor)
    source_features = source_features / source_features.norm(p=2, dim=-1, keepdim=True)
    target_features = target_features / target_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    score = 100 * (source_features * target_features).sum(axis=-1)
    return score, len(source_data)


def _get_clip_model_and_processor(
    model_name_or_path: Literal[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ] = "openai/clip-vit-large-patch14",
) -> tuple[_CLIPModel, _CLIPProcessor]:
    if _TRANSFORMERS_GREATER_EQUAL_4_10:
        from transformers import CLIPModel as _CLIPModel
        from transformers import CLIPProcessor as _CLIPProcessor

        model = _CLIPModel.from_pretrained(model_name_or_path)
        processor = _CLIPProcessor.from_pretrained(model_name_or_path)
        return model, processor

    raise ModuleNotFoundError(
        "`clip_score` metric requires `transformers` package be installed."
        " Either install with `pip install transformers>=4.10.0` or `pip install torchmetrics[multimodal]`."
    )


def clip_score(
    source: Union[Tensor, List[Tensor], List[str], str],
    target: Union[Tensor, List[Tensor], List[str], str],
    model_name_or_path: Literal[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ] = "openai/clip-vit-large-patch14",
) -> Tensor:
    r"""Calculate `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image. It has been found to be highly correlated with human judgement. The
    metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    Args:
        source: Source input. This can be:
            - Images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors.
            - Text: Either a single caption or a list of captions.
        target: Target input. This can be:
            - Images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors.
            - Text: Either a single caption or a list of captions.
        model_name_or_path: String indicating the version of the CLIP model to use. Available models are:
            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`


    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
        ValueError:
            If not all images have format [C, H, W]
        ValueError:
            If the number of images and captions do not match

    Example:
        >>> from torchmetrics.functional.multimodal import clip_score
        >>> score = clip_score(torch.randint(255, (3, 224, 224)), "a photo of a cat", "openai/clip-vit-base-patch16")
        >>> score.detach()
        tensor(24.4255)

    """
    model, processor = _get_clip_model_and_processor(model_name_or_path)
    score, _ = _clip_score_update(source, target, model, processor)
    score = score.mean(0)
    return torch.max(score, torch.zeros_like(score))
