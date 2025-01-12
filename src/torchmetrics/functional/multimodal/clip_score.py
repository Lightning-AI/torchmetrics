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
from typing import TYPE_CHECKING, List, Union, cast

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
        ValueError: If the input_data is an empty list or modality cannot be determined

    """
    if isinstance(input_data, Tensor):
        return "image"

    if isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Empty input list")
        if isinstance(input_data[0], Tensor):
            return "image"
        if isinstance(input_data[0], str):
            return "text"

    if isinstance(input_data, str):
        return "text"

    raise ValueError("Could not automatically determine modality for input_data")


def _process_image_data(images: Union[Tensor, List[Tensor]]) -> List[Tensor]:
    """Helper function to process image data."""
    images = [images] if not isinstance(images, list) and images.ndim == 3 else list(images)
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
    modality: str,
    device: torch.device,
    model: "_CLIPModel",
    processor: "_CLIPProcessor",
) -> Tensor:
    """Get features from the CLIP model for either images or text.

    Args:
       data: List of input data (images or text)
       modality: String indicating the type of input data (must be either "image" or "text")
       device: Device to run the model on
       model: CLIP model instance
       processor: CLIP processor instance

    Returns:
       Tensor of features from the CLIP model

    Raises:
        ValueError: If modality is not "image" or "text"

    """
    if modality == "image":
        # Add type checking for images
        image_data = [i for i in data if isinstance(i, Tensor)]
        processed = processor(images=[i.cpu() for i in image_data], return_tensors="pt", padding=True)
        return model.get_image_features(processed["pixel_values"].to(device))
    if modality == "text":
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
        return model.get_text_features(processed["input_ids"].to(device), processed["attention_mask"].to(device))
    raise ValueError(f"invalid modality {modality}")


def _clip_score_update(
    source: Union[Tensor, List[Tensor], List[str], str],
    target: Union[Tensor, List[Tensor], List[str], str],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> tuple[Tensor, int]:
    source_modality = _detect_modality(source)
    target_modality = _detect_modality(target)

    source_data = (
        _process_image_data(cast(Union[Tensor, List[Tensor]], source))
        if source_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], source))
    )
    target_data = (
        _process_image_data(cast(Union[Tensor, List[Tensor]], target))
        if target_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], target))
    )

    if len(source_data) != len(target_data):
        raise ValueError(
            "Expected the number of source and target examples to be the same but got "
            f"{len(source_data)} and {len(target_data)}"
        )

    device = (
        source_data[0].device
        if source_modality == "image" and isinstance(source_data[0], Tensor)
        else target_data[0].device
        if target_modality == "image" and isinstance(target_data[0], Tensor)
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)

    source_features = _get_features(
        cast(List[Union[Tensor, str]], source_data), source_modality, device, model, processor
    )
    target_features = _get_features(
        cast(List[Union[Tensor, str]], target_data), target_modality, device, model, processor
    )
    source_features = source_features / source_features.norm(p=2, dim=-1, keepdim=True)
    target_features = target_features / target_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    score = 100 * (source_features * target_features).sum(axis=-1)
    score = score.cpu() if source_modality == "text" and target_modality == "text" else score
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
    r"""Calculates `CLIP Score`_ which is a text-to-image similarity metric.

    CLIP Score is a reference free metric that can be used to evaluate the correlation between a generated caption for
    an image and the actual content of the image, as well as the similarity between texts or images. It has been found
    to be highly correlated with human judgement. The metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual `CLIP`_ embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    Additionally, the CLIP Score can be calculated for the same modalities:

    .. math::
        \text{CLIPScore(I_1, I_2)} = max(100 * cos(E_{I_1}, E_{I_2}), 0)

    where :math:`E_{I_1}` and :math:`E_{I_2}` are the visual embeddings for images :math:`I_1` and :math:`I_2`.

    .. math::
        \text{CLIPScore(T_1, T_2)} = max(100 * cos(E_{T_1}, E_{T_2}), 0)

    where :math:`E_{T_1}` and :math:`E_{T_2}` are the textual embeddings for texts :math:`T_1` and :math:`T_2`.

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
        >>> image = torch.randint(255, (3, 224, 224), generator=torch.Generator().manual_seed(42))
        >>> score = clip_score(image, "a photo of a cat", "openai/clip-vit-base-patch16")
        >>> score.detach()
        tensor(24.4255)

    Example:
        >>> from torchmetrics.functional.multimodal import clip_score
        >>> image1 = torch.randint(255, (3, 224, 224), generator=torch.Generator().manual_seed(42))
        >>> image2 = torch.randint(255, (3, 224, 224), generator=torch.Generator().manual_seed(43))
        >>> score = clip_score(image1, image2, "openai/clip-vit-base-patch16")
        >>> score.detach()
        tensor(99.4859)

    Example:
        >>> from torchmetrics.functional.multimodal import clip_score
        >>> score = clip_score(
        ...     "28-year-old chef found dead in San Francisco mall",
        ...     "A 28-year-old chef who recently moved to San Francisco was found dead.",
        ...     "openai/clip-vit-base-patch16"
        ... )
        >>> score.detach()
        tensor(91.3950)

    """
    model, processor = _get_clip_model_and_processor(model_name_or_path)
    score, _ = _clip_score_update(source, target, model, processor)
    score = score.mean(0)
    return torch.max(score, torch.zeros_like(score))
