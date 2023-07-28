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
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

_PROMPTS: Dict[str, Tuple[str, str]] = {
    "quality": ("Good photo.", "Bad photo."),
    "brightness": ("Bright photo.", "Dark photo."),
    "noisiness": ("Clean photo.", "Noisy photo."),
    "colorfullness": ("Colorful photo.", "Dull photo."),
    "sharpness": ("Sharp photo.", "Blurry photo."),
    "contrast": ("High contrast photo.", "Low contrast photo."),
    "complexity": ("Complex photo.", "Simple photo."),
    "natural": ("Natural photo.", "Synthetic photo."),
    "happy": ("Happy photo.", "Sad photo."),
    "scary": ("Scary photo.", "Peaceful photo."),
    "new": ("New photo.", "Old photo."),
    "warm": ("Warm photo.", "Cold photo."),
    "real": ("Real photo.", "Abstract photo."),
    "beutiful": ("Beautiful photo.", "Ugly photo."),
    "lonely": ("Lonely photo.", "Sociable photo."),
    "relaxing": ("Relaxing photo.", "Stressful photo."),
}


def _clip_iqa_format_prompts(prompts: Union[str, List[str], Tuple[str, str]]) -> Tuple[List[str], Optional[List[str]]]:
    """Converts the provided keywords into a list of prompts for the model to calculate the achor vectors.

    Args:
        prompts: A string, list of strings or tuple of strings. If a string is provided, it must be one of the
            availble prompts. If a list of strings is provided, all strings must be one of the availble prompts.
            If a tuple of strings is provided, it must be of length 2 and the first string must be a positive prompt
            and the second string must be a negative prompt.

    Returns:
        A list of prompts and a list of prompt names (if a list of strings or tuple of strings was provided)

    Examples::

        >>> _clip_iqa_format_prompts("quality")
        (['Good photo.', 'Bad photo.'], None)
        >>> _clip_iqa_format_prompts(["quality", "brightness"])
        (['Good photo.', 'Bad photo.', 'Bright photo.', 'Dark photo.'], ['quality', 'brightness'])

    """
    if isinstance(prompts, tuple):
        if len(prompts) != 2:
            raise ValueError(
                f"Invalid prompt: {prompts}. Expected a tuple of two strings, one positive and one negative"
            )
        return prompts, None
    if isinstance(prompts, str):
        if prompts not in _PROMPTS:
            raise ValueError(f"Invalid prompt: {prompts}. Expected one of {_PROMPTS.keys()}")
        return list(_PROMPTS[prompts]), None
    if isinstance(prompts, list):
        if not all(p in _PROMPTS for p in prompts):
            raise ValueError(f"Invalid prompt: {prompts}. Expected all to be one of {_PROMPTS.keys()}")
        return [e for p in prompts for e in _PROMPTS[p]], prompts
    raise ValueError("promts must be a string, list of strings or tuple of strings")


def _get_clip_iqa_anchor_vectors(
    model: _CLIPModel, processor: _CLIPProcessor, prompts: List[str], device: torch.device
) -> Tensor:
    """Calculates the anchor vectors for the CLIP IQA metric.

    Args:
        model: The CLIP model
        processor: The CLIP processor
        prompts: A list of prompts
        device: The device to use for the calculation

    """
    text_processed = processor(text=prompts, return_tensors="pt", padding=True)
    anchors = model.get_text_features(
        text_processed["input_ids"].to(device), text_processed["attention_mask"].to(device)
    )
    return anchors / anchors.norm(p=2, dim=-1, keepdim=True)


def _clip_iqa_update(
    images: Union[Tensor, List[Tensor]],
    model: _CLIPModel,
    processor: _CLIPProcessor,
    prompts: List[str],
) -> Tensor:
    """Update function for CLIP IQA."""
    if not isinstance(images, list):
        if images.ndim == 3:
            images = [images]
    else:  # unwrap into list
        images = list(images)

    if not all(i.ndim == 3 for i in images):
        raise ValueError("Expected all images to be 3d but found image that has either more or less")

    device = images[0].device
    anchors = _get_clip_iqa_anchor_vectors(model, processor, prompts, device)

    processed_input = processor(images=[i.cpu() for i in images], return_tensors="pt", padding=True)
    img_features = model.get_image_features(processed_input["pixel_values"].to(device))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    logits_per_image = 100 * img_features @ anchors.t()
    return logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(-1)[:, :, 0]


def clip_image_quality_assessment(
    images: Union[Tensor, List[Tensor]],
    model_name_or_path: Literal[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "openai/clip-vit-large-patch14",
    ] = "openai/clip-vit-large-patch14",
    prompts: Union[
        Literal[
            "quality",
            "brightness",
            "noisiness",
            "colorfullness",
            "sharpness",
            "contrast",
            "complexity",
            "natural",
            "happy",
            "scary",
            "new",
            "warm",
            "real",
            "beutiful",
            "lonely",
            "relaxing",
        ],
        List[str],
        Tuple[str, str],
    ] = "quality",
) -> Union[Tensor, Dict[str, Tensor]]:
    """Calculates `CLIP-IQA`_, that can be used to measure the visual content of images.

    The metric is based on the `CLIP`_ model, which is a neural network trained on a variety of (image, text) pairs to
    be able to generate a vector representation of the image and the text that is similar if the image and text are
    semantically similar.

    The metric works by calculating the cosine similarity between user provided images and pre-defined promts. The
    promts always comes in pairs of "positive" and "negative" such as "Good photo." and "Bad photo.". By calculating
    the similartity between image embeddings and both the "positive" and "negative" prompt, the metric can determine
    which prompt the image is more similar to. The metric then returns the probability that the image is more similar
    to the first prompt than the second prompt.

    Build in promts are:
        * quality: "Good photo." vs "Bad photo."
        * brightness: "Bright photo." vs "Dark photo."
        * noisiness: "Clean photo." vs "Noisy photo."
        * colorfullness: "Colorful photo." vs "Dull photo."
        * sharpness: "Sharp photo." vs "Blurry photo."
        * contrast: "High contrast photo." vs "Low contrast photo."
        * complexity: "Complex photo." vs "Simple photo."
        * natural: "Natural photo." vs "Synthetic photo."
        * happy: "Happy photo." vs "Sad photo."
        * scary: "Scary photo." vs "Peaceful photo."
        * new: "New photo." vs "Old photo."
        * warm: "Warm photo." vs "Cold photo."
        * real: "Real photo." vs "Abstract photo."
        * beutiful: "Beautiful photo." vs "Ugly photo."
        * lonely: "Lonely photo." vs "Sociable photo."
        * relaxing: "Relaxing photo." vs "Stressful photo."


    Args:
        images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are
            `"openai/clip-vit-base-patch16"`, `"openai/clip-vit-base-patch32"`, `"openai/clip-vit-large-patch14-336"`
            and `"openai/clip-vit-large-patch14"`,
        prompts: A string, list of strings or tuple of strings. If a string is provided, it must be one of the
            availble prompts. If a list of strings is provided, all strings must be one of the availble prompts.
            If a tuple of strings is provided, it must be of length 2 and the first string must be a positive prompt
            and the second string must be a negative prompt.

    Returns:
        A tensor of shape [N, 1] if a single promts is provided. If a list of promts is provided, a dictionary of
        with the promts as keys and tensors of shape [N, 1] as values.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
        ValueError:
            If not all images have format [C, H, W]
        ValueError:
            If promts is a tuple and it is not of length 2
        ValueError:
            If promts is a string and it is not one of the available promts
        ValueError:
            If promts is a list of strings and not all strings are one of the available promts

    Example::
        Single promt:

        >>> from torchmetrics.functional.image import clip_image_quality_assessment
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> imgs = torch.randint(255, (2, 3, 224, 224)).float()
        >>> clip_image_quality_assessment(imgs, prompts="quality")
        >>> tensor([[0.5000], [0.5000]])

    Example::
        Multiple promts:

        >>> from torchmetrics.functional.image import clip_image_quality_assessment
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> imgs = torch.randint(255, (2, 3, 224, 224)).float()
        >>> clip_image_quality_assessment(imgs, prompts=["quality", "brightness"])
        >>> {'quality': tensor([[0.5000], [0.5000]]), 'brightness': tensor([[0.5000], [0.5000]])}

    """
    model, processor = _get_clip_model_and_processor(model_name_or_path)
    device = images.device if isinstance(images, Tensor) else images[0].device
    prompts, prompts_names = _clip_iqa_format_prompts(prompts)
    result = _clip_iqa_update(images, model.to(device), processor, prompts)
    if prompts_names is None:
        return result
    return {p: result[:, i] for i, p in enumerate(prompts_names)}
