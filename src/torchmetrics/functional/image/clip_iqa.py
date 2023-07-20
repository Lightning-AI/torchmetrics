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
from typing import List, Literal, Tuple, Union

from torch import Tensor

from torchmetrics.functional.multimodal.clip_score import _get_model_and_processor
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

_PROMPTS = {
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


def _clip_iqa_format_promts(promts: Union[Literal, List[str], Tuple[str, str]]) -> List[str]:
    if isinstance(promts, tuple):
        return promts
    if isinstance(promts, str):
        return list(_PROMPTS[promts])
    if isinstance(promts, list):
        return [e for p in promts for e in _PROMPTS[p]]
    raise ValueError("promts must be a string, list of strings or tuple of strings")


def _clip_iqa_update(
    images: Union[Tensor, List[Tensor]],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> Tensor:
    if not isinstance(images, list):
        if images.ndim == 3:
            images = [images]
    else:  # unwrap into list
        images = list(images)

    if not all(i.ndim == 3 for i in images):
        raise ValueError("Expected all images to be 3d but found image that has either more or less")

    device = images[0].device
    text_processed = processor(text=["Good photo.", "Bad photo."], return_tensors="pt", padding=True)
    anchors = model.get_text_features(
        text_processed["input_ids"].to(device), text_processed["attention_mask"].to(device)
    )
    anchors = anchors / anchors.norm(p=2, dim=-1, keepdim=True)

    processed_input = processor(images=[i.cpu() for i in images], return_tensors="pt", padding=True)
    img_features = model.get_image_features(processed_input["pixel_values"].to(device))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    logits_per_image = 100 * img_features @ anchors.t()
    return logits_per_image.softmax(dim=-1)[:, 0]


def clip_iqa(
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
) -> Tensor:
    """Something should be here."""
    model, processor = _get_model_and_processor(model_name_or_path)
    device = images.device if isinstance(images, Tensor) else images[0].device
    return _clip_iqa_update(images, model.to(device), processor)
