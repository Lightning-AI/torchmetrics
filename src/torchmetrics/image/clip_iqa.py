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
from typing import Any, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.image.clip_iqa import (
    _clip_iqa_format_prompts,
    _clip_iqa_update,
    _get_clip_iqa_anchor_vectors,
)
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PESQ_AVAILABLE, _TRANSFORMERS_AVAILABLE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CLIPScore.plot"]

_DEFAULT_MODEL: str = "openai/clip-vit-large-patch14"

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

    def _download_clip() -> None:
        _CLIPModel.from_pretrained(_DEFAULT_MODEL)
        _CLIPProcessor.from_pretrained(_DEFAULT_MODEL)

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_clip):
        __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
else:
    __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]


class CLIPIQA(Metric):
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

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``images`` (:class:`~torch.Tensor` or list of tensors): tensor with images feed to the feature extractor with. If
        a single tensor it should have shape ``(N, C, H, W)``. If a list of tensors, each tensor should have shape
        ``(C, H, W)``. ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.

    As output of `forward` and `compute` the metric returns the following output

    - ``clip_iqa`` (:class:`~torch.Tensor` or dict of tensors): tensor with the CLIP-IQA score. If a single prompt is
      provided, a single tensor with shape ``(N,)`` is returned. If a list of prompts is provided, a dict of tensors
      is returned with the prompt as key and the tensor with shape ``(N,)`` as value.

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        prompts: A string, list of strings or tuple of strings. If a string is provided, it must be one of the
            availble prompts. If a list of strings is provided, all strings must be one of the availble prompts.
            If a tuple of strings is provided, it must be of length 2 and the first string must be a positive prompt
            and the second string must be a negative prompt.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        Single promt:

        >>> from torchmetrics.functional.image import clip_iqa
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> imgs = torch.randint(255, (2, 3, 224, 224)).float()
        >>> clip_iqa(imgs, prompts="quality")
        >>> tensor([[0.5000], [0.5000]])

    Example::
        Multiple promts:

        >>> from torchmetrics.functional.image import clip_iqa
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> imgs = torch.randint(255, (2, 3, 224, 224)).float()
        >>> clip_iqa(imgs, prompts=["quality", "brightness"])
        >>> {'quality': tensor([[0.5000], [0.5000]]), 'brightness': tensor([[0.5000], [0.5000]])}

    """

    def __init__(
        self,
        model_name_or_path: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = _DEFAULT_MODEL,  # type: ignore[assignment]
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
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        prompts = _clip_iqa_format_prompts(prompts)
        self.model, self.processor = _get_clip_model_and_processor(model_name_or_path)
        anchors = _get_clip_iqa_anchor_vectors(prompts, self.model, self.processor, self.device)
        self.register_buffer("anchors", anchors)

        self.add_state("score_list", torch.tensor([], dtype=torch.float32), dist_reduce_fx=None)

    def update(self, images: Tensor) -> None:
        """Update metric state with new data."""
        probs = _clip_iqa_update(images, self.model, self.processor)
        self.score_list.append(probs)

    def compute(self) -> Tensor:
        """Compute metric."""
        return dim_zero_cat(self.score_list)
