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
from typing import Any, List, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor
else:
    __doctest_skip__ = ["CLIPScore"]

from torchmetrics import Metric


class CLIPScore(Metric):
    """`CLIP Score`_ is a reference free metric that can be used to evaluate the correlation between an generated
    caption for an image and the actual content of the image. It has been found to be highly correlated with human
    judgement. The metric is defined as:

    .. math::
        \text{CLIPScore(I, C)} = max(100 * cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual CLIP embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`. The score is bound between 0 and 100 and the closer
    to 100 the better.

    .. note:: Metric is not scriptable

    Args:
        version: string indicating the version of the CLIP model to use. Available models are
            `"openai/clip-vit-base-patch16"`, `"openai/clip-vit-base-patch32"`, `"openai/clip-vit-large-patch14-336"`
            and `"openai/clip-vit-large-patch14"`,

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.multimodal import CLIPScore
        >>> metric = CLIPScore()
        >>> metric(torch.randint(255, (3, 224, 224)), "a photo of a cat")
        tensor(19.4135, grad_fn=<SqueezeBackward0>)
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    score: Tensor
    n_samples: Tensor

    def __init__(
        self,
        version: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = "openai/clip-vit-large-patch14",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if _TRANSFORMERS_AVAILABLE:
            self.model = _CLIPModel.from_pretrained(version)
            self.processor = _CLIPProcessor.from_pretrained(version)
        else:
            raise ModuleNotFoundError(
                "`CLIPScore` metric requires `transformers` package be installed."
                " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[multimodal]`."
            )
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, images: Union[Tensor, List[Tensor]], text: Union[str, List[str]]) -> None:
        """Updates CLIP score on a batch of images and text.

        Args:
            images: Either a single [N, C, H, W] tensor or an list of [C, H, W] tensors
            text: Either a single caption or a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match
        """
        if not isinstance(images, List):
            if images.ndim == 3:
                images = [images]
            else:  # unwrap into list
                images = [i for i in images]

        if not all(i.ndim == 3 for i in images):
            raise ValueError("Expected all images to be 3d but found image that has either more or less")

        if not isinstance(text, List):
            text = [text]

        if len(text) != len(images):
            raise ValueError(
                f"Expected the number of images and text examples to be the same but got {len(images)} and {len(text)}"
            )

        processed_input = self.processor(text=text, images=[i.cpu() for i in images], return_tensors="pt", padding=True)

        img_features = self.model.get_image_features(processed_input["pixel_values"].to(self.device))
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        txt_features = self.model.get_text_features(
            processed_input["input_ids"].to(self.device), processed_input["attention_mask"].to(self.device)
        )
        txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity between feature vectors
        score = (img_features * txt_features).sum(axis=-1)
        self.score += 100 * score.sum(0)
        self.n_samples += img_features.shape[0]

    def compute(self) -> Tensor:
        """Computes accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))
