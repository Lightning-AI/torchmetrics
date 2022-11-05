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
from typing import List, Union

import torch
from torch import Tensor

from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import CLIPFeatureExtractor as _CLIPFeatureExtractor
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPTokenizer as _CLIPTokenizer
else:
    _CLIPFeatureExtractor = None
    _CLIPModel = None
    _CLIPTokenizer = None

from torchmetrics import Metric


class CLIPScore(Metric):
    """`CLIP Score`_ is a reference free metric that can be used to evaluate the correlation between an generated
    caption for an image and the actual content of the image. It has been found to be highly correlated with human
    judgement. The metric is defined as.

    .. math::
        \text{CLIPScore(I, C)} = \\max(cos(E_I, E_C), 0)

    which corresponds to the cosine similarity between visual CLIP embedding :math:`E_i` for an image :math:`i` and
    textual CLIP embedding :math:`E_C` for an caption :math:`C`.

    Args:
        version: string indicating the version of the CLIP model to use. See `Huggingface OpenAI`_ for more info
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If `transformers` package is not installed

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.multimodal import CLIPScore
        >>> metric = CLIPScore()
        >>> img = torch.randint(255, (3, 224, 224))
        >>> text = "this is a random sentence"
        >>> metric(img, text)
        tensor([0.595])
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, version="openai/clip-vit-large-patch14", **kwargs) -> None:
        super().__init__(**kwargs)
        if _TRANSFORMERS_AVAILABLE:
            self.tokenizer = _CLIPTokenizer.from_pretrained(version)
            self.model = _CLIPModel.from_pretrained(version)
            self.features = _CLIPFeatureExtractor.from_pretrained(version)
        else:
            raise ModuleNotFoundError(
                "`CLIPScore` metric requires `transformers` package be installed."
                " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[multimodal]`."
            )
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, images: Union[Tensor, List[Tensor]], text: Union[str, List[str]]) -> None:
        """ Updates CLIP score with current batch of images and text

        Args:
            images: either a single tensor with shape `(N, C, H, W)` or an list of tensors each
                with shape `(C, H, W)`
            text: either a single string or a list of strings

        Raises:
            ValueError:
                If not all images have shape `(C, H, W)`
            ValueError:
                If the number of images and number of text samples are different
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

        img_features = [
            self.model.get_image_features(self.features(i, return_tensors="pt")["pixel_values"]) for i in images
        ]
        img_features = torch.cat(img_features, 0)
        img_features = img_features / torch.linalg.norm(img_features, axis=-1, keepdims=True)

        txt_features = [
            self.model.get_text_features(**self.tokenizer(t, padding=True, return_tensors="pt")) for t in text
        ]
        txt_features = torch.cat(txt_features, 0)
        txt_features = txt_features / torch.linalg.norm(txt_features, axis=-1, keepdims=True)

        # cosine similarity between feature vectors
        score = (img_features * txt_features).sum(axis=-1)
        self.score += 100 * score.sum(0)
        self.n_samples += img_features.shape[0]

    def compute(self) -> Tensor:
        """ Calculates the accumulated CLIP score over all samples """
        return self.score / self.n_samples
