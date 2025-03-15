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
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics import Metric
from torchmetrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_10
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CLIPScore.plot"]

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
        __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
else:
    __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
    _CLIPModel = None
    _CLIPProcessor = None


class CLIPScore(Metric):
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

    .. caution::
        Metric is not scriptable

    .. note::
        The default CLIP and processor used in this implementation has a maximum sequence length of 77 for text
        inputs. If you need to process longer captions, you can use the `zer0int/LongCLIP-L-Diffusers` model which
        has a maximum sequence length of 248.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - source: Source input.

        This can be:

        - Images: ``Tensor`` or list of ``Tensor``

            If a single tensor, it should have shape ``(N, C, H, W)``.
            If a list of tensors, each tensor should have shape ``(C, H, W)``.
            ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.

        - Text: ``str`` or list of ``str``

            Either a single caption or a list of captions.

    - target: Target input.

        This can be:

        - Images: ``Tensor`` or list of ``Tensor``

            If a single tensor, it should have shape ``(N, C, H, W)``.
            If a list of tensors, each tensor should have shape ``(C, H, W)``.
            ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.

        - Text: ``str`` or list of ``str``

            Either a single caption or a list of captions.

    As output of `forward` and `compute` the metric returns the following output

    - ``clip_score`` (:class:`~torch.Tensor`): float scalar tensor with mean CLIP score over samples

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`
            - `"jinaai/jina-clip-v2"`
            - `"zer0int/LongCLIP-L-Diffusers"`
            - `"zer0int/LongCLIP-GmP-ViT-L-14"`

            Alternatively, a callable function that returns a tuple of CLIP compatible model and processor instances
            can be passed in. By compatible, we mean that the processors `__call__` method should accept a list of
            strings and list of images and that the model should have a `get_image_features` and `get_text_features`
            methods.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0

    Example:
        >>> from torchmetrics.multimodal.clip_score import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> image = torch.randint(255, (3, 224, 224), generator=torch.Generator().manual_seed(42))
        >>> score = metric(image, "a photo of a cat")
        >>> score.detach().round()
        tensor(24.)

    Example:
        >>> from torchmetrics.multimodal.clip_score import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> image1 = torch.randint(255, (3, 224, 224), generator=torch.Generator().manual_seed(42))
        >>> image2 = torch.randint(255, (3, 224, 224), generator=torch.Generator().manual_seed(43))
        >>> score = metric(image1, image2)
        >>> score.detach().round()
        tensor(99.)

    Example:
        >>> from torchmetrics.multimodal.clip_score import CLIPScore
        >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        >>> score = metric("28-year-old chef found dead in San Francisco mall",
        ...               "A 28-year-old chef who recently moved to San Francisco was found dead.")
        >>> score.detach().round()
        tensor(91.)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound = 100.0

    score: Tensor
    n_samples: Tensor
    feature_network: str = "model"

    def __init__(
        self,
        model_name_or_path: Union[
            Literal[
                "openai/clip-vit-base-patch16",
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-large-patch14-336",
                "openai/clip-vit-large-patch14",
                "jinaai/jina-clip-v2",
                "zer0int/LongCLIP-L-Diffusers",
                "zer0int/LongCLIP-GmP-ViT-L-14",
            ],
            Callable[[], tuple[_CLIPModel, _CLIPProcessor]],
        ] = "openai/clip-vit-large-patch14",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_clip_model_and_processor(model_name_or_path)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(
        self, source: Union[Tensor, List[Tensor], List[str], str], target: Union[Tensor, List[Tensor], List[str], str]
    ) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            source: Source input. This can be:
                - Images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors.
                - Text: Either a single caption or a list of captions.
            target: Target input. This can be:
                - Images: Either a single [N, C, H, W] tensor or a list of [C, H, W] tensors.
                - Text: Either a single caption or a list of captions.

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        score, n_samples = _clip_score_update(source, target, self.model, self.processor)
        self.score += score.sum(0)
        self.n_samples += n_samples

    def compute(self) -> Tensor:
        """Compute accumulated clip score."""
        return torch.max(self.score / self.n_samples, torch.zeros_like(self.score))

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.multimodal.clip_score import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> metric.update(torch.randint(255, (3, 224, 224)), "a photo of a cat")
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal.clip_score import CLIPScore
            >>> metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(255, (3, 224, 224)), "a photo of a cat"))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
