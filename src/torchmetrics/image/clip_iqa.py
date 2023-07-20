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
from typing import Any, Literal, Optional

import torch
from torch import Tensor

from torchmetrics.functional.image.clip_iqa import _clip_iqa_update
from torchmetrics.functional.multimodal.clip_score import _get_model_and_processor
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
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
    """Something should be here."""

    def __init__(
        self,
        model_name_or_path: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = _DEFAULT_MODEL,  # type: ignore[assignment]
        reduction: Optional[Literal["sum", "mean", "none"]] = None,  # type: ignore[assignment
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.model, self.processor = _get_model_and_processor(model_name_or_path)

        text_processed = self.processor(text=["Good photo.", "Bad photo."], return_tensors="pt", padding=True)
        anchors = self.model.get_text_features(text_processed["input_ids"], text_processed["attention_mask"])
        anchors = anchors / anchors.norm(p=2, dim=-1, keepdim=True)
        self.register_buffer("anchors", anchors)

        if reduction is None or reduction == "none":
            self.add_state("score_list", torch.tensor([], dtype=torch.float32), dist_reduce_fx=None)
        else:
            self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        """Update metric state with new data."""
        probs = _clip_iqa_update(x, self.model, self.processor)
        if self.reduction is None or self.reduction == "none":
            self.score_list.append(probs)
        else:
            self.score += probs.sum()
            self.n_samples += probs.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.score_list if self.reduction is None or self.reduction == "none" else self.score / self.n_samples
