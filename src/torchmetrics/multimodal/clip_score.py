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
from transformers import CLIPFeatureExtractor as _CLIPFeatureExtractor
from transformers import CLIPModel as _CLIPModel
from transformers import CLIPTokenizer as _CLIPTokenizer

from torchmetrics import Metric


class CLIPScore(Metric):
    def __init__(self, version="openai/clip-vit-large-patch14", **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = _CLIPTokenizer.from_pretrained(version)
        self.model = _CLIPModel.from_pretrained(version)
        self.features = _CLIPFeatureExtractor.from_pretrained(version)
        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, images: Union[Tensor, List[Tensor]], text: Union[str, List[str]]) -> None:
        if not isinstance(images, List):
            images = [images]
        if not isinstance(text, List):
            text = [text]

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
        self.score += score.sum(0)
        self.n_samples += img_features.shape[0]

    def compute(self) -> Tensor:
        return self.score / self.n_samples


if __name__ == "__main__":
    img = torch.randint(255, (3, 224, 224))
    text = "min hest er meget flot"
    metric = CLIPScore()
    metric.update(img, text)
