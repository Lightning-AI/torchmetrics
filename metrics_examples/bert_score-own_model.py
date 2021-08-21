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
"""
An example of how to use BERTScore with a user's defined/own model and tokenizer.

To run:
python bert_score-own_model.py
"""

from pprint import pprint
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torchmetrics import BERTScore


_NUM_LAYERS = 2
_MODEL_DIM = 4
_NHEAD = 2
_MAX_LEN = 6


class UserTokenizer(object):

    def __init__(self):
        self.cls_token = "<cls>"
        self.sep_token = "<sep>"
        self.pad_token = "<pad>"

        self.word2vec = {
            "hello": 0.5 * torch.ones(1, _MODEL_DIM),
            "world": -0.5 * torch.ones(1, _MODEL_DIM),
            self.cls_token: torch.zeros(1, _MODEL_DIM),
            self.sep_token: torch.zeros(1, _MODEL_DIM),
            self.pad_token: torch.zeros(1, _MODEL_DIM),
        }

    def __call__(self, sentences: Union[str, List[str]], max_len: int = _MAX_LEN) -> Dict[str, torch.Tensor]:
        output_dict: Dict[str, torch.Tensor] = {}
        if isinstance(sentences, str):
            sentences = [sentences]
        # Add special tokens
        sentences = [" ".join([self.cls_token, sentence, self.sep_token]) for sentence in sentences]
        # Tokennize sentence
        tokenized_sentences = [
            sentence.lower().split()[:max_len] + [self.pad_token] * (max_len - len(sentence.lower().split()))
            for sentence in sentences
        ]
        output_dict["input_ids"] = torch.cat(
            [torch.cat([self.word2vec[word] for word in sentence]).unsqueeze(0) for sentence in tokenized_sentences]
        )
        output_dict["attention_mask"] = torch.cat(
            [
                torch.tensor([1 if word != self.pad_token else 0 for word in sentence]).unsqueeze(0)
                for sentence in tokenized_sentences
            ]
        ).long()

        return output_dict


def get_user_model_encoder(
    num_layers: int = _NUM_LAYERS, d_model: int = _MODEL_DIM, nhead: int = _NHEAD
) -> torch.Tensor:
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    return transformer_encoder


def user_forward_fn(model: torch.Tensor, batch: Dict[str, torch.Tensor]):
    return model(batch["input_ids"])


_PREDS = ["hello", "hello world", "world world world"]
_REFS = ["hello", "hello hello", "hello world hello"]


if __name__ == "__main__":
    tokenizer = UserTokenizer()
    model = get_user_model_encoder()

    Scorer = BERTScore(
        model=model, user_tokenizer=tokenizer, user_forward_fn=user_forward_fn, max_length=_MAX_LEN, return_hash=False
    )
    Scorer.update(_PREDS, _REFS)
    print("Predictions")
    pprint(Scorer.predictions)

    pprint(Scorer.compute())
