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
"""An example of how to use BERTScore with a user's defined/own model and tokenizer.

To run: python bert_score-own_model.py
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


class UserTokenizer:
    """The `UserTokenizer` class is required to be defined when a non-default model (i.e. not one from
    `transformers`) is used.

    The user's defined tokenizer is expected to return either token IDs or token embeddings that are fed into the model.
    The tokenizer vocabulary should contain some special tokens, such as a `<pad>` token so that a tokenization will run
    successfully in batches.
    """

    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"
    PAD_TOKEN = "<pad>"

    def __init__(self) -> None:
        self.word2vec = {
            "hello": 0.5 * torch.ones(1, _MODEL_DIM),
            "world": -0.5 * torch.ones(1, _MODEL_DIM),
            self.CLS_TOKEN: torch.zeros(1, _MODEL_DIM),
            self.SEP_TOKEN: torch.zeros(1, _MODEL_DIM),
            self.PAD_TOKEN: torch.zeros(1, _MODEL_DIM),
        }

    def __call__(self, sentences: Union[str, List[str]], max_len: int = _MAX_LEN) -> Dict[str, torch.Tensor]:
        """The `__call__` method must be defined for this class. To ensure the functionality, the `__call__` method
        should obey the input/output arguments structure described below.

        Args:
            sentences:
                Input text. `Union[str, List[str]]`
            max_len:
                Maximum length of pre-processed text. `int`

        Return:
            Python dictionary containing the keys `input_ids` and `attention_mask` with corresponding `torch.Tensor`
            values.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        if isinstance(sentences, str):
            sentences = [sentences]
        # Add special tokens
        sentences = [" ".join([self.CLS_TOKEN, sentence, self.SEP_TOKEN]) for sentence in sentences]
        # Tokennize sentence
        tokenized_sentences = [
            sentence.lower().split()[:max_len] + [self.PAD_TOKEN] * (max_len - len(sentence.lower().split()))
            for sentence in sentences
        ]
        output_dict["input_ids"] = torch.cat(
            [torch.cat([self.word2vec[word] for word in sentence]).unsqueeze(0) for sentence in tokenized_sentences]
        )
        output_dict["attention_mask"] = torch.cat(
            [
                torch.tensor([1 if word != self.PAD_TOKEN else 0 for word in sentence]).unsqueeze(0)
                for sentence in tokenized_sentences
            ]
        ).long()

        return output_dict


def get_user_model_encoder(
    num_layers: int = _NUM_LAYERS, d_model: int = _MODEL_DIM, nhead: int = _NHEAD
) -> torch.nn.Module:
    """Initialize the Transformer encoder."""
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    return transformer_encoder


def user_forward_fn(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """User forward function used for the computation of model embeddings.

    This function might be arbitrarily complicated inside. However, to ensure functionality, it should obey the
    input/output argument structure described below.

    Args:
        model:
            `torch.nn.Module`
        batch:
            `Dict[str, torch.Tensor]`

    Return:
        The model output. `torch.Tensor`
    """
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
