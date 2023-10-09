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
from typing import NamedTuple

import torch
from torch import Tensor

from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES, _Input
from unittests.helpers import seed_all

seed_all(1)


class _SquadInput(NamedTuple):
    preds: Tensor
    target: Tensor
    exact_match: Tensor
    f1: Tensor


# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu and adjusted
# EXAMPLE 1
HYPOTHESIS_A = "It is a guide to action which ensures that the military always obeys the commands of the party"
REFERENCE_1A = "It is a guide to action that ensures that the military will forever heed Party commands"
REFERENCE_2A = "It is a guiding principle which makes the military forces always being under the command of the Party"

# EXAMPLE 2
HYPOTHESIS_B = "he read the book because he was interested in world history"
REFERENCE_1B = "he was interested in world history because he read the book"
REFERENCE_2B = "It is the practical guide for the army always to heed the directions of the party"

# EXAMPLE 3 (add intentionally whitespaces)
HYPOTHESIS_C = "the cat the   cat on the mat "
REFERENCE_1C = "the  cat is     on the mat "
REFERENCE_2C = "there is a   cat on the mat"

TUPLE_OF_REFERENCES = (
    ((REFERENCE_1A, REFERENCE_2A), (REFERENCE_1B, REFERENCE_2B)),
    ((REFERENCE_1B, REFERENCE_2B), (REFERENCE_1C, REFERENCE_2C)),
)
TUPLE_OF_HYPOTHESES = ((HYPOTHESIS_A, HYPOTHESIS_B), (HYPOTHESIS_B, HYPOTHESIS_C))

_inputs_single_sentence_multiple_references = _Input(preds=[HYPOTHESIS_B], target=[[REFERENCE_1B, REFERENCE_2B]])

_inputs_multiple_references = _Input(preds=TUPLE_OF_HYPOTHESES, target=TUPLE_OF_REFERENCES)

_inputs_single_sentence_single_reference = _Input(preds=HYPOTHESIS_B, target=REFERENCE_1B)

ERROR_RATES_BATCHES_1 = {
    "preds": [["hello world"], ["what a day"]],
    "target": [["hello world"], ["what a wonderful day"]],
}

ERROR_RATES_BATCHES_2 = {
    "preds": [
        ["i like python", "what you mean or swallow"],
        ["hello duck", "i like python"],
    ],
    "target": [
        ["i like monthy python", "what do you mean, african or european swallow"],
        ["hello world", "i like monthy python"],
    ],
}

_inputs_error_rate_batch_size_1 = _Input(**ERROR_RATES_BATCHES_1)

_inputs_error_rate_batch_size_2 = _Input(**ERROR_RATES_BATCHES_2)

SAMPLE_1 = {
    "exact_match": 100.0,
    "f1": 100.0,
    "preds": {"prediction_text": "1976", "id": "id1"},
    "target": {"answers": {"answer_start": [97], "text": ["1976"]}, "id": "id1"},
}

SAMPLE_2 = {
    "exact_match": 0.0,
    "f1": 0.0,
    "preds": {"prediction_text": "Hello", "id": "id2"},
    "target": {"answers": {"answer_start": [97], "text": ["World"]}, "id": "id2"},
}

BATCH = {
    "exact_match": [100.0, 0.0],
    "f1": [100.0, 0.0],
    "preds": [
        {"prediction_text": "1976", "id": "id1"},
        {"prediction_text": "Hello", "id": "id2"},
    ],
    "target": [
        {"answers": {"answer_start": [97], "text": ["1976"]}, "id": "id1"},
        {"answers": {"answer_start": [97], "text": ["World"]}, "id": "id2"},
    ],
}

_inputs_squad_exact_match = _SquadInput(
    preds=SAMPLE_1["preds"], target=SAMPLE_1["target"], exact_match=SAMPLE_1["exact_match"], f1=SAMPLE_1["f1"]
)

_inputs_squad_exact_mismatch = _SquadInput(
    preds=SAMPLE_2["preds"], target=SAMPLE_2["target"], exact_match=SAMPLE_2["exact_match"], f1=SAMPLE_2["f1"]
)

_inputs_squad_batch_match = _SquadInput(
    preds=BATCH["preds"], target=BATCH["target"], exact_match=BATCH["exact_match"], f1=BATCH["f1"]
)

# single reference
TUPLE_OF_SINGLE_REFERENCES = ((REFERENCE_1A, REFERENCE_1B), (REFERENCE_1B, REFERENCE_1C))
_inputs_single_reference = _Input(preds=TUPLE_OF_HYPOTHESES, target=TUPLE_OF_SINGLE_REFERENCES)

# Logits-based inputs for perplexity metrics
_logits_inputs_fp32 = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, NUM_CLASSES, dtype=torch.float32),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)
_logits_inputs_fp64 = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM, NUM_CLASSES, dtype=torch.float64),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
)

MASK_INDEX = -100
_target_with_mask = _logits_inputs_fp32.target.clone()
_target_with_mask[:, 0, 1:] = MASK_INDEX
_target_with_mask[:, BATCH_SIZE - 1, :] = MASK_INDEX
_logits_inputs_fp32_with_mask = _Input(preds=_logits_inputs_fp32.preds, target=_target_with_mask)
_logits_inputs_fp64_with_mask = _Input(preds=_logits_inputs_fp64.preds, target=_target_with_mask)
