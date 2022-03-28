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
from collections import namedtuple

Input = namedtuple("Input", ["preds", "targets"])
SquadInput = namedtuple("SquadInput", ["preds", "targets", "exact_match", "f1"])

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

_inputs_single_sentence_multiple_references = Input(preds=[HYPOTHESIS_B], targets=[[REFERENCE_1B, REFERENCE_2B]])

_inputs_multiple_references = Input(preds=TUPLE_OF_HYPOTHESES, targets=TUPLE_OF_REFERENCES)

_inputs_single_sentence_single_reference = Input(preds=HYPOTHESIS_B, targets=REFERENCE_1B)

ERROR_RATES_BATCHES_1 = {
    "preds": [["hello world"], ["what a day"]],
    "targets": [["hello world"], ["what a wonderful day"]],
}

ERROR_RATES_BATCHES_2 = {
    "preds": [
        ["i like python", "what you mean or swallow"],
        ["hello duck", "i like python"],
    ],
    "targets": [
        ["i like monthy python", "what do you mean, african or european swallow"],
        ["hello world", "i like monthy python"],
    ],
}

_inputs_error_rate_batch_size_1 = Input(**ERROR_RATES_BATCHES_1)

_inputs_error_rate_batch_size_2 = Input(**ERROR_RATES_BATCHES_2)

SAMPLE_1 = {
    "exact_match": 100.0,
    "f1": 100.0,
    "preds": {"prediction_text": "1976", "id": "id1"},
    "targets": {"answers": {"answer_start": [97], "text": ["1976"]}, "id": "id1"},
}

SAMPLE_2 = {
    "exact_match": 0.0,
    "f1": 0.0,
    "preds": {"prediction_text": "Hello", "id": "id2"},
    "targets": {"answers": {"answer_start": [97], "text": ["World"]}, "id": "id2"},
}

BATCH = {
    "exact_match": [100.0, 0.0],
    "f1": [100.0, 0.0],
    "preds": [
        {"prediction_text": "1976", "id": "id1"},
        {"prediction_text": "Hello", "id": "id2"},
    ],
    "targets": [
        {"answers": {"answer_start": [97], "text": ["1976"]}, "id": "id1"},
        {"answers": {"answer_start": [97], "text": ["World"]}, "id": "id2"},
    ],
}

_inputs_squad_exact_match = SquadInput(
    preds=SAMPLE_1["preds"], targets=SAMPLE_1["targets"], exact_match=SAMPLE_1["exact_match"], f1=SAMPLE_1["f1"]
)

_inputs_squad_exact_mismatch = SquadInput(
    preds=SAMPLE_2["preds"], targets=SAMPLE_2["targets"], exact_match=SAMPLE_2["exact_match"], f1=SAMPLE_2["f1"]
)

_inputs_squad_batch_match = SquadInput(
    preds=BATCH["preds"], targets=BATCH["targets"], exact_match=BATCH["exact_match"], f1=BATCH["f1"]
)

# single reference
TUPLE_OF_SINGLE_REFERENCES = ((REFERENCE_1A, REFERENCE_1B), (REFERENCE_1B, REFERENCE_1C))
_inputs_single_reference = Input(preds=TUPLE_OF_HYPOTHESES, targets=TUPLE_OF_SINGLE_REFERENCES)
