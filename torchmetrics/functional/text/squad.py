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
# Adapted from:
# Link: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Link: https://github.com/huggingface/datasets/blob/master/metrics/squad/squad.py
import re
import string
from collections import Counter
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor, tensor

from torchmetrics.utilities import rank_zero_warn

SINGLE_PRED_TYPE = Dict[str, str]
PREDS_TYPE = Union[SINGLE_PRED_TYPE, List[SINGLE_PRED_TYPE]]
SINGLE_TARGET_TYPE = Dict[str, Union[str, Dict[str, Union[List[str], List[int]]]]]
TARGETS_TYPE = Union[SINGLE_TARGET_TYPE, List[SINGLE_TARGET_TYPE]]


def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Split a sentence into separate tokens."""
    if not s:
        return []
    return normalize_text(s).split()


def compute_f1_score(predictied_answer, target_answer) -> Tensor:
    """Compute F1 Score for two sentences."""
    target_tokens: Tensor = get_tokens(target_answer)
    predicted_tokens: Tensor = get_tokens(predictied_answer)
    common = Counter(target_tokens) & Counter(predicted_tokens)
    num_same: Tensor = tensor(sum(common.values()))
    if len(target_tokens) == 0 or len(predicted_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return tensor(int(target_tokens == predicted_tokens))
    if num_same == 0:
        return tensor(0.0)
    precision: Tensor = 1.0 * num_same / tensor(len(predicted_tokens))
    recall: Tensor = 1.0 * num_same / tensor(len(target_tokens))
    f1: Tensor = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact_match_score(prediction, ground_truth) -> Tensor:
    """Compute Exact Match for two sentences."""
    return tensor(int(normalize_text(prediction) == normalize_text(ground_truth)))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths) -> Tensor:
    """Calculate maximum score for a predicted answer with all reference answers."""
    scores_for_ground_truths: List[Tensor] = []
    for ground_truth in ground_truths:
        score: Tensor = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return torch.max(tensor(scores_for_ground_truths))


def _squad_update(
    preds: Dict[str, str], targets: List[Dict[str, List[Dict[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]]]]]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute F1 Score and Exact Match for a collection of predictions and references.

    Args:
        preds:
            A dictionary mapping an `id` to the predicted `answer`.
        targets:
            A list of dictionary mapping `paragraphs` to list of dictionary mapping `qas` to a list of dictionary
            containing `id` and list of all possible `answers`.

    Return:
        Tuple containing F1 score, Exact match score and total number of examples.

    Example:
        >>> from torchmetrics.functional.text.squad import _squad_update
        >>> predictions = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
        >>> targets = [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]
        >>> preds_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
        >>> targets_dict = [
        ...     {
        ...         "paragraphs": [
        ...             {
        ...                 "qas": [
        ...                     {
        ...                         "answers": [{"text": answer_text} for answer_text in target["answers"]["text"]],
        ...                         "id": target["id"],
        ...                     }
        ...                     for target in targets
        ...                 ]
        ...             }
        ...         ]
        ...     }
        ... ]
        >>> _squad_update(preds_dict, targets_dict)
        (tensor(1.), tensor(1.), tensor(1))
    """
    f1: Tensor = tensor(0.0)
    exact_match: Tensor = tensor(0.0)
    total: Tensor = tensor(0)
    for article in targets:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in preds:
                    rank_zero_warn(f"Unanswered question {qa['id']} will receive score 0.")
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = preds[qa["id"]]
                exact_match += metric_max_over_ground_truths(compute_exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(compute_f1_score, prediction, ground_truths)

    return f1, exact_match, total


def _squad_compute(scores: Tuple[Tensor, Tensor, Tensor]) -> Dict[str, Tensor]:
    """Aggregate the F1 Score and Exact match for the batch.

    Args:
        scores:
            F1 Score, Exact Match, and Total number of examples in the batch

    Return:
        Dictionary containing the F1 score, Exact match score for the batch.
    """
    f1: Tensor = scores[0]
    exact_match: Tensor = scores[1]
    total: Tensor = scores[2]
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {"exact_match": exact_match, "f1": f1}


def squad(
    preds: PREDS_TYPE,
    targets: TARGETS_TYPE,
) -> Dict[str, Tensor]:
    """Calculate `SQuAD Metric`_ .

    Args:
        preds:
            A Dictionary or List of Dictionary-s that map `id` and `prediction_text` to the respective values.

            Example prediction:

            .. code-block:: python

                {"prediction_text": "TorchMetrics is awesome", "id": "123"}

        targets:
            A Dictioinary or List of Dictionary-s that contain the `answers` and `id` in the SQuAD Format.

            Example target:

            .. code-block:: python

                {
                    'answers': [{'answer_start': [1], 'text': ['This is a test answer']}],
                    'id': '1',
                }

            Reference SQuAD Format:

            .. code-block:: python

                {
                    'answers': {'answer_start': [1], 'text': ['This is a test text']},
                    'context': 'This is a test context.',
                    'id': '1',
                    'question': 'Is this a test?',
                    'title': 'train test'
                }


    Return:
        Dictionary containing the F1 score, Exact match score for the batch.

    Example:
        >>> from torchmetrics.functional.text.squad import squad
        >>> predictions = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
        >>> references = [{"answers": {"answer_start": [97], "text": ["1976"]},"id": "56e10a3be3433e1400422b22"}]
        >>> squad(predictions, references)
        {'exact_match': tensor(100.), 'f1': tensor(100.)}

    Raises:
        KeyError:
            If the required keys are missing in either predictions or targets.

    References:
        [1] SQuAD: 100,000+ Questions for Machine Comprehension of Text by Pranav Rajpurkar, Jian Zhang, Konstantin
        Lopyrev, Percy Liang `SQuAD Metric`_ .
    """

    if isinstance(preds, Dict):
        preds = [preds]

    if isinstance(targets, Dict):
        targets = [targets]

    for pred in preds:
        keys = pred.keys()
        if "prediction_text" not in keys or "id" not in keys:
            raise KeyError(
                "Expected keys in a single prediction are 'prediction_text' and 'id'."
                "Please make sure that 'prediction_text' maps to the answer string and 'id' maps to the key string."
            )

    for target in targets:
        keys = target.keys()
        if "answers" not in keys or "id" not in keys:
            raise KeyError(
                "Expected keys in a single target are 'answers' and 'id'."
                "Please make sure that 'answers' maps to a `SQuAD` format dictionary and 'id' maps to the key string.\n"
                "SQuAD Format: "
                "{"
                "    'answers': {"
                "        'answer_start': [1],"
                "        'text': ['This is a test text']"
                "    },"
                "    'context': 'This is a test context.',"
                "    'id': '1',"
                "    'question': 'Is this a test?',"
                "    'title': 'train test'"
                "}"
            )

        answers_keys = target["answers"].keys()
        if "text" not in answers_keys:
            raise KeyError(
                "Expected keys in a 'answers' are 'text'."
                "Please make sure that 'answer' maps to a `SQuAD` format dictionary.\n"
                "SQuAD Format: "
                "{"
                "    'answers': {"
                "        'answer_start': [1],"
                "        'text': ['This is a test text']"
                "    },"
                "    'context': 'This is a test context.',"
                "    'id': '1',"
                "    'question': 'Is this a test?',"
                "    'title': 'train test'"
                "}"
            )

    preds_dict = {prediction["id"]: prediction["prediction_text"] for prediction in preds}
    targets_dict = [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "answers": [{"text": answer_text} for answer_text in target["answers"]["text"]],
                            "id": target["id"],
                        }
                        for target in targets
                    ]
                }
            ]
        }
    ]
    scores: Tuple[Tensor, Tensor, Tensor] = _squad_update(preds_dict, targets_dict)
    return _squad_compute(scores)
