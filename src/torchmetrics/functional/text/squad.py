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
# Adapted from:
# Link: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Link: https://github.com/huggingface/datasets/blob/master/metrics/squad/squad.py
import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union

from torch import Tensor, tensor

from torchmetrics.utilities import rank_zero_warn

SINGLE_PRED_TYPE = Dict[str, str]
PREDS_TYPE = Union[SINGLE_PRED_TYPE, List[SINGLE_PRED_TYPE]]
SINGLE_TARGET_TYPE = Dict[str, Union[str, Dict[str, Union[List[str], List[int]]]]]
TARGETS_TYPE = Union[SINGLE_TARGET_TYPE, List[SINGLE_TARGET_TYPE]]
UPDATE_METHOD_SINGLE_PRED_TYPE = Union[List[Dict[str, Union[str, int]]], str, Dict[str, Union[List[str], List[int]]]]

SQuAD_FORMAT = {
    "answers": {"answer_start": [1], "text": ["This is a test text"]},
    "context": "This is a test context.",
    "id": "1",
    "question": "Is this a test?",
    "title": "train test",
}


def _normalize_text(s: str) -> str:
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


def _get_tokens(s: str) -> List[str]:
    """Split a sentence into separate tokens."""
    return [] if not s else _normalize_text(s).split()


def _compute_f1_score(predicted_answer: str, target_answer: str) -> Tensor:
    """Compute F1 Score for two sentences."""
    target_tokens = _get_tokens(target_answer)
    predicted_tokens = _get_tokens(predicted_answer)
    common = Counter(target_tokens) & Counter(predicted_tokens)
    num_same = tensor(sum(common.values()))
    if len(target_tokens) == 0 or len(predicted_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return tensor(int(target_tokens == predicted_tokens))
    if num_same == 0:
        return tensor(0.0)
    precision = 1.0 * num_same / tensor(len(predicted_tokens))
    recall = 1.0 * num_same / tensor(len(target_tokens))
    return (2 * precision * recall) / (precision + recall)


def _compute_exact_match_score(prediction: str, ground_truth: str) -> Tensor:
    """Compute Exact Match for two sentences."""
    return tensor(int(_normalize_text(prediction) == _normalize_text(ground_truth)))


def _metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], Tensor], prediction: str, ground_truths: List[str]
) -> Tensor:
    """Calculate maximum score for a predicted answer with all reference answers."""
    return max(metric_fn(prediction, truth) for truth in ground_truths)  # type: ignore[type-var]


def _squad_input_check(
    preds: PREDS_TYPE, targets: TARGETS_TYPE
) -> Tuple[Dict[str, str], List[Dict[str, List[Dict[str, List[Dict[str, Any]]]]]]]:
    """Check for types and convert the input to necessary format to compute the input."""
    if isinstance(preds, Dict):
        preds = [preds]

    if isinstance(targets, Dict):
        targets = [targets]

    for pred in preds:
        pred_keys = pred.keys()
        if "prediction_text" not in pred_keys or "id" not in pred_keys:
            raise KeyError(
                "Expected keys in a single prediction are 'prediction_text' and 'id'."
                "Please make sure that 'prediction_text' maps to the answer string and 'id' maps to the key string."
            )

    for target in targets:
        target_keys = target.keys()
        if "answers" not in target_keys or "id" not in target_keys:
            raise KeyError(
                "Expected keys in a single target are 'answers' and 'id'."
                "Please make sure that 'answers' maps to a `SQuAD` format dictionary and 'id' maps to the key string.\n"
                "SQuAD Format: "
                f"{SQuAD_FORMAT}"
            )

        answers: Dict[str, Union[List[str], List[int]]] = target["answers"]  # type: ignore[assignment]
        if "text" not in answers.keys():
            raise KeyError(
                "Expected keys in a 'answers' are 'text'."
                "Please make sure that 'answer' maps to a `SQuAD` format dictionary.\n"
                "SQuAD Format: "
                f"{SQuAD_FORMAT}"
            )

    preds_dict = {prediction["id"]: prediction["prediction_text"] for prediction in preds}
    _fn_answer = lambda tgt: {"answers": [{"text": txt} for txt in tgt["answers"]["text"]], "id": tgt["id"]}
    targets_dict = [{"paragraphs": [{"qas": [_fn_answer(target) for target in targets]}]}]
    return preds_dict, targets_dict


def _squad_update(
    preds: Dict[str, str],
    target: List[Dict[str, List[Dict[str, List[Dict[str, Any]]]]]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute F1 Score and Exact Match for a collection of predictions and references.

    Args:
        preds: A dictionary mapping an `id` to the predicted `answer`.
        target:
            A list of dictionary mapping `paragraphs` to list of dictionary mapping `qas` to a list of dictionary
            containing `id` and list of all possible `answers`.

    Return:
        Tuple containing F1 score, Exact match score and total number of examples.

    Example:
        >>> from torchmetrics.functional.text.squad import _squad_update
        >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
        >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]
        >>> preds_dict = {pred["id"]: pred["prediction_text"] for pred in preds}
        >>> targets_dict = [
        ...     dict(paragraphs=[dict(qas=[dict(answers=[
        ...         {"text": txt} for txt in tgt["answers"]["text"]], id=tgt["id"]) for tgt in target
        ...     ])])
        ... ]
        >>> _squad_update(preds_dict, targets_dict)
        (tensor(1.), tensor(1.), tensor(1))
    """
    f1 = tensor(0.0)
    exact_match = tensor(0.0)
    total = tensor(0)
    for article in target:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in preds:
                    rank_zero_warn(f"Unanswered question {qa['id']} will receive score 0.")
                    continue
                ground_truths = [x["text"] for x in qa["answers"]]
                pred = preds[qa["id"]]
                exact_match += _metric_max_over_ground_truths(_compute_exact_match_score, pred, ground_truths)
                f1 += _metric_max_over_ground_truths(_compute_f1_score, pred, ground_truths)

    return f1, exact_match, total


def _squad_compute(f1: Tensor, exact_match: Tensor, total: Tensor) -> Dict[str, Tensor]:
    """Aggregate the F1 Score and Exact match for the batch.

    Return:
        Dictionary containing the F1 score, Exact match score for the batch.
    """
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {"exact_match": exact_match, "f1": f1}


def squad(preds: PREDS_TYPE, target: TARGETS_TYPE) -> Dict[str, Tensor]:
    """Calculate `SQuAD Metric`_ .

    Args:
        preds: A Dictionary or List of Dictionary-s that map `id` and `prediction_text` to the respective values.

            Example prediction:

            .. code-block:: python

                {"prediction_text": "TorchMetrics is awesome", "id": "123"}

        target: A Dictionary or List of Dictionary-s that contain the `answers` and `id` in the SQuAD Format.

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
        >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
        >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]},"id": "56e10a3be3433e1400422b22"}]
        >>> squad(preds, target)
        {'exact_match': tensor(100.), 'f1': tensor(100.)}

    Raises:
        KeyError:
            If the required keys are missing in either predictions or targets.

    References:
        [1] SQuAD: 100,000+ Questions for Machine Comprehension of Text by Pranav Rajpurkar, Jian Zhang, Konstantin
        Lopyrev, Percy Liang `SQuAD Metric`_ .
    """
    preds_dict, target_dict = _squad_input_check(preds, target)
    f1, exact_match, total = _squad_update(preds_dict, target_dict)
    return _squad_compute(f1, exact_match, total)
