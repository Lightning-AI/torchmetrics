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
from collections.abc import Sequence
from functools import partial

import pytest
from torch import Tensor, tensor

from torchmetrics.functional.text.meteor import meteor_score
from torchmetrics.text.meteor import METEORScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
from unittests.text._helpers import TextTester
from unittests.text._inputs import _inputs_multiple_references, _inputs_single_sentence_multiple_references

pytestmark = pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk package to be installed")


def _reference_nltk_meteor(
    preds: Sequence[str],
    targets: Sequence[Sequence[str]],
    alpha: float,
    beta: float,
    gamma: float,
) -> Tensor:
    import nltk
    from nltk.translate.meteor_score import meteor_score as nltk_meteor

    nltk.download("wordnet", quiet=True)

    scores = []
    for pred, references in zip(preds, targets):
        tokenized_references = [reference.lower().split() for reference in references]
        scores.append(nltk_meteor(tokenized_references, pred.lower().split(), alpha=alpha, beta=beta, gamma=gamma))
    return tensor(sum(scores) / len(scores))


@pytest.mark.parametrize(
    ("alpha", "beta", "gamma"),
    [
        (0.9, 3.0, 0.5),
        (0.5, 3.0, 0.5),
        (0.9, 2.0, 0.7),
    ],
)
@pytest.mark.parametrize(
    ("preds", "targets"),
    [(_inputs_multiple_references.preds, _inputs_multiple_references.target)],
)
class TestMETEORScore(TextTester):
    """Test class for `METEORScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_meteor_score_class(self, ddp, preds, targets, alpha, beta, gamma):
        """Test class implementation of metric."""
        metric_args = {"alpha": alpha, "beta": beta, "gamma": gamma}
        reference = partial(_reference_nltk_meteor, alpha=alpha, beta=beta, gamma=gamma)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=METEORScore,
            reference_metric=reference,
            metric_args=metric_args,
        )

    def test_meteor_score_functional(self, preds, targets, alpha, beta, gamma):
        """Test functional implementation of metric."""
        metric_args = {"alpha": alpha, "beta": beta, "gamma": gamma}
        reference = partial(_reference_nltk_meteor, alpha=alpha, beta=beta, gamma=gamma)

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=meteor_score,
            reference_metric=reference,
            metric_args=metric_args,
        )


def test_meteor_empty_functional():
    """Test that METEOR returns 0 when no input is provided."""
    assert meteor_score([], [[]]) == tensor(0.0)


def test_meteor_empty_class():
    """Test that METEOR returns 0 when no input is provided."""
    meteor = METEORScore()
    assert meteor([], [[]]) == tensor(0.0)


def test_meteor_return_sentence_level_score_functional():
    """Test that METEOR can return sentence level scores."""
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.target
    _, sentence_score = meteor_score(preds, targets, return_sentence_level_score=True)
    assert isinstance(sentence_score, Tensor)


def test_meteor_return_sentence_level_class():
    """Test that METEOR can return sentence level scores."""
    meteor = METEORScore(return_sentence_level_score=True)
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.target
    _, sentence_score = meteor(preds, targets)
    assert isinstance(sentence_score, Tensor)


@pytest.mark.parametrize(("arg", "value"), [("alpha", 1.5), ("beta", -1.0), ("gamma", -0.5)])
def test_meteor_invalid_arguments(arg, value):
    """Test that METEOR raises an error for out of range arguments."""
    with pytest.raises(ValueError, match=f"Expected argument `{arg}`"):
        METEORScore(**{arg: value})
