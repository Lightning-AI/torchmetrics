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
from functools import partial
from typing import Sequence

import pytest
from torch import Tensor, tensor
from torchmetrics.functional.text.chrf import chrf_score
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.utilities.imports import _SACREBLEU_AVAILABLE

from unittests.text.helpers import TextTester
from unittests.text.inputs import _inputs_multiple_references, _inputs_single_sentence_multiple_references

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import CHRF


def _sacrebleu_chrf_fn(
    preds: Sequence[str],
    targets: Sequence[Sequence[str]],
    char_order: int,
    word_order: int,
    lowercase: bool,
    whitespace: bool,
) -> Tensor:
    sacrebleu_chrf = CHRF(
        char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace, eps_smoothing=True
    )
    # Sacrebleu CHRF expects different format of input
    targets = [[target[i] for target in targets] for i in range(len(targets[0]))]
    sacrebleu_chrf = sacrebleu_chrf.corpus_score(preds, targets).score / 100
    return tensor(sacrebleu_chrf)


@pytest.mark.parametrize(
    ["char_order", "word_order", "lowercase", "whitespace"],
    [
        (6, 2, False, False),
        (6, 2, False, True),
        (4, 2, True, False),
        (6, 0, True, False),
        (6, 0, True, True),
        (4, 0, False, True),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_multiple_references.preds, _inputs_multiple_references.target)],
)
@pytest.mark.skipif(not _SACREBLEU_AVAILABLE, reason="test requires sacrebleu")
class TestCHRFScore(TextTester):
    """Test class for `CHRFScore` metric."""

    @pytest.mark.parametrize("ddp", [False, True])
    def test_chrf_score_class(self, ddp, preds, targets, char_order, word_order, lowercase, whitespace):
        """Test class implementation of metric."""
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }
        nltk_metric = partial(
            _sacrebleu_chrf_fn, char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CHRFScore,
            reference_metric=nltk_metric,
            metric_args=metric_args,
        )

    def test_chrf_score_functional(self, preds, targets, char_order, word_order, lowercase, whitespace):
        """Test functional implementation of metric."""
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }
        nltk_metric = partial(
            _sacrebleu_chrf_fn, char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=chrf_score,
            reference_metric=nltk_metric,
            metric_args=metric_args,
        )

    def test_chrf_score_differentiability(self, preds, targets, char_order, word_order, lowercase, whitespace):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CHRFScore,
            metric_functional=chrf_score,
            metric_args=metric_args,
        )


def test_chrf_empty_functional():
    """Test that eed returns 0 when no input is provided."""
    preds = []
    targets = [[]]
    assert chrf_score(preds, targets) == tensor(0.0)


def test_chrf_empty_class():
    """Test that eed returns 0 when no input is provided."""
    chrf = CHRFScore()
    preds = []
    targets = [[]]
    assert chrf(preds, targets) == tensor(0.0)


def test_chrf_return_sentence_level_score_functional():
    """Test that chrf can return sentence level scores."""
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.target
    _, chrf_sentence_score = chrf_score(preds, targets, return_sentence_level_score=True)
    isinstance(chrf_sentence_score, Tensor)


def test_chrf_return_sentence_level_class():
    """Test that chrf can return sentence level scores."""
    chrf = CHRFScore(return_sentence_level_score=True)
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.target
    _, chrf_sentence_score = chrf(preds, targets)
    isinstance(chrf_sentence_score, Tensor)
