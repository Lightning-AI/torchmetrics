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
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor, tensor
from torchmetrics.functional.text.sacre_bleu import AVAILABLE_TOKENIZERS, _TokenizersLiteral, sacre_bleu_score
from torchmetrics.text.sacre_bleu import SacreBLEUScore

from unittests.text._helpers import TextTester
from unittests.text._inputs import _inputs_multiple_references


def _reference_sacre_bleu(
    preds: Sequence[str], targets: Sequence[Sequence[str]], tokenize: str, lowercase: bool
) -> Tensor:
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        pytest.skip("test requires sacrebleu package to be installed")

    sacrebleu_fn = BLEU(tokenize=tokenize, lowercase=lowercase)
    # Sacrebleu expects different format of input
    targets = [[target[i] for target in targets] for i in range(len(targets[0]))]
    sacrebleu_score = sacrebleu_fn.corpus_score(preds, targets).score / 100
    return tensor(sacrebleu_score)


@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_multiple_references.preds, _inputs_multiple_references.target)],
)
@pytest.mark.parametrize(["lowercase"], [(False,), (True,)])
@pytest.mark.parametrize("tokenize", AVAILABLE_TOKENIZERS)
class TestSacreBLEUScore(TextTester):
    """Test class for `SacreBLEUScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_bleu_score_class(self, ddp, preds, targets, tokenize, lowercase):
        """Test class implementation of metric."""
        metric_args = {"tokenize": tokenize, "lowercase": lowercase}
        original_sacrebleu = partial(_reference_sacre_bleu, tokenize=tokenize, lowercase=lowercase)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=SacreBLEUScore,
            reference_metric=original_sacrebleu,
            metric_args=metric_args,
        )

    def test_bleu_score_functional(self, preds, targets, tokenize, lowercase):
        """Test functional implementation of metric."""
        metric_args = {"tokenize": tokenize, "lowercase": lowercase}
        original_sacrebleu = partial(_reference_sacre_bleu, tokenize=tokenize, lowercase=lowercase)

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=sacre_bleu_score,
            reference_metric=original_sacrebleu,
            metric_args=metric_args,
        )

    def test_bleu_score_differentiability(self, preds, targets, tokenize, lowercase):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {"tokenize": tokenize, "lowercase": lowercase}

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=SacreBLEUScore,
            metric_functional=sacre_bleu_score,
            metric_args=metric_args,
        )


def test_no_and_uniform_weights_functional():
    """Test that implementation works with no weights and uniform weights, and it gives the same result."""
    preds = ["My full pytorch-lightning"]
    targets = [["My full pytorch-lightning test", "Completely Different"]]
    no_weights_score = sacre_bleu_score(preds, targets, n_gram=2)
    uniform_weights_score = sacre_bleu_score(preds, targets, n_gram=2, weights=[0.5, 0.5])
    assert no_weights_score == uniform_weights_score


def test_no_and_uniform_weights_class():
    """Test that implementation works with no weights and uniform weights, and it gives the same result."""
    no_weights_bleu = SacreBLEUScore(n_gram=2)
    uniform_weights_bleu = SacreBLEUScore(n_gram=2, weights=[0.5, 0.5])

    preds = ["My full pytorch-lightning"]
    targets = [["My full pytorch-lightning test", "Completely Different"]]
    no_weights_score = no_weights_bleu(preds, targets)
    uniform_weights_score = uniform_weights_bleu(preds, targets)
    assert no_weights_score == uniform_weights_score


def test_tokenize_ja_mecab():
    """Test that `ja-mecab` tokenizer works on a Japanese text in alignment with the SacreBleu implementation."""
    sacrebleu = SacreBLEUScore(tokenize="ja-mecab")

    preds = ["これは美しい花です。"]
    targets = [["これは美しい花です。", "おいしい寿司を食べたい。"]]
    assert sacrebleu(preds, targets) == _reference_sacre_bleu(preds, targets, tokenize="ja-mecab", lowercase=False)


@pytest.mark.skipif(not RequirementCache("mecab-ko"))
def test_tokenize_ko_mecab():
    """Test that `ja-mecab` tokenizer works on a Japanese text in alignment with the SacreBleu implementation."""
    sacrebleu = SacreBLEUScore(tokenize="ko-mecab")

    preds = ["이 책은 정말 재미있어요."]
    targets = [["이 책은 정말 재미있어요.", "고마워요, 너무 도와줘서."]]
    assert sacrebleu(preds, targets) == _reference_sacre_bleu(preds, targets, tokenize="ko-mecab", lowercase=False)


def test_equivalence_of_available_tokenizers_and_annotation():
    """Test equivalence of SacreBLEU available tokenizers and corresponding type annotation."""
    assert set(AVAILABLE_TOKENIZERS) == set(_TokenizersLiteral.__args__)
