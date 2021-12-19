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

from functools import partial
from typing import Sequence

import pytest
from pytest_cases import parametrize_with_cases
from torch import Tensor, tensor

from tests.helpers.testers import MetricTesterDDPCases
from tests.text.helpers import INPUT_ORDER, TextTester
from tests.text.inputs import _inputs_multiple_references
from torchmetrics.functional.text.sacre_bleu import sacre_bleu_score
from torchmetrics.text.sacre_bleu import SacreBLEUScore
from torchmetrics.utilities.imports import _SACREBLEU_AVAILABLE

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import BLEU


TOKENIZERS = ("none", "13a", "zh", "intl", "char")


def sacrebleu_fn(targets: Sequence[Sequence[str]], preds: Sequence[str], tokenize: str, lowercase: bool) -> Tensor:
    sacrebleu_fn = BLEU(tokenize=tokenize, lowercase=lowercase)
    # Sacrebleu expects different format of input
    targets = [[target[i] for target in targets] for i in range(len(targets[0]))]
    sacrebleu_score = sacrebleu_fn.corpus_score(preds, targets).score / 100
    return tensor(sacrebleu_score)


@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_multiple_references.preds, _inputs_multiple_references.targets)],
)
@pytest.mark.parametrize(["lowercase"], [(False,), (True,)])
@pytest.mark.parametrize("tokenize", TOKENIZERS)
@pytest.mark.skipif(not _SACREBLEU_AVAILABLE, reason="test requires sacrebleu")
class TestSacreBLEUScore(TextTester):
    @parametrize_with_cases("ddp,device", cases=MetricTesterDDPCases, has_tag="strategy")
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_bleu_score_class(self, ddp, dist_sync_on_step, preds, targets, tokenize, lowercase, device):
        metric_args = {"tokenize": tokenize, "lowercase": lowercase}
        original_sacrebleu = partial(sacrebleu_fn, tokenize=tokenize, lowercase=lowercase)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=SacreBLEUScore,
            sk_metric=original_sacrebleu,
            dist_sync_on_step=dist_sync_on_step,
            device=device,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    @parametrize_with_cases("device", cases=MetricTesterDDPCases, has_tag="device")
    def test_bleu_score_functional(self, preds, targets, tokenize, lowercase, device):
        metric_args = {"tokenize": tokenize, "lowercase": lowercase}
        original_sacrebleu = partial(sacrebleu_fn, tokenize=tokenize, lowercase=lowercase)

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=sacre_bleu_score,
            sk_metric=original_sacrebleu,
            device=device,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_bleu_score_differentiability(self, preds, targets, tokenize, lowercase):
        metric_args = {"tokenize": tokenize, "lowercase": lowercase}

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=SacreBLEUScore,
            metric_functional=sacre_bleu_score,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )
