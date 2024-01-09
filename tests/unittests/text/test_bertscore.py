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
import os
from functools import partial
from typing import Sequence

import pytest
from torch import Tensor
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.text.bert import BERTScore
from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
from typing_extensions import Literal

from unittests.text.helpers import TextTester, skip_on_connection_issues
from unittests.text.inputs import _inputs_single_reference

if _BERTSCORE_AVAILABLE:
    from bert_score import score as original_bert_score
else:
    original_bert_score = None

_METRIC_KEY_TO_IDX = {
    "precision": 0,
    "recall": 1,
    "f1": 2,
}

MODEL_NAME = "albert-base-v2"

# Disable tokenizers parallelism (forking not friendly with parallelism)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
@skip_on_connection_issues()
def _reference_bert_score(
    preds: Sequence[str],
    target: Sequence[str],
    num_layers: int,
    all_layers: bool,
    idf: bool,
    rescale_with_baseline: bool,
    metric_key: Literal["f1", "precision", "recall"],
) -> Tensor:
    score_tuple = original_bert_score(
        preds,
        target,
        model_type=MODEL_NAME,
        lang="en",
        num_layers=num_layers,
        all_layers=all_layers,
        idf=idf,
        batch_size=len(preds),
        rescale_with_baseline=rescale_with_baseline,
        nthreads=0,
    )
    return score_tuple[_METRIC_KEY_TO_IDX[metric_key]]


@pytest.mark.parametrize(
    ["num_layers", "all_layers", "idf", "rescale_with_baseline", "metric_key"],
    [
        (8, False, False, False, "precision"),
        (12, True, False, False, "recall"),
        (12, False, True, False, "f1"),
        (8, False, False, True, "precision"),
        (12, True, True, False, "recall"),
        (12, True, False, True, "f1"),
        (8, False, True, True, "precision"),
        (12, True, True, True, "f1"),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_single_reference.preds, _inputs_single_reference.target)],
)
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.skipif(not _BERTSCORE_AVAILABLE, reason="test requires bert_score")
class TestBERTScore(TextTester):
    """Tests for BERTScore."""

    @pytest.mark.parametrize("ddp", [False, True])
    @skip_on_connection_issues()
    def test_bertscore_class(self, ddp, preds, targets, num_layers, all_layers, idf, rescale_with_baseline, metric_key):
        """Test the bertscore class."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }
        reference_bert_score_metric = partial(
            _reference_bert_score,
            num_layers=num_layers,
            all_layers=all_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            metric_key=metric_key,
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=BERTScore,
            reference_metric=reference_bert_score_metric,
            metric_args=metric_args,
            key=metric_key,
            check_scriptable=False,  # huggingface transformers are not usually scriptable
            ignore_order=ddp,  # ignore order of predictions when DDP is used
        )

    @skip_on_connection_issues()
    def test_bertscore_functional(self, preds, targets, num_layers, all_layers, idf, rescale_with_baseline, metric_key):
        """Test the bertscore functional."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }
        reference_bert_score_metric = partial(
            _reference_bert_score,
            num_layers=num_layers,
            all_layers=all_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            metric_key=metric_key,
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=bert_score,
            reference_metric=reference_bert_score_metric,
            metric_args=metric_args,
            key=metric_key,
        )

    def test_bertscore_differentiability(
        self, preds, targets, num_layers, all_layers, idf, rescale_with_baseline, metric_key
    ):
        """Test the bertscore differentiability."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "all_layers": all_layers,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=BERTScore,
            metric_functional=bert_score,
            metric_args=metric_args,
            key=metric_key,
        )
