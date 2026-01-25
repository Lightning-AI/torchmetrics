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
from collections.abc import Sequence
from functools import partial

import pytest
import torch
from torch import Tensor

from torchmetrics.functional.text.depth_score import depth_score
from torchmetrics.text.depth_score import DepthScore
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
from unittests._helpers import (
    _IS_WINDOWS,
    _TORCH_LESS_THAN_2_1,
    _TRANSFORMERS_GREATER_EQUAL_4_54,
    _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    skip_on_connection_issues,
)
from unittests.text._helpers import TextTester
from unittests.text._inputs import (
    _inputs_multiple_references,
    _inputs_single_reference,
    _inputs_single_sentence_multiple_references,
)

MODEL_NAME = "albert-base-v2"

# Disable tokenizers parallelism (forking not friendly with parallelism)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
def _reference_depth_score(
    preds: Sequence[str],
    target: Sequence[str],
    num_layers: int,
    depth_measure: str = "irw",
) -> Tensor:
    # Reference source code depthscore implementation
    try:
        from nlg_eval_via_simi_measures.depth_score import DepthScoreMetric
    except ImportError:
        pytest.skip("test requires `nlg_eval_via_simi_measures` to be installed.")

    metric_call = DepthScoreMetric(MODEL_NAME, layers_to_consider=num_layers, considered_measure=depth_measure)
    out = metric_call.evaluate_batch(list(target), list(preds))
    return torch.as_tensor(out["depth_score"], dtype=torch.float32)


@pytest.mark.parametrize("num_layers", [4, 8])
@pytest.mark.parametrize("depth_measure", ["irw", "ai_irw", "sliced", "wasserstein", "mmd"])
@pytest.mark.parametrize(
    ("preds", "targets"),
    [(_inputs_single_reference.preds, _inputs_single_reference.target)],
)
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
class TestDepthScore(TextTester):
    """Tests for DepthScore."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @skip_on_connection_issues()
    def test_depthscore_class(self, ddp, preds, targets, num_layers, depth_measure):
        """Test the depth score class."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "depth_measure": depth_measure,
            "device": "cpu",
            "batch_size": 8,
            "max_length": 128,
            "truncation": True,  # nlg_eval reference always truncates
        }
        reference_depth_score_metric = partial(
            _reference_depth_score,
            num_layers=num_layers,
            depth_measure=depth_measure,
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=DepthScore,
            reference_metric=reference_depth_score_metric,
            metric_args=metric_args,
            check_scriptable=False,  # huggingface transformers are not usually scriptable
            ignore_order=ddp,  # ignore order of predictions when DDP is used
        )

    @skip_on_connection_issues()
    def test_depthscore_functional(self, preds, targets, num_layers, depth_measure):
        """Test the depthscore functional."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "depth_measure": depth_measure,
            "truncation": True,  # nlg_eval reference always truncates
        }
        reference_depth_score_metric = partial(
            _reference_depth_score,
            num_layers=num_layers,
            depth_measure=depth_measure,
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=depth_score,
            reference_metric=reference_depth_score_metric,
            metric_args=metric_args,
        )

    @skip_on_connection_issues()
    def test_depthscore_differentiability(self, preds, targets, num_layers, depth_measure):
        """Test the depthscore differentiability."""
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "num_layers": num_layers,
            "depth_measure": depth_measure,
            "truncation": True,  # nlg_eval reference always truncates
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=DepthScore,
            metric_functional=depth_score,
            metric_args=metric_args,
        )


@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
def test_depthscore_sorting():
    """Test that DepthScore is invariant to the order of the inputs."""
    short = "Short text"
    long = "This is a longer text"

    preds = [long, long]
    targets = [long, short]

    metric = DepthScore(model_name_or_path=MODEL_NAME, num_layers=4, device="cpu", batch_size=2, max_length=64)
    score = metric(preds, targets)

    # First index should be the self-comparison - sorting by length should not shuffle this.
    # Distance metric: self-comparison should have a smaller distance than mismatched pair.
    assert score[0] < score[1]


@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
@pytest.mark.parametrize("truncation", [True, False])
def test_depthscore_truncation(truncation: bool):
    """Test that DepthScore truncation works as expected."""
    pred = ["abc " * 2000]
    gt = ["def " * 2000]
    metric = DepthScore(
        model_name_or_path=MODEL_NAME,
        num_layers=4,
        device="cpu",
        batch_size=1,
        max_length=64,
        truncation=truncation,
    )

    if truncation:
        res = metric(pred, gt)
        # Should produce a finite tensor (not error). Value itself is not bounded.
        assert torch.isfinite(res).all()
    else:
        with pytest.raises(RuntimeError, match="The expanded size of the tensor.*must match.*"):
            metric(pred, gt)


@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
def test_depthscore_single_str_input():
    """Test if DepthScore works with single string preds and target."""
    preds = "hello there"
    target = "hello there"

    metric = DepthScore(model_name_or_path=MODEL_NAME, num_layers=4, device="cpu", batch_size=1, max_length=64)
    score_class = metric(preds, target)

    # Distance for identical text should be smaller than for different text.
    score_class_ident = score_class.item()

    score_functional = depth_score(
        preds,
        target,
        model_name_or_path=MODEL_NAME,
        num_layers=4,
        device="cpu",
        batch_size=1,
        max_length=64,
    )
    score_func_ident = score_functional.item()

    assert score_class_ident == pytest.approx(score_func_ident, abs=1e-6)

    # Compare to a different target to assert "identical is better"
    score_diff = metric("hello there", "general kenobi").item()
    assert score_class_ident <= score_diff


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (
            _inputs_single_sentence_multiple_references.preds,
            _inputs_single_sentence_multiple_references.target,
        ),
        (
            ["hello there", "I'm in the middle", "general kenobi"],
            (["hello there", "master kenobi"], "I'm here", ("hello there", "master kenobi")),
        ),
    ],
)
@skip_on_connection_issues()
@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
@pytest.mark.xfail(
    RuntimeError,
    condition=_TORCH_LESS_THAN_2_1 and _TRANSFORMERS_RANGE_GE_4_50_LT_4_54,
    reason="could be due to torch compatibility issues with transformers",
)
@pytest.mark.xfail(
    ImportError,
    condition=_TORCH_LESS_THAN_2_1 and _IS_WINDOWS and _TRANSFORMERS_GREATER_EQUAL_4_54,
    reason="another strange behaviour of transformers on windows",
)
def test_depthscore_multiple_references(preds, target):
    """Test both functional and class APIs with multiple references."""
    # Functional returns a 1D tensor; class returns dict with "depth_score"
    result_func = depth_score(preds, target)
    metric = DepthScore()
    result_class = metric(preds, target)

    # They should match exactly (same code path), and output should be per-pred after reduction (min across refs).
    assert torch.allclose(result_func, result_class, atol=1e-6)

    # Sanity: output length equals number of predictions (not flattened refs)
    if isinstance(preds, str):
        assert result_func.numel() == 1
    else:
        assert result_func.numel() == len(preds)


@pytest.mark.skipif(not _TRANSFORMERS_GREATER_EQUAL_4_4, reason="test requires transformers>4.4")
def test_depthscore_invalid_references():
    """Test both functional and class APIs with invalid references."""
    preds = _inputs_multiple_references.preds
    target = _inputs_multiple_references.target

    with pytest.raises(ValueError, match="Invalid input provided."):
        depth_score(preds, target)

    metric = DepthScore()
    with pytest.raises(ValueError, match="Invalid input provided."):
        metric(preds, target)
