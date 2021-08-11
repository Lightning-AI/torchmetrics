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

from typing import List

import pytest
import torch
from torch import tensor

from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE


ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")

SINGLE_SENTENCE_EXAMPLE_PREDS = "The quick brown fox jumps over the lazy dog"
SINGLE_SENTENCE_EXAMPLE_TARGET = "The quick brown dog jumps on the log."

SINGLE_SENTENCE_EXAMPLE_RESULTS = {
    "rouge1_precision": torch.tensor(0.6666666666666666),
    "rouge1_recall": torch.tensor(0.75),
    "rouge1_fmeasure": torch.tensor(0.7058823529411765),
    "rouge2_precision": torch.tensor(0.25),
    "rouge2_recall": torch.tensor(0.2857142857142857),
    "rouge2_fmeasure": torch.tensor(0.26666666666666666),
    "rougeL_precision": torch.tensor(0.5555555555555556),
    "rougeL_recall": torch.tensor(0.625),
    "rougeL_fmeasure": torch.tensor(0.5882352941176471),
    "rougeLsum_precision": torch.tensor(0.5555555555555556),
    "rougeLsum_recall": torch.tensor(0.625),
    "rougeLsum_fmeasure": torch.tensor(0.5882352941176471),
}

PREDS = "My name is John".split()
TARGETS = "Is your name John".split()

PREDS_SPLIT_RESULTS = {
    "rouge1_precision": torch.tensor(0.25),
    "rouge1_recall": torch.tensor(0.25),
    "rouge1_fmeasure": torch.tensor(0.25),
    "rouge2_precision": torch.tensor(0.0),
    "rouge2_recall": torch.tensor(0.0),
    "rouge2_fmeasure": torch.tensor(0.0),
    "rougeL_precision": torch.tensor(0.25),
    "rougeL_recall": torch.tensor(0.25),
    "rougeL_fmeasure": torch.tensor(0.25),
    "rougeLsum_precision": torch.tensor(0.25),
    "rougeLsum_recall": torch.tensor(0.25),
    "rougeLsum_fmeasure": torch.tensor(0.25),
}

BATCHES_RS_PREDS = [SINGLE_SENTENCE_EXAMPLE_PREDS]
BATCHES_RS_PREDS.extend(PREDS)
BATCHES_RS_TARGETS = [SINGLE_SENTENCE_EXAMPLE_TARGET]
BATCHES_RS_TARGETS.extend(TARGETS)

BATCHES = [
    dict(preds=[SINGLE_SENTENCE_EXAMPLE_PREDS], targets=[SINGLE_SENTENCE_EXAMPLE_TARGET]),
    dict(preds=PREDS, targets=TARGETS),
]

BATCHES_RESULTS = [SINGLE_SENTENCE_EXAMPLE_RESULTS, PREDS_SPLIT_RESULTS]


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", True, True),
        pytest.param("rouge1_recall", True, False),
        pytest.param("rouge1_fmeasure", False, True),
        pytest.param("rouge2_precision", False, False),
        pytest.param("rouge2_recall", True, True),
        pytest.param("rouge2_fmeasure", True, False),
        pytest.param("rougeL_precision", False, True),
        pytest.param("rougeL_recall", False, False),
        pytest.param("rougeL_fmeasure", True, True),
        pytest.param("rougeLsum_precision", True, False),
        pytest.param("rougeLsum_recall", False, True),
        pytest.param("rougeLsum_fmeasure", False, False),
    ],
)
def test_rouge_metric_functional_single_sentence(pl_rouge_metric_key, use_stemmer, newline_sep):
    pl_output = rouge_score(
        [SINGLE_SENTENCE_EXAMPLE_PREDS],
        [SINGLE_SENTENCE_EXAMPLE_TARGET],
        newline_sep=newline_sep,
        use_stemmer=use_stemmer,
    )

    assert torch.allclose(pl_output[pl_rouge_metric_key], SINGLE_SENTENCE_EXAMPLE_RESULTS[pl_rouge_metric_key])


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", True, True),
        pytest.param("rouge1_recall", True, False),
        pytest.param("rouge1_fmeasure", False, True),
        pytest.param("rouge2_precision", False, False),
        pytest.param("rouge2_recall", True, True),
        pytest.param("rouge2_fmeasure", True, False),
        pytest.param("rougeL_precision", False, True),
        pytest.param("rougeL_recall", False, False),
        pytest.param("rougeL_fmeasure", True, True),
        pytest.param("rougeLsum_precision", True, False),
        pytest.param("rougeLsum_recall", False, True),
        pytest.param("rougeLsum_fmeasure", False, False),
    ],
)
def test_rouge_metric_functional(pl_rouge_metric_key, use_stemmer, newline_sep):
    pl_output = rouge_score(PREDS, TARGETS, newline_sep=newline_sep, use_stemmer=use_stemmer)

    assert torch.allclose(pl_output[pl_rouge_metric_key], PREDS_SPLIT_RESULTS[pl_rouge_metric_key])


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", True, True),
        pytest.param("rouge1_recall", True, False),
        pytest.param("rouge1_fmeasure", False, True),
        pytest.param("rouge2_precision", False, False),
        pytest.param("rouge2_recall", True, True),
        pytest.param("rouge2_fmeasure", True, False),
        pytest.param("rougeL_precision", False, True),
        pytest.param("rougeL_recall", False, False),
        pytest.param("rougeL_fmeasure", True, True),
        pytest.param("rougeLsum_precision", True, False),
        pytest.param("rougeLsum_recall", False, True),
        pytest.param("rougeLsum_fmeasure", False, False),
    ],
)
def test_rouge_metric_class(pl_rouge_metric_key, use_stemmer, newline_sep):
    rouge = ROUGEScore(newline_sep=newline_sep, use_stemmer=use_stemmer)
    pl_output = rouge([SINGLE_SENTENCE_EXAMPLE_PREDS], [SINGLE_SENTENCE_EXAMPLE_TARGET])

    assert torch.allclose(pl_output[pl_rouge_metric_key], SINGLE_SENTENCE_EXAMPLE_RESULTS[pl_rouge_metric_key])


@pytest.mark.skipif(not _NLTK_AVAILABLE, reason="test requires nltk")
@pytest.mark.parametrize(
    ["pl_rouge_metric_key", "use_stemmer", "newline_sep"],
    [
        pytest.param("rouge1_precision", True, True),
        pytest.param("rouge1_recall", True, False),
        pytest.param("rouge1_fmeasure", False, True),
        pytest.param("rouge2_precision", False, False),
        pytest.param("rouge2_recall", True, True),
        pytest.param("rouge2_fmeasure", True, False),
        pytest.param("rougeL_precision", False, True),
        pytest.param("rougeL_recall", False, False),
        pytest.param("rougeL_fmeasure", True, True),
        pytest.param("rougeLsum_precision", True, False),
        pytest.param("rougeLsum_recall", False, True),
        pytest.param("rougeLsum_fmeasure", False, False),
    ],
)
def test_rouge_metric_class_batches(pl_rouge_metric_key, use_stemmer, newline_sep):
    rouge = ROUGEScore(newline_sep=newline_sep, use_stemmer=use_stemmer)
    for batch, results in zip(BATCHES, BATCHES_RESULTS):
        rouge.update(batch["preds"], batch["targets"])
    pl_output = rouge.compute()

    assert torch.allclose(pl_output[pl_rouge_metric_key], results[pl_rouge_metric_key])


def test_rouge_metric_raises_errors_and_warnings():
    """Test that expected warnings and errors are raised."""
    if not _NLTK_AVAILABLE:
        with pytest.raises(
            ValueError,
            match="ROUGE metric requires that nltk is installed."
            "Either as `pip install torchmetrics[text]` or `pip install nltk`",
        ):
            ROUGEScore()


def test_rouge_metric_wrong_key_value_error():
    key = ("rouge1", "rouge")

    with pytest.raises(ValueError):
        ROUGEScore(rouge_keys=key)

    with pytest.raises(ValueError):
        rouge_score(PREDS, TARGETS, rouge_keys=key)
