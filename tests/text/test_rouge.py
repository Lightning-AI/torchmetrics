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

import pytest
import torch
from torch import tensor

from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE

PREDS = "My name is John".split()
TARGET = "Is your name John".split()


@pytest.mark.skipif(not (_NLTK_AVAILABLE or _ROUGE_SCORE_AVAILABLE), reason='test requires nltk and rouge-score')
@pytest.mark.parametrize("rouge_metric, expected", [("rouge1_recall", 0.25)])
def test_rouge_metric_functional(rouge_metric, expected):
    pl_output = tensor(rouge_score(PREDS, TARGET)[rouge_metric]).float()
    assert torch.allclose(pl_output, tensor(expected), 1e-4)


@pytest.mark.skipif(not (_NLTK_AVAILABLE or _ROUGE_SCORE_AVAILABLE), reason='test requires nltk and rouge-score')
@pytest.mark.parametrize("rouge_metric, expected", [("rouge1_recall", 0.25)])
def test_rouge_metric_class(rouge_metric, expected):
    rouge = ROUGEScore()
    pl_output = tensor(rouge(PREDS, TARGET)[rouge_metric]).float()
    assert torch.allclose(pl_output, tensor(expected), 1e-4)


def test_rouge_metric_raises_errors_and_warnings():
    """ Test that expected warnings and errors are raised """
    if not (_NLTK_AVAILABLE or _ROUGE_SCORE_AVAILABLE):
        with pytest.raises(
            ValueError,
            match='ROUGE metric requires that both nltk and rouge-score is installed.'
            'Either as `pip install torchmetrics[text]`'
            ' or `pip install nltk rouge-score`'
        ):
            ROUGEScore()
