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
from torchmetrics.text.rouge import ROUGEMetric

PREDS = "My name is John".split()
TARGET = "Is your name John".split()


@pytest.mark.parametrize("rouge_metric, expected", [("rouge1_recall", 0.25)])
def test_rouge_metric_functional(rouge_metric, expected):
    pl_output = tensor(rouge_score(PREDS, TARGET)[rouge_metric]).float()
    assert torch.allclose(pl_output, tensor(expected), 1e-4)


@pytest.mark.parametrize("rouge_metric, expected", [("rouge1_recall", 0.25)])
def test_rouge_metric_class(rouge_metric, expected):
    rouge = ROUGEMetric()
    pl_output = tensor(rouge(PREDS, TARGET)[rouge_metric]).float()
    assert torch.allclose(pl_output, tensor(expected), 1e-4)
