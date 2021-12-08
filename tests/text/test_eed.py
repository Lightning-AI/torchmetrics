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

import numpy as np
import pytest

from torchmetrics.functional.text.eed import eed
from torchmetrics.text.eed import EED

HYPOTHESES_1 = "perfect match"
REFERENCES_1 = "perfect match"

HYPOTHESES_2 = "imperfect match"
REFERENCES_2 = "blue orange"

# test batches
BATCH_1 = {"hypotheses": [HYPOTHESES_1, HYPOTHESES_2], "references": [REFERENCES_1, REFERENCES_2]}


# test blank edge cases
def test_eed_empty_functional():
    with pytest.raises(ValueError):
        hyp = []
        ref = [[]]
        eed(ref, hyp)


def test_eed_empty_class():
    with pytest.raises(ValueError):
        eed_metric = EED()
        hyp = []
        ref = [[]]
        eed_metric(ref, hyp)


def test_eed_empty_with_non_empty_hyp_functional():
    with pytest.raises(ValueError):
        hyp = ["python"]
        ref = [[]]
        eed(ref, hyp)


def test_eed_empty_with_non_empty_hyp_class():
    with pytest.raises(ValueError):
        eed_metric = EED()
        hyp = ["python"]
        ref = [[]]
        eed_metric(ref, hyp)


# test ability to parallelize
def test_parallelisation_eed():
    hypotheses = BATCH_1["hypotheses"]
    references = BATCH_1["references"]

    # batch_size == length of data
    metric = EED()

    sequential_score = metric(hypotheses, references)

    # batch of 1 with compute_on_step == False
    metric = EED(compute_on_step=False)

    for hypothesis, reference in zip(hypotheses, references):
        metric(hypothesis, reference)

    parallel_score = metric.compute()

    assert sequential_score == parallel_score
