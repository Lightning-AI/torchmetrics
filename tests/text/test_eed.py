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
from torch import tensor

from torchmetrics.functional.text.eed import eed
from torchmetrics.text.eed import EED

REFERENCES_1 = "perfect match"
HYPOTHESES_1 = "perfect match"

REFERENCES_2 = "blue orange"
HYPOTHESES_2 = "imperfect match"

# test batches
BATCH_1 = {"references": [REFERENCES_1, REFERENCES_2], "hypotheses": [HYPOTHESES_1, HYPOTHESES_2]}


# test blank edge cases
def test_eed_empty_functional():
    ref = [[]]
    hyp = []
    assert eed(ref, hyp) == tensor(0.0)


def test_eed_empty_class():
    eed_metric = EED()
    ref = [[]]
    hyp = []
    assert eed_metric(ref, hyp) == tensor(0.0)


def test_eed_empty_with_non_empty_hyp_functional():
    ref = [[]]
    hyp = ["python"]
    assert eed(ref, hyp) == tensor(0.0)


def test_eed_empty_with_non_empty_hyp_class():
    eed_metric = EED()
    ref = [[]]
    hyp = ["python"]
    assert eed_metric(ref, hyp) == tensor(0.0)


# test ability to parallelize
def test_parallelisation_eed():
    references = BATCH_1["references"]
    hypotheses = BATCH_1["hypotheses"]

    # batch_size == length of data
    metric = EED()

    sequential_score = metric(references, hypotheses)

    # batch of 1 with compute_on_step == False
    metric = EED(compute_on_step=False)

    for reference, hypothesis in zip(references, hypotheses):
        metric([reference], [hypothesis])

    parallel_score = metric.compute()

    score_comparison = bool(np.isclose(sequential_score, parallel_score))

    assert bool(score_comparison) is True
