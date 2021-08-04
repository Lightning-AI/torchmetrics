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

import pytest
import torch

from tests.helpers import seed_all
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, Precision, Recall
from torchmetrics.wrappers import MetricTracker

seed_all(42)


def test_raises_error_on_wrong_input():
    with pytest.raises(TypeError, match="metric arg need to be an instance of a torchmetrics metric .*"):
        MetricTracker([1, 2, 3])


@pytest.mark.parametrize(
    "method, method_input",
    [
        ("update", (torch.randint(10, (50,)), torch.randint(10, (50,)))),
        ("forward", (torch.randint(10, (50,)), torch.randint(10, (50,)))),
        ("compute", None),
    ],
)
def test_raises_error_if_increment_not_called(method, method_input):
    with pytest.raises(ValueError, match=f"`{method}` cannot be called before .*"):
        tracker = MetricTracker(Accuracy(num_classes=10))
        if method_input is not None:
            getattr(tracker, method)(*method_input)
        else:
            getattr(tracker, method)()


@pytest.mark.parametrize(
    "base_metric, metric_input, maximize",
    [
        (partial(Accuracy, num_classes=10), (torch.randint(10, (50,)), torch.randint(10, (50,))), True),
        (partial(Precision, num_classes=10), (torch.randint(10, (50,)), torch.randint(10, (50,))), True),
        (partial(Recall, num_classes=10), (torch.randint(10, (50,)), torch.randint(10, (50,))), True),
        (
            MeanSquaredError,
            (
                torch.randn(50),
                torch.randn(50),
            ),
            False,
        ),
        (
            MeanAbsoluteError,
            (
                torch.randn(50),
                torch.randn(50),
            ),
            False,
        ),
    ],
)
def test_tracker(base_metric, metric_input, maximize):
    tracker = MetricTracker(base_metric(), maximize=maximize)
    for i in range(5):
        tracker.increment()
        # check both update and forward works
        for _ in range(5):
            tracker.update(*metric_input)
        for _ in range(5):
            tracker(*metric_input)

        val = tracker.compute()
        assert val != 0.0
        assert tracker.n_steps == i + 1

    assert tracker.n_steps == 5
    assert tracker.compute_all().shape[0] == 5
    val, idx = tracker.best_metric(return_step=True)
    assert val != 0.0
    assert idx in list(range(5))
