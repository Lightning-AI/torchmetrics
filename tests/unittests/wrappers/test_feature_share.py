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
import pytest
import torch
from torchmetrics import MetricCollection
from torchmetrics.image import FrechetInceptionDistance, InceptionScore, KernelInceptionDistance
from torchmetrics.wrappers import FeatureShare


def test_initialization():
    """Test that the feature share wrapper can be initialized."""
    fs = FeatureShare(
        [FrechetInceptionDistance(), InceptionScore(), KernelInceptionDistance()],
    )
    assert isinstance(fs, MetricCollection)
    assert len(fs) == 3


# def test_speed():
#     """Test that the feature share wrapper is faster than the metric collection."""

#     mc = MetricCollection([FrechetInceptionDistance(), InceptionScore(), KernelInceptionDistance()])

#     x = torch.randint(255, (1, 3, 64, 64), dtype=torch.uint8)

#     import time
#     start = time.time()
#     for _ in range(10):
#         x = torch.randint(255, (1, 3, 64, 64), dtype=torch.uint8)
#         mc.update(x, real=True)
#     end = time.time()
#     print(end - start)

#     start = time.time()
#     for _ in range(10):
#         x = torch.randint(255, (1, 3, 64, 64), dtype=torch.uint8)
#         fs.update(x, real=True)
#     end = time.time()
#     print(end - start)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_memory():
    """Test that the feature share wrapper uses less memory than the metric collection."""
    base_memory = torch.cuda.memory_allocated()

    fid = FrechetInceptionDistance().cuda()
    inception = InceptionScore().cuda()
    kid = KernelInceptionDistance().cuda()

    memory_before_fs = torch.cuda.memory_allocated()
    assert memory_before_fs > base_memory, "The memory usage should be higher after initializing the metrics."

    FeatureShare([fid, inception, kid], network_names=["inception", "inception", "inception"]).cuda()
    memory_after_fs = torch.cuda.memory_allocated()

    assert (
        memory_after_fs > base_memory
    ), "The memory usage should be higher after initializing the feature share wrapper."
    assert (
        memory_after_fs < memory_before_fs
    ), "The memory usage should be higher after initializing the feature share wrapper."
