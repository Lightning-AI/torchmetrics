import pytest
import torch
import numpy as np

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from unittests.detection.benchmark.benchmark_setup import run_benchmark

_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)
_gpu_test_condition = not torch.cuda.is_available()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.skipif(_gpu_test_condition, reason="test requires CUDA availability")
def test_performance_with_cuda():
    expected_results = {"init": 1.0296829228755087, "update": 0.07939263735897839, "compute": 2.355824921047315}
    results = run_benchmark()
    for name, time in results.items():
        assert np.isclose(expected_results[name], results[name], rtol=1.0)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_performance_with_cuda():
    expected_results = {"init": 0.02504527010023594, "update": 0.11929452884942293, "compute": 1.9458624629769474}
    results = run_benchmark()
    for name, time in results.items():
        assert np.isclose(expected_results[name], results[name], rtol=1.0)
