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
    expected_results = {"init": 1.5171937870327383, "update": 0.11278650932945311, "compute": 146.6234815029893}
    results = run_benchmark()
    for name, time in results.items():
        assert np.isclose(expected_results[name], results[name], rtol=1.0)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_performance_with_cpu():
    expected_results = {"init": 0.025219694012776017, "update": 0.11010122182779014, "compute": 77.43531435285695}
    results = run_benchmark(device="cpu")
    for name, time in results.items():
        assert np.isclose(expected_results[name], results[name], rtol=1.0)
