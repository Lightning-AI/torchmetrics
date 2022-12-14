import pytest
import torch
import numpy as np

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from unittests.benchmark.benchmark_setup import run_mean_ap_benchmark, run_speed_benchmark

_pytest_condition = not (_TORCHVISION_AVAILABLE and _TORCHVISION_GREATER_EQUAL_0_8)
_gpu_test_condition = not torch.cuda.is_available()


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
@pytest.mark.skipif(_gpu_test_condition, reason="test requires CUDA availability")
def test_performance_with_cuda():
    """Test performance on CUDA device."""
    results = run_speed_benchmark(device=0)
    expected_results = {"init": 0.0011681069154292345, "update": 0.11278650932945311, "compute": 146.6234815029893}
    for step_name, step_time in results.items():
        print(f"Total time in {step_name}: {step_time}")
        assert np.isclose(expected_results[step_name], results[step_name], atol=8.0e-1, rtol=8.0e-1)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_performance_with_cpu():
    """Test performance on CPU device."""
    results = run_speed_benchmark(device="cpu")
    expected_results = {"init": 0.0011681069154292345, "update": 0.11010122182779014, "compute": 77.43531435285695}
    for step_name, step_time in results.items():
        print(f"Total time in {step_name}: {step_time}")
        assert np.isclose(expected_results[step_name], results[step_name], atol=8.0e-1, rtol=8.0e-1)


@pytest.mark.skipif(_pytest_condition, reason="test requires that torchvision=>0.8.0 is installed")
def test_test_consecutive_runs_with_cpu():
    """Test consistency on CPU device."""
    restuls_1 = run_mean_ap_benchmark(device="cpu")
    restuls_2 = run_mean_ap_benchmark(device="cpu")

    for key in restuls_1:
        assert np.isclose(restuls_1[key], restuls_2[key], atol=8.0e-1, rtol=8.0e-1)
