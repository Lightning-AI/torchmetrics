import time
from typing import Dict, Union

import torch

from torchmetrics.detection import MeanAveragePrecision
from unittests.helpers import seed_all

seed_all(42)
total_time = {}


class UpdateTime:
    def __init__(self, step_name: str):
        self._step_name = step_name

    def __enter__(self):
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        if self._step_name in total_time:
            total_time[self._step_name] += end_time - self._start_time
        else:
            total_time[self._step_name] = end_time - self._start_time
        return True


def generate(n, factor: int = 1000):
    boxes = torch.rand(n, 4) * factor
    boxes[:, 2:] += boxes[:, :2]
    labels = torch.randint(0, 10, (n,))
    scores = torch.rand(n)
    return {"boxes": boxes, "labels": labels, "scores": scores}


def run_mean_ap_benchmark(device: Union[str, int] = "cuda") -> Dict[str, float]:
    mean_ap = MeanAveragePrecision()
    mean_ap.to(device=torch.device(device))

    for batch_idx in range(64):
        detections = [generate(100, 10) for _ in range(10)]
        targets = [generate(10, 10) for _ in range(10)]
        mean_ap.update(detections, targets)

    mean_ap_results = mean_ap.compute()
    mean_ap.reset()
    return mean_ap_results


def run_speed_benchmark(device: Union[str, int] = "cuda") -> Dict[str, float]:
    with UpdateTime("init"):
        mean_ap = MeanAveragePrecision()
        mean_ap.to(device=torch.device(device))

    for batch_idx in range(100):
        with UpdateTime("update"):
            detections = [generate(100) for _ in range(10)]
            targets = [generate(10) for _ in range(10)]
            mean_ap.update(detections, targets)

    with UpdateTime("compute"):
        try:
            _ = mean_ap.compute()
            mean_ap.reset()
        except Exception as e:
            print(f"Error occurred when running compute -> {e}")

    return total_time


if __name__ == "__main__":
    results = run_mean_ap_benchmark(device="cpu")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value}")
    print("\\\\\\\\\\\\//////")
    results = run_speed_benchmark(device="cpu")
    for step, step_time in results.items():
        print(f"Total time in {step}: {step_time}")
