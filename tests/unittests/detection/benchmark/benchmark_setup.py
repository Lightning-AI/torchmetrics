import time

import torch
from torchmetrics.detection import MeanAveragePrecision

total_time = {}


class UpdateTime:
    def __init__(self, name) -> None:
        self._name = name

    def __enter__(self):
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        if self._name in total_time:
            total_time[self._name] += end_time - self._start_time
        else:
            total_time[self._name] = end_time - self._start_time
        return True


def generate(n):
    boxes = torch.rand(n, 4) * 1000
    boxes[:, 2:] += boxes[:, :2]
    labels = torch.randint(0, 10, (n,))
    scores = torch.rand(n)
    return {"boxes": boxes, "labels": labels, "scores": scores}


def run_benchmark(device: str = "cuda") -> dict[str, float]:
    with UpdateTime("init"):
        mean_ap = MeanAveragePrecision()
        mean_ap.to(device=torch.device(device))

    for _batch_idx in range(100):
        with UpdateTime("update"):
            detections = [generate(100) for _ in range(10)]
            targets = [generate(10) for _ in range(10)]
            mean_ap.update(detections, targets)

    with UpdateTime("compute"):
        try:
            mean_ap.compute()
        except Exception as e:
            print(f"Error occurred when running compute -> {e}")

    return total_time
