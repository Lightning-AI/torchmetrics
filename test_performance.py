import contextlib
import json
import time
from typing import List, Callable

import line_profiler
import torch

from torchmetrics.detection.map import MAP

limit_samples = 10
# device = "cuda:0"
device = "cpu"
# profile_functions = False
profile_functions = True


class WrappedLineProfiler(line_profiler.LineProfiler):
    """Measures time for executing code in the specified profiling_functions.
    More info: https://github.com/pyutils/line_profiler

    Call the print_stats() method after profiling to get results"""

    def __init__(self, profiling_functions: List[Callable]):
        super().__init__(*profiling_functions)

    @contextlib.contextmanager
    def __call__(self):
        self.enable()  # Start measuring time
        yield  # profiling_functions are expected to run here
        self.disable()  # Stop measuring time


profiling_functions = [
    MAP.compute,
    MAP._calculate,
    MAP._evaluate_image,
    MAP._find_best_gt_match,
]
profiler = WrappedLineProfiler(profiling_functions)

# load data
with open("test_performance_targets.json") as f:
    targets = json.load(f)

with open("test_performance_preds.json") as f:
    # with open('test_performance_targets.json') as f:
    preds = json.load(f)

# init metric
metric = MAP(class_metrics=True, box_format="xywh")
metric = metric.to(device)

# update metric
# slow but will do for the test
for sample in targets["images"][:limit_samples]:
    id = sample["id"]

    target_boxes = []
    target_labels = []
    prediction_boxes = []
    prediction_labels = []
    prediction_scores = []

    for annotation in targets["annotations"]:
        if annotation["image_id"] == id:
            target_boxes.append(annotation["bbox"])  # xywh
            target_labels.append(annotation["category_id"])

    for prediction in preds:
        if prediction["image_id"] == id:
            prediction_boxes.append(prediction["bbox"])  # xywh
            prediction_labels.append(prediction["category_id"])
            prediction_scores.append(prediction["score"])

    image_preds = [
        {
            "boxes": torch.tensor(prediction_boxes).to(device),
            "labels": torch.tensor(prediction_labels).to(device),
            "scores": torch.tensor(prediction_scores).to(device),
        }
    ]
    image_targets = [{"boxes": torch.tensor(target_boxes).to(device), "labels": torch.tensor(target_labels).to(device)}]
    metric.update(image_preds, image_targets)

# calculate metric
print(f"\nRunning metric on {len(metric.groundtruth_boxes)} samples")
start = time.time()
if profile_functions:
    with profiler():
        score = metric.compute()
    profiler.print_stats()
else:
    score = metric.compute()
end = time.time()
print(f"Total time: {end - start}")
print(f"Time per sample {(time.time() - start) / len(metric.groundtruth_boxes)}")
print("Score sanity check:\n", score)
