from torchmetrics.functional import confusion_matrix, multiclass_confusion_matrix, stat_scores, multiclass_stat_scores
from time import perf_counter
import torch
from functools import partial
from statistics import mean, stdev

OUTER_REPS = 5
INNER_REPS = 1000

preds = torch.randn(100, 10).softmax(dim=-1)
target = torch.randint(10, (100,))


def time(metric_fn, name, base=None):
    timings = []
    for _ in range(OUTER_REPS):
        start = perf_counter()
        for _ in range(INNER_REPS):
            metric_fn(preds, target)
        end = perf_counter()
        timings.append(end - start)
    extra = f", speedup: {base / mean(timings)}" if base is not None else ""
    print(f"{name.ljust(15)}: {mean(timings):0.3E} +- {stdev(timings):0.3E}{extra}")
    return mean(timings)


print(f"\nExperiments running {INNER_REPS} calculations, repeting {OUTER_REPS} times:")
print("\nMulticlass Confusion matrix")
base = time(partial(confusion_matrix, num_classes=10), name="Old")
time(partial(multiclass_confusion_matrix, num_classes=10), name="New with IV", base=base)
time(partial(multiclass_confusion_matrix, num_classes=10, validate_args=False), name="New without IV", base=base)

print("\nMulticlass Stat Scores")
base = time(partial(stat_scores, num_classes=10), name="Old")
time(partial(multiclass_stat_scores, num_classes=10), name="New with IV", base=base)
time(partial(multiclass_stat_scores, num_classes=10, validate_args=False), name="New without IV", base=base)
