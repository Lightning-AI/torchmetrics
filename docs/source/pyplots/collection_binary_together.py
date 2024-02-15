import matplotlib.pyplot as plt
import torch
import torchmetrics

N = 10
num_updates = 10
num_steps = 5

w = torch.tensor([0.2, 0.8])
target = lambda it: torch.multinomial((it * w).softmax(dim=-1), 100, replacement=True)
preds = lambda it: torch.multinomial((it * w).softmax(dim=-1), 100, replacement=True)

collection = torchmetrics.MetricCollection(
    torchmetrics.Accuracy(task="binary"),
    torchmetrics.Recall(task="binary"),
    torchmetrics.Precision(task="binary"),
)

values = []
fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=500)
for step in range(num_steps):
    for _ in range(N):
        collection.update(preds(step), target(step))
    values.append(collection.compute())
    collection.reset()
collection.plot(val=values, ax=ax, together=True)
fig.tight_layout()
fig.show()
