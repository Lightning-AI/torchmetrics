import matplotlib.pyplot as plt
import torch

import torchmetrics

N = 10
num_updates = 10
num_steps = 5
dpi = 200

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=dpi)
metric = torchmetrics.Accuracy(task="binary")
for _ in range(N):
    metric.update(torch.rand(10), torch.randint(2, (10,)))
metric.plot(ax=ax)
fig.savefig("docs/blogs/v1.0.0_plot_1.png")

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=dpi)
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3, average=None)
for _ in range(N):
    metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
metric.plot(ax=ax)
fig.savefig("docs/blogs/v1.0.0_plot_2.png")

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=dpi)
w = torch.tensor([0.2, 0.8])
target = lambda it: torch.multinomial((it * w).softmax(dim=-1), 100, replacement=True)
preds = lambda it: torch.multinomial((it * w).softmax(dim=-1), 100, replacement=True)
metric = torchmetrics.Accuracy(task="binary")
values = []
for step in range(num_steps):
    for _ in range(N):
        metric.update(preds(step), target(step))
    values.append(metric.compute())  # save value
    metric.reset()
metric.plot(values, ax=ax)
fig.savefig("docs/blogs/v1.0.0_plot_3.png")

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=dpi)
metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3)
for _ in range(N):
    metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
metric.plot(ax=ax)
fig.savefig("docs/blogs/v1.0.0_plot_4.png")

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=dpi)
metric = torchmetrics.ROC(task="binary")
for _ in range(N):
    metric.update(torch.rand(10), torch.randint(2, (10,)))
metric.plot(ax=ax)
fig.savefig("docs/blogs/v1.0.0_plot_5.png")


plt.show()