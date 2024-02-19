import matplotlib.pyplot as plt
import torch
import torchmetrics

N = 10
num_updates = 10
num_steps = 5

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=500)

metric = torchmetrics.Accuracy(task="multiclass", num_classes=3, average=None)
for _ in range(N):
    metric.update(torch.randint(3, (10,)), torch.randint(3, (10,)))
metric.plot(ax=ax)
fig.show()
