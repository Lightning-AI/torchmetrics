import matplotlib.pyplot as plt
import torch

import torchmetrics

N = 10
num_updates = 10
num_steps = 5

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=500)
metric = torchmetrics.Accuracy(task="binary")
for _ in range(N):
    metric.update(torch.rand(10), torch.randint(2, (10,)))
metric.plot(ax=ax)
fig.show()
