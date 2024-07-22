import matplotlib.pyplot as plt
import torch
import torchmetrics

N = 10
num_updates = 10
num_steps = 5

w = torch.tensor([0.2, 0.8])
target = lambda it: torch.multinomial((it * w).softmax(dim=-1), 100, replacement=True)
preds = lambda it: (it * torch.randn(100)).sigmoid()

confmat = torchmetrics.ConfusionMatrix(task="binary")
roc = torchmetrics.ROC(task="binary")
tracker = torchmetrics.wrappers.MetricTracker(
    torchmetrics.MetricCollection(
        torchmetrics.Accuracy(task="binary"),
        torchmetrics.Recall(task="binary"),
        torchmetrics.Precision(task="binary"),
        confmat,
        roc,
    )
)

fig = plt.figure(layout="constrained", figsize=(6.8, 4.8), dpi=500)
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, (3, 4))

for step in range(num_steps):
    tracker.increment()
    for _ in range(N):
        tracker.update(preds(step), target(step))

# get the results from all steps and extract for confusion matrix and roc
all_results = tracker.compute_all()
confmat.plot(val=all_results[-1]["BinaryConfusionMatrix"], ax=ax1)
roc.plot(all_results[-1]["BinaryROC"], ax=ax2)

scalar_results = [{k: v for k, v in ar.items() if isinstance(v, torch.Tensor) and v.numel() == 1} for ar in all_results]

tracker.plot(val=scalar_results, ax=ax3)
fig.show()
