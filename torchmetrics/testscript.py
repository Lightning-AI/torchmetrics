import torch
from torchmetrics import MetricCollection
from torchmetrics import Accuracy, F1, Recall, ConfusionMatrix
m = MetricCollection({
    'acc': Accuracy(num_classes=5), 
    'acc2': Accuracy(num_classes=5), 
    'acc3': Accuracy(num_classes=5, average='macro'), 
    'f1': F1(num_classes=5), 
    'recall': Recall(num_classes=5), 
    'confmat': ConfusionMatrix(num_classes=5)
})
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))
m.update(preds, target)