import torch

from torchmetrics import MAP

preds = [
    dict(
        boxes=torch.Tensor([[258.0, 41.0, 606.0, 285.0]]),
        scores=torch.Tensor([0.536]),
        labels=torch.IntTensor([0]),
    )
]

target = [
    dict(
        boxes=torch.Tensor([[214.0, 41.0, 562.0, 285.0]]),
        labels=torch.IntTensor([0]),
    )
]

if __name__ == "__main__":
    metric = MAP()
    metric.update(preds, target)
    result = metric.compute()
    print(result["map"].cpu().item())
