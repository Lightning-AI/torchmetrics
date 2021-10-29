from collections import namedtuple

import torch

from torchmetrics import MAP

Input = namedtuple("Input", ["preds", "target"])

inputs = Input(
    preds=[
        dict(
            boxes=torch.Tensor([[258., 41., 606., 285.]]),
            scores=torch.Tensor([0.536]),
            labels=torch.IntTensor([0]),
        )  # coco image id 42
    ],
    target=[
        dict(
            boxes=torch.Tensor([[214., 41., 562., 285.]]),
            labels=torch.IntTensor([0]),
        )  # coco image id 42
    ]
)

if __name__ == "__main__":
    metric = MAP(class_metrics=True)
    metric.update(inputs.preds, inputs.target)
    result = metric.compute()
    print(result)
