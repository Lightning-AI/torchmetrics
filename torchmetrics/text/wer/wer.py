from typing import Any

from jiwer import compute_measures

from torchmetrics.metric import Metric


class WER(Metric):

    def __init__(self, concatenate_texts=False):
        super().__init__()
        self.concatenate_texts = concatenate_texts

    def update(self, preds: Any, target: Any) -> None:
        self.preds = preds
        self.target = target

    def compute(self) -> Any:
        if self.concatenate_texts:
            return compute_measures(self.target, self.preds)["wer"]
        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(self.preds, self.target):
                measures = compute_measures(reference, prediction)
                incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
                total += measures["substitutions"] + measures["deletions"] + measures["hits"]
            return incorrect / total
