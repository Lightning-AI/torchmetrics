from typing import Any

from jiwer import compute_measures

from torchmetrics.metric import Metric


class WER(Metric):
    """
    Word error rate (WER) is a common metric of the performance of an automatic speech recognition system.

    WER's output is always a number between 0 and 1.
    This value indicates the percentage of words that were incorrectly predicted.
    The lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.

    Word error rate can then be computed as:
    WER = (S + D + I) / N = (S + D + I) / (S + D + C)
    where
    S is the number of substitutions,
    D is the number of deletions,
    I is the number of insertions,
    C is the number of correct words,
    N is the number of words in the reference (N=S+D+C).

    Compute WER score of transcribed segments against references.

    Args:
        references: List of references for each speech input.
        predictions: List of transcriptions to score.
        concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute WER iteratively.
    Returns:
        (float): the word error rate
    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> wer = WER(predictions=predictions, references=references)
        >>> wer_score = wer.compute()
        >>> print(wer_score)
        0.5
    """

    def __init__(self, concatenate_texts: bool = False):
        super().__init__()
        self.concatenate_texts = concatenate_texts

    def update(self, preds: Any, target: Any) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Any:
        if self.concatenate_texts:
            return compute_measures(self.target, self.preds)["wer"]
        incorrect = 0
        total = 0
        for prediction, reference in zip(self.preds, self.target):
            measures = compute_measures(reference, prediction)
            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total += measures["substitutions"] + measures["deletions"] + measures["hits"]
        return incorrect / total
