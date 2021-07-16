from typing import Any

from jiwer import compute_measures


def wer(target: Any, preds: Any, concatenate_texts: bool = False) -> float:
    """
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
    if concatenate_texts:
        return compute_measures(target, preds)["wer"]
    incorrect = 0
    total = 0
    for prediction, reference in zip(preds, target):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total
