from typing import Any, Dict

from torchmetrics.functional.text import (
    apply_no_ans_threshold,
    find_all_best_thresh,
    get_raw_scores,
    make_eval_dict,
    make_qid_to_has_ans,
    merge_eval,
)
from torchmetrics.metric import Metric


class SQuADv2(Metric):
    """
    This metric wrap the official scoring script for version 2 of the Stanford Question
    Answering Dataset (SQuAD).
    Computes SQuAD v2 scores (F1 and EM).
Args:
    predictions: List of triple for question-answers to score with the following elements:
        - the question-answer 'id' field as given in the references (see below)
        - the text of the answer
        - the probability that the question has no answer
    references: List of question-answers dictionaries with the following key-values:
            - 'id': id of the question-answer pair (see above),
            - 'answers': a list of Dict {'text': text of the answer as a string}
    no_answer_threshold: float
        Probability threshold to decide that a question has no answer.
Returns:
    'exact': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
    'total': Number of score considered
    'HasAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'HasAns_f1': The F-score of predicted tokens versus the gold answer
    'HasAns_total': Number of score considered
    'NoAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'NoAns_f1': The F-score of predicted tokens versus the gold answer
    'NoAns_total': Number of score considered
    'best_exact': Best exact match (with varying threshold)
    'best_exact_thresh': No-answer probability threshold associated to the best exact match
    'best_f1': Best F1 (with varying threshold)
    'best_f1_thresh': No-answer probability threshold associated to the best F1
    Examples:
    >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
    >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    >>> metrics = SQuADv2()
    >>> results = metrics.compute(predictions=predictions, references=references)
    >>> print(results)
    {'exact': 100.0,
     'f1': 100.0,
     'total': 1,
     'HasAns_exact': 100.0,
     'HasAns_f1': 100.0,
     'HasAns_total': 1,
      'best_exact': 100.0,
      'best_exact_thresh': 0.0,
      'best_f1': 100.0,
      'best_f1_thresh': 0.0}
    """

    def __init__(self, n_gram: int = 4, smooth: bool = False, no_answer_threshold: float = 1.0):
        """
               Args:
                   n_gram: Gram value ranged from 1 to 4 (Default 4)
                   smooth: Whether or not to apply smoothing – Lin et al. 2004
               """
        super().__init__()
        self.n_gram = n_gram
        self.smooth = smooth
        self.no_answer_threshold = no_answer_threshold

    def update(self, predictions: Any, references: Any) -> None:
        self.predictions = predictions
        self.references = references

    def compute(self) -> Dict[str, float]:
        no_answer_probabilities = dict((p["id"], p["no_answer_probability"]) for p in self.predictions)
        dataset = [{"paragraphs": [{"qas": self.references}]}]
        predictions = dict((p["id"], p["prediction_text"]) for p in self.predictions)

        qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

        exact_raw, f1_raw = get_raw_scores(dataset, predictions)
        exact_thresh = apply_no_ans_threshold(
            exact_raw, no_answer_probabilities, qid_to_has_ans, self.no_answer_threshold
        )
        f1_thresh = apply_no_ans_threshold(f1_raw, no_answer_probabilities, qid_to_has_ans, self.no_answer_threshold)
        out_eval = make_eval_dict(exact_thresh, f1_thresh)

        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
            merge_eval(out_eval, has_ans_eval, "HasAns")
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
            merge_eval(out_eval, no_ans_eval, "NoAns")
        find_all_best_thresh(out_eval, predictions, exact_raw, f1_raw, no_answer_probabilities, qid_to_has_ans)
        return dict(out_eval)
