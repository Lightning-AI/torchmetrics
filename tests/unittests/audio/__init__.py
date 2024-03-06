import os
from typing import Callable, Optional

from torch import Tensor

from unittests import _PATH_ALL_TESTS

_SAMPLE_AUDIO_SPEECH = os.path.join(_PATH_ALL_TESTS, "_data", "audio", "audio_speech.wav")
_SAMPLE_AUDIO_SPEECH_BAB_DB = os.path.join(_PATH_ALL_TESTS, "_data", "audio", "audio_speech_bab_0dB.wav")
_SAMPLE_NUMPY_ISSUE_895 = os.path.join(_PATH_ALL_TESTS, "_data", "audio", "issue_895.npz")


def _average_metric_wrapper(
    preds: Tensor, target: Tensor, metric_func: Callable, res_index: Optional[int] = None
) -> Tensor:
    """Average the metric values.

    Args:
        preds: predictions, shape[batch, spk, time]
        target: targets, shape[batch, spk, time]
        metric_func: a function which return best_metric and best_perm
        res_index: if not None, return best_metric[res_index]

    Returns:
        the average of best_metric

    """
    if res_index is not None:
        return metric_func(preds, target)[res_index].mean()
    return metric_func(preds, target).mean()
