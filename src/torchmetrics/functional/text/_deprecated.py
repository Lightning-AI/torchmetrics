import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.functional.text.chrf import chrf_score
from torchmetrics.functional.text.eed import extended_edit_distance
from torchmetrics.functional.text.infolm import (
    _ALLOWED_INFORMATION_MEASURE_LITERAL as _INFOLM_ALLOWED_INFORMATION_MEASURE_LITERAL,
)
from torchmetrics.functional.text.infolm import infolm
from torchmetrics.functional.text.mer import match_error_rate
from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.sacre_bleu import sacre_bleu_score
from torchmetrics.functional.text.squad import squad
from torchmetrics.functional.text.ter import translation_edit_rate
from torchmetrics.functional.text.wer import word_error_rate
from torchmetrics.functional.text.wil import word_information_lost
from torchmetrics.functional.text.wip import word_information_preserved
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _TRANSFORMERS_AVAILABLE
from torchmetrics.utilities.prints import _deprecated_root_import_func

__doctest_requires__ = {("_rouge_score"): ["nltk"]}

if not _TRANSFORMERS_AVAILABLE:
    __doctest_skip__ = ["_bert_score", "_infolm"]

SQUAD_SINGLE_TARGET_TYPE = Dict[str, Union[str, Dict[str, Union[List[str], List[int]]]]]
SQUAD_TARGETS_TYPE = Union[SQUAD_SINGLE_TARGET_TYPE, List[SQUAD_SINGLE_TARGET_TYPE]]


def _bert_score(
    preds: Union[List[str], Dict[str, Tensor]],
    target: Union[List[str], Dict[str, Tensor]],
    model_name_or_path: Optional[str] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    model: Optional[Module] = None,
    user_tokenizer: Any = None,
    user_forward_fn: Optional[Callable[[Module, Dict[str, Tensor]], Tensor]] = None,
    verbose: bool = False,
    idf: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 512,
    batch_size: int = 64,
    num_threads: int = 4,
    return_hash: bool = False,
    lang: str = "en",
    rescale_with_baseline: bool = False,
    baseline_path: Optional[str] = None,
    baseline_url: Optional[str] = None,
) -> Dict[str, Union[Tensor, List[float], str]]:
    """Wrapper for deprecated import.

    >>> preds = ["hello there", "general kenobi"]
    >>> target = ["hello there", "master kenobi"]
    >>> score = _bert_score(preds, target)
    >>> from pprint import pprint
    >>> pprint(score)
    {'f1': tensor([1.0000, 0.9961]),
     'precision': tensor([1.0000, 0.9961]),
     'recall': tensor([1.0000, 0.9961])}
    """
    _deprecated_root_import_func("bert_score", "text")
    return bert_score(
        preds=preds,
        target=target,
        model_name_or_path=model_name_or_path,
        num_layers=num_layers,
        all_layers=all_layers,
        model=model,
        user_tokenizer=user_tokenizer,
        user_forward_fn=user_forward_fn,
        verbose=verbose,
        idf=idf,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        num_threads=num_threads,
        return_hash=return_hash,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        baseline_path=baseline_path,
        baseline_url=baseline_url,
    )


def _bleu_score(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    n_gram: int = 4,
    smooth: bool = False,
    weights: Optional[Sequence[float]] = None,
) -> Tensor:
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> _bleu_score(preds, target)
    tensor(0.7598)
    """
    _deprecated_root_import_func("bleu_score", "text")
    return bleu_score(preds=preds, target=target, n_gram=n_gram, smooth=smooth, weights=weights)


def _char_error_rate(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> _char_error_rate(preds=preds, target=target)
    tensor(0.3415)
    """
    _deprecated_root_import_func("char_error_rate", "text")
    return char_error_rate(preds=preds, target=target)


def _chrf_score(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    n_char_order: int = 6,
    n_word_order: int = 2,
    beta: float = 2.0,
    lowercase: bool = False,
    whitespace: bool = False,
    return_sentence_level_score: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> _chrf_score(preds, target)
    tensor(0.8640)
    """
    _deprecated_root_import_func("chrf_score", "text")
    return chrf_score(
        preds=preds,
        target=target,
        n_char_order=n_char_order,
        n_word_order=n_word_order,
        beta=beta,
        lowercase=lowercase,
        whitespace=whitespace,
        return_sentence_level_score=return_sentence_level_score,
    )


def _extended_edit_distance(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    language: Literal["en", "ja"] = "en",
    return_sentence_level_score: bool = False,
    alpha: float = 2.0,
    rho: float = 0.3,
    deletion: float = 0.2,
    insertion: float = 1.0,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "here is an other sample"]
    >>> target = ["this is the reference", "here is another one"]
    >>> _extended_edit_distance(preds=preds, target=target)
    tensor(0.3078)
    """
    _deprecated_root_import_func("extended_edit_distance", "text")
    return extended_edit_distance(
        preds=preds,
        target=target,
        language=language,
        return_sentence_level_score=return_sentence_level_score,
        alpha=alpha,
        rho=rho,
        deletion=deletion,
        insertion=insertion,
    )


def _infolm(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    model_name_or_path: Union[str, os.PathLike] = "bert-base-uncased",
    temperature: float = 0.25,
    information_measure: _INFOLM_ALLOWED_INFORMATION_MEASURE_LITERAL = "kl_divergence",
    idf: bool = True,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_length: Optional[int] = None,
    batch_size: int = 64,
    num_threads: int = 0,
    verbose: bool = True,
    return_sentence_level_score: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Wrapper for deprecated import.

    >>> preds = ['he read the book because he was interested in world history']
    >>> target = ['he was interested in world history because he read the book']
    >>> _infolm(preds, target, model_name_or_path='google/bert_uncased_L-2_H-128_A-2', idf=False)
    tensor(-0.1784)
    """
    _deprecated_root_import_func("infolm", "text")
    return infolm(
        preds=preds,
        target=target,
        model_name_or_path=model_name_or_path,
        temperature=temperature,
        information_measure=information_measure,
        idf=idf,
        alpha=alpha,
        beta=beta,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        num_threads=num_threads,
        verbose=verbose,
        return_sentence_level_score=return_sentence_level_score,
    )


def _match_error_rate(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> _match_error_rate(preds=preds, target=target)
    tensor(0.4444)
    """
    _deprecated_root_import_func("match_error_rate", "text")
    return match_error_rate(preds=preds, target=target)


def _perplexity(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> Tensor:
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
    >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
    >>> target[0, 6:] = -100
    >>> _perplexity(preds, target, ignore_index=-100)
    tensor(5.2545)
    """
    _deprecated_root_import_func("perplexity", "text")
    return perplexity(preds=preds, target=target, ignore_index=ignore_index)


def _rouge_score(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str], Sequence[Sequence[str]]],
    accumulate: Literal["avg", "best"] = "best",
    use_stemmer: bool = False,
    normalizer: Optional[Callable[[str], str]] = None,
    tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
    rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
) -> Dict[str, Tensor]:
    """Wrapper for deprecated import.

    >>> preds = "My name is John"
    >>> target = "Is your name John"
    >>> from pprint import pprint
    >>> pprint(_rouge_score(preds, target))
    {'rouge1_fmeasure': tensor(0.7500),
        'rouge1_precision': tensor(0.7500),
        'rouge1_recall': tensor(0.7500),
        'rouge2_fmeasure': tensor(0.),
        'rouge2_precision': tensor(0.),
        'rouge2_recall': tensor(0.),
        'rougeL_fmeasure': tensor(0.5000),
        'rougeL_precision': tensor(0.5000),
        'rougeL_recall': tensor(0.5000),
        'rougeLsum_fmeasure': tensor(0.5000),
        'rougeLsum_precision': tensor(0.5000),
        'rougeLsum_recall': tensor(0.5000)}
    """
    _deprecated_root_import_func("rouge_score", "text")
    return rouge_score(
        preds=preds,
        target=target,
        accumulate=accumulate,
        use_stemmer=use_stemmer,
        normalizer=normalizer,
        tokenizer=tokenizer,
        rouge_keys=rouge_keys,
    )


def _sacre_bleu_score(
    preds: Sequence[str],
    target: Sequence[Sequence[str]],
    n_gram: int = 4,
    smooth: bool = False,
    tokenize: Literal["none", "13a", "zh", "intl", "char"] = "13a",
    lowercase: bool = False,
    weights: Optional[Sequence[float]] = None,
) -> Tensor:
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> _sacre_bleu_score(preds, target)
    tensor(0.7598)
    """
    _deprecated_root_import_func("sacre_bleu_score", "text")
    return sacre_bleu_score(
        preds=preds,
        target=target,
        n_gram=n_gram,
        smooth=smooth,
        tokenize=tokenize,
        lowercase=lowercase,
        weights=weights,
    )


def _squad(preds: Union[Dict[str, str], List[Dict[str, str]]], target: SQUAD_TARGETS_TYPE) -> Dict[str, Tensor]:
    """Wrapper for deprecated import.

    >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
    >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]},"id": "56e10a3be3433e1400422b22"}]
    >>> _squad(preds, target)
    {'exact_match': tensor(100.), 'f1': tensor(100.)}
    """
    _deprecated_root_import_func("squad", "text")
    return squad(preds=preds, target=target)


def _translation_edit_rate(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    normalize: bool = False,
    no_punctuation: bool = False,
    lowercase: bool = True,
    asian_support: bool = False,
    return_sentence_level_score: bool = False,
) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> _translation_edit_rate(preds, target)
    tensor(0.1538)
    """
    _deprecated_root_import_func("translation_edit_rate", "text")
    return translation_edit_rate(
        preds=preds,
        target=target,
        normalize=normalize,
        no_punctuation=no_punctuation,
        lowercase=lowercase,
        asian_support=asian_support,
        return_sentence_level_score=return_sentence_level_score,
    )


def _word_error_rate(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> _word_error_rate(preds=preds, target=target)
    tensor(0.5000)
    """
    _deprecated_root_import_func("word_error_rate", "text")
    return word_error_rate(preds=preds, target=target)


def _word_information_lost(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> _word_information_lost(preds, target)
    tensor(0.6528)
    """
    _deprecated_root_import_func("word_information_lost", "text")
    return word_information_lost(preds=preds, target=target)


def _word_information_preserved(preds: Union[str, List[str]], target: Union[str, List[str]]) -> Tensor:
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> _word_information_preserved(preds, target)
    tensor(0.3472)
    """
    _deprecated_root_import_func("word_information_preserved", "text")
    return word_information_preserved(preds=preds, target=target)
