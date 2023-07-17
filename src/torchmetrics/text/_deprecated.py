from typing import Any, Literal, Optional, Sequence

from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.text.eed import ExtendedEditDistance
from torchmetrics.text.mer import MatchErrorRate
from torchmetrics.text.perplexity import Perplexity
from torchmetrics.text.sacre_bleu import SacreBLEUScore
from torchmetrics.text.squad import SQuAD
from torchmetrics.text.ter import TranslationEditRate
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.wil import WordInfoLost
from torchmetrics.text.wip import WordInfoPreserved
from torchmetrics.utilities.prints import _deprecated_root_import_class


class _BLEUScore(BLEUScore):
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> bleu = _BLEUScore()
    >>> bleu(preds, target)
    tensor(0.7598)
    """

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("BLEUScore", "text")
        super().__init__(n_gram=n_gram, smooth=smooth, weights=weights, **kwargs)


class _CharErrorRate(CharErrorRate):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> cer = _CharErrorRate()
    >>> cer(preds, target)
    tensor(0.3415)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("CharErrorRate", "text")
        super().__init__(**kwargs)


class _CHRFScore(CHRFScore):
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> chrf = _CHRFScore()
    >>> chrf(preds, target)
    tensor(0.8640)
    """

    def __init__(
        self,
        n_char_order: int = 6,
        n_word_order: int = 2,
        beta: float = 2.0,
        lowercase: bool = False,
        whitespace: bool = False,
        return_sentence_level_score: bool = False,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("CHRFScore", "text")
        super().__init__(
            n_char_order=n_char_order,
            n_word_order=n_word_order,
            beta=beta,
            lowercase=lowercase,
            whitespace=whitespace,
            return_sentence_level_score=return_sentence_level_score,
            **kwargs,
        )


class _ExtendedEditDistance(ExtendedEditDistance):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "here is an other sample"]
    >>> target = ["this is the reference", "here is another one"]
    >>> eed = _ExtendedEditDistance()
    >>> eed(preds=preds, target=target)
    tensor(0.3078)
    """

    def __init__(
        self,
        language: Literal["en", "ja"] = "en",
        return_sentence_level_score: bool = False,
        alpha: float = 2.0,
        rho: float = 0.3,
        deletion: float = 0.2,
        insertion: float = 1.0,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("ExtendedEditDistance", "text")
        super().__init__(
            language=language,
            return_sentence_level_score=return_sentence_level_score,
            alpha=alpha,
            rho=rho,
            deletion=deletion,
            insertion=insertion,
            **kwargs,
        )


class _MatchErrorRate(MatchErrorRate):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> mer = _MatchErrorRate()
    >>> mer(preds, target)
    tensor(0.4444)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("MatchErrorRate", "text")
        super().__init__(**kwargs)


class _Perplexity(Perplexity):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand(2, 8, 5, generator=torch.manual_seed(22))
    >>> target = torch.randint(5, (2, 8), generator=torch.manual_seed(22))
    >>> target[0, 6:] = -100
    >>> perp = _Perplexity(ignore_index=-100)
    >>> perp(preds, target)
    tensor(5.2545)
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("Perplexity", "text")
        super().__init__(ignore_index=ignore_index, **kwargs)


class _SacreBLEUScore(SacreBLEUScore):
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> sacre_bleu = _SacreBLEUScore()
    >>> sacre_bleu(preds, target)
    tensor(0.7598)
    """

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        tokenize: Literal["none", "13a", "zh", "intl", "char"] = "13a",
        lowercase: bool = False,
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("SacreBLEUScore", "text")
        super().__init__(
            n_gram=n_gram, smooth=smooth, tokenize=tokenize, lowercase=lowercase, weights=weights, **kwargs
        )


class _SQuAD(SQuAD):
    """Wrapper for deprecated import.

    >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
    >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]
    >>> squad = _SQuAD()
    >>> squad(preds, target)
    {'exact_match': tensor(100.), 'f1': tensor(100.)}
    """

    def __init__(self, **kwargs: Any) -> None:
        _deprecated_root_import_class("SQuAD", "text")
        super().__init__(**kwargs)


class _TranslationEditRate(TranslationEditRate):
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> ter = _TranslationEditRate()
    >>> ter(preds, target)
    tensor(0.1538)
    """

    def __init__(
        self,
        normalize: bool = False,
        no_punctuation: bool = False,
        lowercase: bool = True,
        asian_support: bool = False,
        return_sentence_level_score: bool = False,
        **kwargs: Any,
    ) -> None:
        _deprecated_root_import_class("TranslationEditRate", "text")
        super().__init__(
            normalize=normalize,
            no_punctuation=no_punctuation,
            lowercase=lowercase,
            asian_support=asian_support,
            return_sentence_level_score=return_sentence_level_score,
            **kwargs,
        )


class _WordErrorRate(WordErrorRate):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> wer = _WordErrorRate()
    >>> wer(preds, target)
    tensor(0.5000)
    """

    def __init__(self, **kwargs: Any) -> None:
        _deprecated_root_import_class("WordErrorRate", "text")
        super().__init__(**kwargs)


class _WordInfoLost(WordInfoLost):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> wil = _WordInfoLost()
    >>> wil(preds, target)
    tensor(0.6528)
    """

    def __init__(self, **kwargs: Any) -> None:
        _deprecated_root_import_class("WordInfoLost", "text")
        super().__init__(**kwargs)


class _WordInfoPreserved(WordInfoPreserved):
    """Wrapper for deprecated import.

    >>> preds = ["this is the prediction", "there is an other sample"]
    >>> target = ["this is the reference", "there is another one"]
    >>> wip = WordInfoPreserved()
    >>> wip(preds, target)
    tensor(0.3472)
    """

    def __init__(self, **kwargs: Any) -> None:
        _deprecated_root_import_class("WordInfoPreserved", "text")
        super().__init__(**kwargs)
