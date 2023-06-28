# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# referenced from
# Library Name: torchtext
# Authors: torchtext authors and @sluks
# Date: 2020-07-18
# Link: https://pytorch.org/text/_modules/torchtext/data/metrics.html#bleu_score

##############

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

##############

# MIT License
# Copyright (c) 2017 - Shujian Huang <huangsj@nju.edu.cn>

import re
from functools import partial
from typing import ClassVar, Optional, Sequence

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.utilities.imports import _REGEX_AVAILABLE

AVAILABLE_TOKENIZERS = ("none", "13a", "zh", "intl", "char")

_UCODE_RANGES = (
    ("\u3400", "\u4db5"),  # CJK Unified Ideographs Extension A, release 3.0
    ("\u4e00", "\u9fa5"),  # CJK Unified Ideographs, release 1.1
    ("\u9fa6", "\u9fbb"),  # CJK Unified Ideographs, release 4.1
    ("\uf900", "\ufa2d"),  # CJK Compatibility Ideographs, release 1.1
    ("\ufa30", "\ufa6a"),  # CJK Compatibility Ideographs, release 3.2
    ("\ufa70", "\ufad9"),  # CJK Compatibility Ideographs, release 4.1
    ("\u20000", "\u2a6d6"),  # (UTF16) CJK Unified Ideographs Extension B, release 3.1
    ("\u2f800", "\u2fa1d"),  # (UTF16) CJK Compatibility Supplement, release 3.1
    ("\uff00", "\uffef"),  # Full width ASCII, full width of English punctuation,
    # half width Katakana, half wide half width kana, Korean alphabet
    ("\u2e80", "\u2eff"),  # CJK Radicals Supplement
    ("\u3000", "\u303f"),  # CJK punctuation mark
    ("\u31c0", "\u31ef"),  # CJK stroke
    ("\u2f00", "\u2fdf"),  # Kangxi Radicals
    ("\u2ff0", "\u2fff"),  # Chinese character structure
    ("\u3100", "\u312f"),  # Phonetic symbols
    ("\u31a0", "\u31bf"),  # Phonetic symbols (Taiwanese and Hakka expansion)
    ("\ufe10", "\ufe1f"),
    ("\ufe30", "\ufe4f"),
    ("\u2600", "\u26ff"),
    ("\u2700", "\u27bf"),
    ("\u3200", "\u32ff"),
    ("\u3300", "\u33ff"),
)


class _SacreBLEUTokenizer:
    """Tokenizer used for SacreBLEU calculation.

    Source: https://github.com/mjpost/sacrebleu/tree/master/sacrebleu/tokenizers
    """

    _REGEX = (
        # language-dependent part (assuming Western languages)
        (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
        # tokenize period and comma unless preceded by a digit
        (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
        # tokenize period and comma unless followed by a digit
        (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
        # tokenize dash when preceded by a digit
        (re.compile(r"([0-9])(-)"), r"\1 \2 "),
        # one space only between words
        # NOTE: Doing this in Python (below) is faster
        # (re.compile(r'\s+'), r' '),
    )

    if _REGEX_AVAILABLE:
        import regex

        _INT_REGEX = (
            # Separate out punctuations preceeded by a non-digit
            (regex.compile(r"(\P{N})(\p{P})"), r"\1 \2 "),
            # Separate out punctuations followed by a non-digit
            (regex.compile(r"(\p{P})(\P{N})"), r" \1 \2"),
            # Separate out symbols
            (regex.compile(r"(\p{S})"), r" \1 "),
        )

    _TOKENIZE_FN: ClassVar[dict] = {
        "none": "_tokenize_base",
        "13a": "_tokenize_13a",
        "zh": "_tokenize_zh",
        "intl": "_tokenize_international",
        "char": "_tokenize_char",
    }

    def __init__(self, tokenize: Literal["none", "13a", "zh", "intl", "char"], lowercase: bool = False) -> None:
        self.tokenize_fn = getattr(self, self._TOKENIZE_FN[tokenize])
        self.lowercase = lowercase

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self.tokenize_fn(line)
        return self._lower(tokenized_line, self.lowercase).split()

    @classmethod
    def tokenize(
        cls, line: str, tokenize: Literal["none", "13a", "zh", "intl", "char"], lowercase: bool = False
    ) -> Sequence[str]:
        tokenize_fn = getattr(cls, cls._TOKENIZE_FN[tokenize])
        tokenized_line = tokenize_fn(line)
        return cls._lower(tokenized_line, lowercase).split()

    @classmethod
    def _tokenize_regex(cls, line: str) -> str:
        """Post-processing tokenizer for `13a` and `zh` tokenizers.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line
        """
        for _re, repl in cls._REGEX:
            line = _re.sub(repl, line)
        # no leading or trailing spaces, single space within words
        return " ".join(line.split())

    @staticmethod
    def _is_chinese_char(uchar: str) -> bool:
        """Check if character is chinese.

        Args:
            uchar: input char in unicode.

        Return:
            whether the input char is a Chinese character.
        """
        return any(start <= uchar <= end for start, end in _UCODE_RANGES)

    @classmethod
    def _tokenize_base(cls, line: str) -> str:
        """Tokenizes an input line with the tokenizer.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line
        """
        return line

    @classmethod
    def _tokenize_13a(cls, line: str) -> str:
        """Tokenizes a line using a relatively minimal tokenization that is equivalent to mteval-v13a, used by WMT.

        Args:
            line: input sentence

        Return:
            tokenized sentence
        """
        # language-independent part:
        line = line.replace("<skipped>", "")
        line = line.replace("-\n", "")
        line = line.replace("\n", " ")

        if "&" in line:
            line = line.replace("&quot;", '"')
            line = line.replace("&amp;", "&")
            line = line.replace("&lt;", "<")
            line = line.replace("&gt;", ">")

        return cls._tokenize_regex(f" {line} ")

    @classmethod
    def _tokenize_zh(cls, line: str) -> str:
        """Tokenization of Chinese text.

        This is done in two steps: separate each Chinese characters (by utf-8 encoding) and afterwards tokenize the
        Chinese part (following the `13a` i.e. mteval tokenizer).
        Author: Shujian Huang huangsj@nju.edu.cn.

        Args:
            line: input sentence

        Return:
            tokenized sentence
        """
        line = line.strip()
        line_in_chars = ""

        for char in line:
            if cls._is_chinese_char(char):
                line_in_chars += " "
                line_in_chars += char
                line_in_chars += " "
            else:
                line_in_chars += char

        return cls._tokenize_regex(line_in_chars)

    @classmethod
    def _tokenize_international(cls, line: str) -> str:
        r"""Tokenizes a string following the official BLEU implementation.

        See github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983

        In our case, the input string is expected to be just one line.
        We just tokenize on punctuation and symbols,
        except when a punctuation is preceded and followed by a digit
        (e.g. a comma/dot as a thousand/decimal separator).
        We do not recover escaped forms of punctuations such as &apos; or &gt;
        as these should never appear in MT system outputs (see issue #138)

        Note that a number (e.g., a year) followed by a dot at the end of
        sentence is NOT tokenized, i.e. the dot stays with the number because
        `s/(\\p{P})(\\P{N})/ $1 $2/g` does not match this case (unless we add a
        space after each sentence). However, this error is already in the
        original mteval-v14.pl and we want to be consistent with it.
        The error is not present in the non-international version,
        which uses `$norm_text = " $norm_text "`.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.
        """
        for _re, repl in cls._INT_REGEX:
            line = _re.sub(repl, line)

        return " ".join(line.split())

    @classmethod
    def _tokenize_char(cls, line: str) -> str:
        """Tokenizes all the characters in the input line.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line
        """
        return " ".join(char for char in line)

    @staticmethod
    def _lower(line: str, lowercase: bool) -> str:
        if lowercase:
            return line.lower()
        return line


def sacre_bleu_score(
    preds: Sequence[str],
    target: Sequence[Sequence[str]],
    n_gram: int = 4,
    smooth: bool = False,
    tokenize: Literal["none", "13a", "zh", "intl", "char"] = "13a",
    lowercase: bool = False,
    weights: Optional[Sequence[float]] = None,
) -> Tensor:
    """Calculate `BLEU score`_ [1] of machine translated text with one or more references.

    This implementation follows the behaviour of SacreBLEU [2] implementation from https://github.com/mjpost/sacrebleu.

    Args:
        preds: An iterable of machine translated corpus
        target: An iterable of iterables of reference corpus
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether to apply smoothing - see [2]
        tokenize: Tokenization technique to be used.
            Supported tokenization: ['none', '13a', 'zh', 'intl', 'char']
        lowercase: If ``True``, BLEU score over lowercased text is calculated.
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Return:
        Tensor with BLEU Score

    Raises:
        ValueError: If ``preds`` and ``target`` corpus have different lengths.
        ValueError: If a length of a list of weights is not ``None`` and not equal to ``n_gram``.

    Example:
        >>> from torchmetrics.functional.text import sacre_bleu_score
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> sacre_bleu_score(preds, target)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_

        [2] A Call for Clarity in Reporting BLEU Scores by Matt Post.

        [3] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_
    """
    if tokenize not in AVAILABLE_TOKENIZERS:
        raise ValueError(f"Argument `tokenize` expected to be one of {AVAILABLE_TOKENIZERS} but got {tokenize}.")

    if tokenize not in _SacreBLEUTokenizer._TOKENIZE_FN.keys():
        raise ValueError(
            f"Unsupported tokenizer selected. Please, choose one of {list(_SacreBLEUTokenizer._TOKENIZE_FN.keys())}"
        )
    if len(preds) != len(target):
        raise ValueError(f"Corpus has different size {len(preds)} != {len(target)}")
    if tokenize == "intl" and not _REGEX_AVAILABLE:
        raise ModuleNotFoundError(
            "`'intl'` tokenization requires that `regex` is installed."
            " Use `pip install regex` or `pip install torchmetrics[text]`."
        )

    if weights is not None and len(weights) != n_gram:
        raise ValueError(f"List of weights has different weights than `n_gram`: {len(weights)} != {n_gram}")
    if weights is None:
        weights = [1.0 / n_gram] * n_gram

    numerator = torch.zeros(n_gram)
    denominator = torch.zeros(n_gram)
    preds_len = tensor(0.0)
    target_len = tensor(0.0)

    tokenize_fn = partial(_SacreBLEUTokenizer.tokenize, tokenize=tokenize, lowercase=lowercase)
    preds_len, target_len = _bleu_score_update(
        preds,
        target,
        numerator,
        denominator,
        preds_len,
        target_len,
        n_gram,
        tokenize_fn,
    )

    return _bleu_score_compute(preds_len, target_len, numerator, denominator, n_gram, weights, smooth)
