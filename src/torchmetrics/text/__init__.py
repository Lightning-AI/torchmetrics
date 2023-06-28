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
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.text.eed import ExtendedEditDistance
from torchmetrics.text.mer import MatchErrorRate
from torchmetrics.text.perplexity import Perplexity
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.sacre_bleu import SacreBLEUScore
from torchmetrics.text.squad import SQuAD
from torchmetrics.text.ter import TranslationEditRate
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.wil import WordInfoLost
from torchmetrics.text.wip import WordInfoPreserved
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from torchmetrics.text.bert import BERTScore
    from torchmetrics.text.infolm import InfoLM


__all__ = [
    "BLEUScore",
    "CharErrorRate",
    "CHRFScore",
    "ExtendedEditDistance",
    "MatchErrorRate",
    "Perplexity",
    "ROUGEScore",
    "SacreBLEUScore",
    "SQuAD",
    "TranslationEditRate",
    "WordErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
]
