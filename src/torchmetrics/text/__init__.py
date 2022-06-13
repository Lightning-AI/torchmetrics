# Copyright The PyTorch Lightning team.
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
from torchmetrics.text.bleu import BLEUScore  # noqa: F401
from torchmetrics.text.cer import CharErrorRate  # noqa: F401
from torchmetrics.text.chrf import CHRFScore  # noqa: F401
from torchmetrics.text.eed import ExtendedEditDistance  # noqa: F401
from torchmetrics.text.mer import MatchErrorRate  # noqa: F401
from torchmetrics.text.sacre_bleu import SacreBLEUScore  # noqa: F401
from torchmetrics.text.squad import SQuAD  # noqa: F401
from torchmetrics.text.ter import TranslationEditRate  # noqa: F401
from torchmetrics.text.wer import WordErrorRate  # noqa: F401
from torchmetrics.text.wil import WordInfoLost  # noqa: F401
from torchmetrics.text.wip import WordInfoPreserved  # noqa: F401
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _TRANSFORMERS_AUTO_AVAILABLE

if _TRANSFORMERS_AUTO_AVAILABLE:
    from torchmetrics.text.bert import BERTScore  # noqa: F401

if _NLTK_AVAILABLE:
    from torchmetrics.text.rouge import ROUGEScore  # noqa: F401
