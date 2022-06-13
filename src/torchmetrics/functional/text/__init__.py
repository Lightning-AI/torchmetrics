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

from torchmetrics.functional.text.bleu import bleu_score  # noqa: F401
from torchmetrics.functional.text.cer import char_error_rate  # noqa: F401
from torchmetrics.functional.text.chrf import chrf_score  # noqa: F401
from torchmetrics.functional.text.eed import extended_edit_distance  # noqa: F401
from torchmetrics.functional.text.mer import match_error_rate  # noqa: F401
from torchmetrics.functional.text.sacre_bleu import sacre_bleu_score  # noqa: F401
from torchmetrics.functional.text.squad import squad  # noqa: F401
from torchmetrics.functional.text.ter import translation_edit_rate  # noqa: F401
from torchmetrics.functional.text.wer import word_error_rate  # noqa: F401
from torchmetrics.functional.text.wil import word_information_lost  # noqa: F401
from torchmetrics.functional.text.wip import word_information_preserved  # noqa: F401
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _TRANSFORMERS_AUTO_AVAILABLE

if _TRANSFORMERS_AUTO_AVAILABLE:
    from torchmetrics.functional.text.bert import bert_score  # noqa: F401

if _NLTK_AVAILABLE:
    from torchmetrics.functional.text.rouge import rouge_score  # noqa: F401
