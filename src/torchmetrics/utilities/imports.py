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
"""Import utilities."""
import shutil
import sys

from lightning_utilities.core.imports import RequirementCache
from packaging.version import Version, parse

_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
_PYTHON_LOWER_3_8: bool = parse(_PYTHON_VERSION) < Version("3.8")
_TORCH_LOWER_2_0: bool = RequirementCache("torch<2.0.0")
_TORCH_GREATER_EQUAL_1_11: bool = RequirementCache("torch>=1.11.0")
_TORCH_GREATER_EQUAL_1_12: bool = RequirementCache("torch>=1.12.0")
_TORCH_GREATER_EQUAL_1_13: bool = RequirementCache("torch>=1.13.0")
_TORCH_GREATER_EQUAL_2_0: bool = RequirementCache("torch>=2.0.0")
_TORCH_GREATER_EQUAL_2_1: bool = RequirementCache("torch>=2.1.0")

_JIWER_AVAILABLE: bool = RequirementCache("jiwer")
_NLTK_AVAILABLE: bool = RequirementCache("nltk")
_ROUGE_SCORE_AVAILABLE: bool = RequirementCache("rouge_score")
_BERTSCORE_AVAILABLE: bool = RequirementCache("bert_score")
_SCIPY_AVAILABLE: bool = RequirementCache("scipy")
_SCIPY_GREATER_EQUAL_1_8 = RequirementCache("scipy>=1.8.0")
_TORCH_FIDELITY_AVAILABLE: bool = RequirementCache("torch_fidelity")
_LPIPS_AVAILABLE: bool = RequirementCache("lpips")
_PYCOCOTOOLS_AVAILABLE: bool = RequirementCache("pycocotools")
_TORCHVISION_AVAILABLE: bool = RequirementCache("torchvision")
_TORCHVISION_GREATER_EQUAL_0_8: bool = RequirementCache("torchvision>=0.8.0")
_TORCHVISION_GREATER_EQUAL_0_13: bool = RequirementCache("torchvision>=0.13.0")
_TQDM_AVAILABLE: bool = RequirementCache("tqdm")
_TRANSFORMERS_AVAILABLE: bool = RequirementCache("transformers")
_TRANSFORMERS_GREATER_EQUAL_4_4: bool = RequirementCache("transformers>=4.4.0")
_TRANSFORMERS_GREATER_EQUAL_4_10: bool = RequirementCache("transformers>=4.10.0")
_PESQ_AVAILABLE: bool = RequirementCache("pesq")
_GAMMATONE_AVAILABLE: bool = RequirementCache("gammatone")
_TORCHAUDIO_AVAILABLE: bool = RequirementCache("torchaudio")
_TORCHAUDIO_GREATER_EQUAL_0_10: bool = RequirementCache("torchaudio>=0.10.0")
_SACREBLEU_AVAILABLE: bool = RequirementCache("sacrebleu")
_REGEX_AVAILABLE: bool = RequirementCache("regex")
_PYSTOI_AVAILABLE: bool = RequirementCache("pystoi")
_FAST_BSS_EVAL_AVAILABLE: bool = RequirementCache("fast_bss_eval")
_MATPLOTLIB_AVAILABLE: bool = RequirementCache("matplotlib")
_SCIENCEPLOT_AVAILABLE: bool = RequirementCache("scienceplots")
_MULTIPROCESSING_AVAILABLE: bool = RequirementCache("multiprocessing")
_XLA_AVAILABLE: bool = RequirementCache("torch_xla")
_PIQ_GREATER_EQUAL_0_8: bool = RequirementCache("piq>=0.8.0")
_FASTER_COCO_EVAL_AVAILABLE: bool = RequirementCache("faster_coco_eval")
_MECAB_AVAILABLE: bool = RequirementCache("MeCab")
_MECAB_KO_AVAILABLE: bool = RequirementCache("mecab_ko")
_MECAB_KO_DIC_AVAILABLE: bool = RequirementCache("mecab_ko_dic")
_IPADIC_AVAILABLE: bool = RequirementCache("ipadic")
_SENTENCEPIECE_AVAILABLE: bool = RequirementCache("sentencepiece")

_LATEX_AVAILABLE: bool = shutil.which("latex") is not None
