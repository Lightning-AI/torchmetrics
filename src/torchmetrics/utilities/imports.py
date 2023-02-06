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
import operator
from typing import Optional

from lightning_utilities.core.imports import compare_version, package_available

_TORCH_LOWER_1_12_DEV: Optional[bool] = compare_version("torch", operator.lt, "1.12.0.dev")
_TORCH_GREATER_EQUAL_1_9: Optional[bool] = compare_version("torch", operator.ge, "1.9.0")
_TORCH_GREATER_EQUAL_1_10: Optional[bool] = compare_version("torch", operator.ge, "1.10.0")
_TORCH_GREATER_EQUAL_1_11: Optional[bool] = compare_version("torch", operator.ge, "1.11.0")
_TORCH_GREATER_EQUAL_1_12: Optional[bool] = compare_version("torch", operator.ge, "1.12.0")

_JIWER_AVAILABLE: bool = package_available("jiwer")
_NLTK_AVAILABLE: bool = package_available("nltk")
_ROUGE_SCORE_AVAILABLE: bool = package_available("rouge_score")
_BERTSCORE_AVAILABLE: bool = package_available("bert_score")
_SCIPY_AVAILABLE: bool = package_available("scipy")
_TORCH_FIDELITY_AVAILABLE: bool = package_available("torch_fidelity")
_LPIPS_AVAILABLE: bool = package_available("lpips")
_PYCOCOTOOLS_AVAILABLE: bool = package_available("pycocotools")
_TORCHVISION_AVAILABLE: bool = package_available("torchvision")
_TORCHVISION_GREATER_EQUAL_0_8: Optional[bool] = compare_version("torchvision", operator.ge, "0.8.0")
_TQDM_AVAILABLE: bool = package_available("tqdm")
_TRANSFORMERS_AVAILABLE: bool = package_available("transformers")
_PESQ_AVAILABLE: bool = package_available("pesq")
_SACREBLEU_AVAILABLE: bool = package_available("sacrebleu")
_REGEX_AVAILABLE: bool = package_available("regex")
_PYSTOI_AVAILABLE: bool = package_available("pystoi")
_FAST_BSS_EVAL_AVAILABLE: bool = package_available("fast_bss_eval")
_MATPLOTLIB_AVAILABLE: bool = package_available("matplotlib")
_MULTIPROCESSING_AVAILABLE: bool = package_available("multiprocessing")
_XLA_AVAILABLE: bool = package_available("torch_xla")
