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
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_10

if _TRANSFORMERS_GREATER_EQUAL_4_10:
    from torchmetrics.functional.multimodal.clip_iqa import clip_image_quality_assessment
    from torchmetrics.functional.multimodal.clip_score import clip_score

    __all__ = ["clip_image_quality_assessment", "clip_score"]
