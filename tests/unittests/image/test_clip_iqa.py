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
import piq
import pytest
import torch
from torchmetrics.functional.image.clip_iqa import clip_iqa


def test_compare():
    """Compare the output of the function with the output of the reference implementation."""
    img = torch.randint(255, (1, 3, 256, 256))
    clip_iqa(img)

    piq.CLIPIQA()(img)


@pytest.mark.parametrize("prompts", [("Good", "Bad", "Ugly"), "something", ["quality", "ugly"]])
def test_raises_error_on_wrong_prompts(prompts):
    """Test that the function raises an error if the prompts argument are not valid."""
    img = torch.randint(255, (1, 3, 256, 256))

    with pytest.raises(ValueError, match="Invalid prompt:.*"):
        clip_iqa(img, prompts=prompts)
