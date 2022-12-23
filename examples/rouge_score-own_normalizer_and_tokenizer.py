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
"""An example of how to use ROUGEScore with a user's defined/own normalizer and tokenizer.

To run: python rouge_score-own_normalizer_and_tokenizer.py
"""

import re
from pprint import pprint
from typing import Sequence

from torchmetrics.text.rouge import ROUGEScore


class UserNormalizer:
    """The `UserNormalizer` class is required to normalize a non-alphabet language text input.

    The user's defined normalizer is expected to return string that are fed into the tokenizer.
    """

    def __init__(self) -> None:
        self.pattern = r"[^a-z0-9]+"

    def __call__(self, text: str) -> str:
        """The `__call__` method must be defined for this class. To ensure the functionality, the `__call__` method
        should obey the input/output arguments structure described below.

        Args:
            text: Input text.

        Return:
            Normalized python string object
        """
        output_text = re.sub(self.pattern, " ", text.lower())

        return output_text


class UserTokenizer:
    """The `UserNormalizer` class is required to tokenize a non-alphabet language text input.

    The user's defined tokenizer is expected to return ``Sequence[str]`` that are fed into the rouge score.
    """

    pattern = r"\s+"

    def __call__(self, text: str) -> Sequence[str]:
        """The `__call__` method must be defined for this class. To ensure the functionality, the `__call__` method
        should obey the input/output arguments structure described below.

        Args:
            text: Input text.

        Return:
            Tokenized sentence
        """
        output_tokens = re.split(self.pattern, text)

        return output_tokens


_PREDS = ["hello", "hello world", "world world world"]
_REFS = ["hello", "hello hello", "hello world hello"]


if __name__ == "__main__":
    normalizer = UserNormalizer()
    tokenizer = UserTokenizer()

    rouge_score = ROUGEScore(normalizer=normalizer, tokenizer=tokenizer)

    rouge_score.update(_PREDS, _REFS)

    pprint(rouge_score.compute())
