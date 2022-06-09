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
from typing import Any, Dict

import torch
from torch import Tensor

from torchmetrics import Metric
from torchmetrics.functional.text.squad import (
    PREDS_TYPE,
    TARGETS_TYPE,
    _squad_compute,
    _squad_input_check,
    _squad_update,
)


class SQuAD(Metric):
    """Calculate `SQuAD Metric`_ which corresponds to the scoring script for version 1 of the Stanford Question
    Answering Dataset (SQuAD).

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics import SQuAD
        >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
        >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]
        >>> squad = SQuAD()
        >>> squad(preds, target)
        {'exact_match': tensor(100.), 'f1': tensor(100.)}

    References:
        [1] SQuAD: 100,000+ Questions for Machine Comprehension of Text by Pranav Rajpurkar, Jian Zhang, Konstantin
        Lopyrev, Percy Liang `SQuAD Metric`_ .
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    f1_score: Tensor
    exact_match: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.add_state(name="f1_score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state(name="exact_match", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state(name="total", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds: PREDS_TYPE, target: TARGETS_TYPE) -> None:  # type: ignore
        """Compute F1 Score and Exact Match for a collection of predictions and references.

        Args:
            preds:
                A Dictionary or List of Dictionary-s that map ``id`` and ``prediction_text`` to the respective values.
                Example prediction:

                .. code-block:: python

                    {"prediction_text": "TorchMetrics is awesome", "id": "123"}

            target:
                A Dictionary or List of Dictionary-s that contain the ``answers`` and ``id`` in the SQuAD Format.
                Example target:

                .. code-block:: python

                    {
                        'answers': [{'answer_start': [1], 'text': ['This is a test answer']}],
                        'id': '1',
                    }

                Reference SQuAD Format:

                .. code-block:: python

                    {
                        'answers': {'answer_start': [1], 'text': ['This is a test text']},
                        'context': 'This is a test context.',
                        'id': '1',
                        'question': 'Is this a test?',
                        'title': 'train test'
                    }

        Raises:
            KeyError:
                If the required keys are missing in either predictions or targets.
        """
        preds_dict, target_dict = _squad_input_check(preds, target)
        f1_score, exact_match, total = _squad_update(preds_dict, target_dict)
        self.f1_score += f1_score
        self.exact_match += exact_match
        self.total += total

    def compute(self) -> Dict[str, Tensor]:
        """Aggregate the F1 Score and Exact match for the batch.

        Return:
            Dictionary containing the F1 score, Exact match score for the batch.
        """
        return _squad_compute(self.f1_score, self.exact_match, self.total)
