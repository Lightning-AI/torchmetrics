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
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from torchmetrics import Metric
from torchmetrics.functional.text.squad import PREDS_TYPE, TARGETS_TYPE, _squad_compute, _squad_update


class SQuAD(Metric):
    """Calculate `SQuAD Metric`_.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    Example:
        >>> from torchmetrics.text.squad import SQuAD
        >>> predictions = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
        >>> references = [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]
        >>> _sqaud = SQuAD()
        >>> _sqaud(predictions, references)
        {'exact_match': tensor(100.), 'f1': tensor(100.)}

    References:
        [1] SQuAD: 100,000+ Questions for Machine Comprehension of Text by Pranav Rajpurkar, Jian Zhang, Konstantin
        Lopyrev, Percy Liang `SQuAD Metric`_ .
    """

    higher_is_better = True

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state(name="f1_score", default=[], dist_reduce_fx=None)
        self.add_state(name="exact_match", default=[], dist_reduce_fx=None)
        self.add_state(name="total", default=[], dist_reduce_fx=None)

    def update(
        self,
        preds: PREDS_TYPE,
        targets: TARGETS_TYPE,
    ) -> None:  # type: ignore
        """Compute F1 Score and Exact Match for a collection of predictions and references.

        Args:
            preds:
                An iterable of predicted sentences or a single predicted sentence.
            targets:
                An iterable of target sentences or a single target sentence.

        Raises:
            KeyError:
                If the required keys are missing in either predictions or targets.
        """
        if isinstance(preds, Dict):
            preds = [preds]

        if isinstance(targets, Dict):
            targets = [targets]

        for pred in preds:
            keys = pred.keys()
            if "prediction_text" not in keys or "id" not in keys:
                raise KeyError(
                    "Expected keys in a single prediction are 'prediction_text' and 'id'."
                    "Please make sure that 'prediction_text' maps to the answer string and 'id' maps to the key string."
                )

        for target in targets:
            keys = target.keys()
            if "answers" not in keys or "id" not in keys:
                raise KeyError(
                    "Expected keys in a single target are 'answers' and 'id'."
                    "Please make sure that 'answers' maps to a `SQuAD` format dictionary and 'id' maps to the key"
                    "string.\n"
                    "SQuAD Format: "
                    "{"
                    "    'answers': {"
                    "        'answer_start': [1],"
                    "        'text': ['This is a test text']"
                    "    },"
                    "    'context': 'This is a test context.',"
                    "    'id': '1',"
                    "    'question': 'Is this a test?',"
                    "    'title': 'train test'"
                    "}"
                )

            answers_keys = target["answers"].keys()
            if "text" not in answers_keys:
                raise KeyError(
                    "Expected keys in a 'answers' are 'text'."
                    "Please make sure that 'answer' maps to a `SQuAD` format dictionary.\n"
                    "SQuAD Format: "
                    "{"
                    "    'answers': {"
                    "        'answer_start': [1],"
                    "        'text': ['This is a test text']"
                    "    },"
                    "    'context': 'This is a test context.',"
                    "    'id': '1',"
                    "    'question': 'Is this a test?',"
                    "    'title': 'train test'"
                    "}"
                )

        preds_dict = {prediction["id"]: prediction["prediction_text"] for prediction in preds}
        targets_dict = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [{"text": answer_text} for answer_text in target["answers"]["text"]],
                                "id": target["id"],
                            }
                            for target in targets
                        ]
                    }
                ]
            }
        ]
        scores: Tuple[Tensor, Tensor, Tensor] = _squad_update(preds_dict, targets_dict)
        getattr(self, "f1_score").append(scores[0].to(self.device))
        getattr(self, "exact_match").append(scores[1].to(self.device))
        getattr(self, "total").append(scores[2].to(self.device))

    def compute(self) -> Dict[str, Tensor]:
        """Aggregate the F1 Score and Exact match for the batch.

        Return:
            Dictionary containing the F1 score, Exact match score for the batch.
        """
        f1_score = torch.sum(torch.tensor(getattr(self, "f1_score")))
        exact_match = torch.sum(torch.tensor(getattr(self, "exact_match")))
        total = torch.sum(torch.tensor(getattr(self, "total")))
        return _squad_compute((f1_score, exact_match, total))
