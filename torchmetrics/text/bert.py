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
from typing import Any, Callable, Dict, List, Optional

from torchmetrics.functional import bert_score
from torchmetrics.metric import Metric


def _flatten(x: List[List[str]]) -> List[str]:
    """converts list of list to single list of strings."""
    return [e for y in x for e in y]


class BERTScore(Metric):
    """`BERTScore <https://arxiv.org/abs/1904.09675>`_ leverages the pre-trained contextual embeddings from BERT
    and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate
    with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision,
    recall, and F1 measure, which can be useful for evaluating different language generation tasks.

    Args:
        predictions: candidate sentences
        references: reference sentences
        model_type: bert specification
        num_layers: the layer of representation to use.
        verbose: turn on intermediate status update
        idf: use idf weighting, can also be a precomputed idf_dict
        device: on which the contextual embedding model will be allocated on.
        num_threads: number of threads
        batch_size: bert score processing batch size
        lang: language of the sentences
        rescale_with_baseline: rescale bertscore with pre-computed baseline
        baseline_path: customized baseline file
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Returns:
        Dict containing the keys `precision`, `recall`, `f1` and `hashcode` with corresponding values

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "master kenobi"]
        >>> bertscore = BERTScore()
        >>> bertscore.update(predictions=predictions,references=references)
        >>> bertscore.compute()  # doctest: +SKIP
        {'f1': [0.99..., 0.99...],
         'hashcode': '...',
         'precision': [0.99..., 0.99...],
         'recall': [0.99..., 0.99...]}
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        lang: str = "en",
        num_layers: int = None,
        verbose: bool = False,
        idf: bool = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        num_threads: int = 4,
        all_layers: bool = False,
        rescale_with_baseline: bool = False,
        baseline_path: Optional[str] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.baseline_path = baseline_path
        self.rescale_with_baseline = rescale_with_baseline
        self.lang = lang
        self.all_layers = all_layers
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.device = device
        self.idf = idf
        self.verbose = verbose
        self.num_layers = num_layers
        self.model_type = model_type
        self.add_state("predictions", [], dist_reduce_fx="cat")
        self.add_state("references", [], dist_reduce_fx="cat")

    def update(self, predictions: List[str], references: List[str]) -> None:  # type: ignore
        """Store predictions/references for computing BERT scores.

        Args:
            predictions: List of predicted sentences
            references: List of references
        """
        self.predictions.append(predictions)
        self.references.append(references)

    def compute(self) -> Dict:
        """Calculate Bertscores.

        Return:
            Dict with Bertscores.
        """
        return bert_score(
            predictions=_flatten(self.predictions),
            references=_flatten(self.references),
            model_type=self.model_type,
            num_layers=self.num_layers,
            verbose=self.verbose,
            idf=self.idf,
            device=self.device,
            baseline_path=self.baseline_path,
            batch_size=self.batch_size,
            lang=self.lang,
            all_layers=self.all_layers,
            num_threads=self.num_threads,
        )
