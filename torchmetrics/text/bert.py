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


class BERTScore(Metric):
    """BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and
    reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level
    and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be
    useful for evaluating different language generation tasks.

    Args:
        - `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - `verbose` (bool): turn on intermediate status update
        - `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - `nthreads` (int): number of threads
        - `batch_size` (int): bert score processing batch size
        - `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - `return_hash` (bool): return hash code of the setting
        - `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - `baseline_path` (str): customized baseline file

    Returns:
        - precision: Precision.
        - recall: Recall.
        - f1: F1 score.
        - hashcode: Hashcode of the library.

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "general kenobi"]
        >>> results = bert_score(predictions=predictions, references=references, lang="en")
        >>> print([round(v, 2) for v in results["f1"]])
        [1.0, 1.0]
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
        nthreads: int = 4,
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
        self.nthreads = nthreads
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
            references: List of refernces
        """
        self.predictions.append(predictions)
        self.references.append(references)

    def compute(self) -> Dict:
        """Calculate Bertscores.

        Return:
            Dict with Bertscores.
        """
        return bert_score(
            predictions=self.predictions[0],
            references=self.references[0],
            model_type=self.model_type,
            num_layers=self.num_layers,
            verbose=self.verbose,
            idf=self.idf,
            device=self.device,
            baseline_path=self.baseline_path,
            batch_size=self.batch_size,
            lang=self.lang,
            all_layers=self.all_layers,
        )
