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
from typing import Dict, List

from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE

if _BERTSCORE_AVAILABLE:
    import bert_score as bert


def bert_score(
    predictions: List,
    references: List,
    model_type: str = None,
    num_layers: int = None,
    verbose: bool = False,
    idf: bool = False,
    device: str = None,
    batch_size: int = 64,
    nthreads: int = 4,
    all_layers: bool = False,
    lang: str = "en",
    rescale_with_baseline: bool = False,
    baseline_path: str = None,
) -> Dict:
    """BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and
    reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level
    and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be
    useful for evaluating different language generation tasks.

    Args:
        - `predictions` (list of str): candidate sentences
        - `references: `refs` (list of str): reference sentences
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
        - `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - `baseline_path` (str): customized baseline file

    Returns:
        - `(P, R, F)`: each is of shape (N); N = number of input
                      candidate reference pairs. if returning hashcode, the
                      output will be ((P, R, F), hashcode). If a candidate have
                      multiple references, the returned score of this candidate is
                      the *best* score among all references.

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "general kenobi"]
        >>> results = bertscore(predictions=predictions, references=references, lang="en")
        >>> print([round(v, 2) for v in results["f1"]])
        [1.0, 1.0]
    """
    if model_type is None:
        model_type = bert.lang2model[lang.lower()]

    if num_layers is None:
        num_layers = bert.model2layers[model_type]

    hashcode = bert.get_hash(
        model=model_type,
        num_layers=num_layers,
        idf=idf,
        rescale_with_baseline=rescale_with_baseline,
        use_custom_baseline=baseline_path is not None,
    )

    cached_bertscorer = bert.BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        nthreads=nthreads,
        all_layers=all_layers,
        idf=idf,
        device=device,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        baseline_path=baseline_path,
    )
    if cached_bertscorer.hash != hashcode:
        cached_bertscorer = bert.BERTScorer(
            model_type=model_type,
            num_layers=num_layers,
            batch_size=batch_size,
            nthreads=nthreads,
            all_layers=all_layers,
            idf=idf,
            device=device,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            baseline_path=baseline_path,
        )

    (P, R, F) = cached_bertscorer.score(
        cands=predictions,
        refs=references,
        verbose=verbose,
        batch_size=batch_size,
    )
    output_dict = {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F.tolist(),
        "hashcode": hashcode,
    }
    return output_dict
