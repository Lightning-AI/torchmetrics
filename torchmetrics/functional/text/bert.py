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
from typing import Dict, List, Optional

from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE

if _BERTSCORE_AVAILABLE:
    import bert_score as bert


def bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    num_layers: int = None,
    verbose: bool = False,
    idf: bool = False,
    device: Optional[str] = None,
    batch_size: int = 64,
    num_threads: int = 4,
    all_layers: bool = False,
    rescale_with_baseline: bool = False,
    baseline_path: Optional[str] = None,
) -> Dict:
    """
    BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and
    reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level
    and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be
    useful for evaluating different language generation tasks.

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

    Returns:
        (Dict): containing: Precision, Recall, F1 score, Hashcode of the library

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "general kenobi"]
        >>> results = bert_score(predictions=predictions, references=references, lang="en")
        >>> print([round(v, 2) for v in results["f1"]])
        [1.0, 1.0]
    """
    if not _BERTSCORE_AVAILABLE:
        raise ValueError(
            "bert_score metric requires that bert-score package is installed."
            " Either install with `pip install bert-score` or `pip install torchmetrics[text]`"
        )

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
        use_fast_tokenizer=True,
    )

    cached_bertscorer = bert.BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        nthreads=num_threads,
        all_layers=all_layers,
        idf=idf,
        device=device,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        baseline_path=baseline_path,
    )

    P, R, F = cached_bertscorer.score(
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
