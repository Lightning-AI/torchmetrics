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
import numpy as np
import pytest

from tests.retrieval.helpers import _test_dtypes, _test_input_args, _test_input_shapes, _test_retrieval_against_sklearn
from torchmetrics.retrieval.retrieval_precision import RetrievalPrecision


def _precision_at_k(target: np.array, preds: np.array, k: int = None):
    """
    Didn't find a reliable implementation of Precision in Information Retrieval, so,
    reimplementing here. A good explanation can be found
    `here <https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf>_`.
    """
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    if k is None:
        k = len(preds)

    if target.sum() > 0:
        order_indexes = np.argsort(preds, axis=0)[::-1]
        relevant = np.sum(target[order_indexes][:k])
        return relevant * 1.0 / k
    else:
        return np.NaN


@pytest.mark.parametrize('size', [1, 4, 10])
@pytest.mark.parametrize('n_documents', [1, 5])
@pytest.mark.parametrize('empty_target_action', ['skip', 'pos', 'neg'])
@pytest.mark.parametrize('k', [None, 1, 4, 10])
def test_results(size, n_documents, empty_target_action, k):
    """ Test metrics are computed correctly. """
    _test_retrieval_against_sklearn(
        _precision_at_k, RetrievalPrecision, size, n_documents, empty_target_action, k=k
    )


def test_dtypes():
    """ Check dypes are managed correctly. """
    _test_dtypes(RetrievalPrecision)


def test_input_shapes() -> None:
    """Check inputs shapes are managed correctly. """
    _test_input_shapes(RetrievalPrecision)


@pytest.mark.parametrize('k', [-1, 1.0])
def test_input_params(k) -> None:
    """Check invalid args are managed correctly. """
    _test_input_args(RetrievalPrecision, "`k` has to be a positive integer or None", k=k)
