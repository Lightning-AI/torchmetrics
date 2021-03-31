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
import pytest
from sklearn.metrics import average_precision_score as sk_average_precision

from tests.retrieval.helpers import _test_dtypes, _test_input_shapes, _test_retrieval_against_sklearn
from torchmetrics.retrieval.mean_average_precision import RetrievalMAP


@pytest.mark.parametrize('size', [1, 4, 10])
@pytest.mark.parametrize('n_documents', [1, 5])
@pytest.mark.parametrize('empty_target_action', ['skip', 'pos', 'neg'])
def test_results(size, n_documents, empty_target_action):
    """ Test metrics are computed correctly. """
    _test_retrieval_against_sklearn(
        sk_average_precision, RetrievalMAP, size, n_documents, empty_target_action
    )


def test_dtypes():
    """ Check dypes are managed correctly. """
    _test_dtypes(RetrievalMAP)


def test_input_shapes() -> None:
    """Check inputs shapes are managed correctly. """
    _test_input_shapes(RetrievalMAP)
