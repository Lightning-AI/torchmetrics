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
import pytest
from torchmetrics.functional.text.edit import edit_distance


@pytest.mark.parametrize(
    ("left", "right", "substitution_cost", "expected"),
    [
        ("abc", "ca", 1, 3),
        ("abc", "ca", 5, 3),
        ("wants", "wasp", 1, 3),
        ("wants", "wasp", 5, 3),
        ("rain", "shine", 1, 3),
        ("rain", "shine", 2, 5),
        ("acbdef", "abcdef", 1, 2),
        ("acbdef", "abcdef", 2, 2),
        ("lnaguaeg", "language", 1, 4),
        ("lnaguaeg", "language", 2, 4),
        ("lnaugage", "language", 1, 3),
        ("lnaugage", "language", 2, 4),
        ("lngauage", "language", 1, 2),
        ("lngauage", "language", 2, 2),
        ("wants", "swim", 1, 5),
        ("wants", "swim", 2, 7),
        ("kitten", "sitting", 1, 3),
        ("kitten", "sitting", 2, 5),
        ("duplicated", "duuplicated", 1, 1),
        ("duplicated", "duuplicated", 2, 1),
        ("very duplicated", "very duuplicateed", 2, 2),
    ],
)
def test_for_correctness(
    left: str,
    right: str,
    substitution_cost: int,
    expected,
):
    """Test the underlying implementation of edit distance.

    Test cases taken from:
    https://github.com/nltk/nltk/blob/develop/nltk/test/unit/test_distance.py
    """
    for s1, s2 in ((left, right), (right, left)):
        predicted = edit_distance(
            s1,
            s2,
            substitution_cost=substitution_cost,
        )
        assert predicted == expected
