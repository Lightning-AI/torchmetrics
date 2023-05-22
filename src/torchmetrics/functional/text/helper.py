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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Copyright 2020 Memsource
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

import math
from enum import Enum, IntEnum, unique
from typing import Dict, List, Sequence, Tuple, Union

# Tercom-inspired limits
_BEAM_WIDTH = 25

# Sacrebleu-inspired limits
_MAX_CACHE_SIZE = 10000
_INT_INFINITY = int(1e16)


@unique
class _EditOperations(str, Enum):
    """Enumerations for the Levenhstein edit operations."""

    OP_INSERT = "insert"
    OP_DELETE = "delete"
    OP_SUBSTITUTE = "substitute"
    OP_NOTHING = "nothing"
    OP_UNDEFINED = "undefined"


class _EditOperationsCost(IntEnum):
    """Enumerations for the Levenhstein edit operation costs."""

    OP_INSERT = 1
    OP_DELETE = 1
    OP_SUBSTITUTE = 1
    OP_NOTHING = 0
    OP_UNDEFINED = _INT_INFINITY


class _LevenshteinEditDistance:
    """A convenience class for calculating the Levenshtein edit distance.

    Class will cache some intermediate values to hasten the calculation. The implementation follows the implemenation
    from https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/lib_ter.py, where the most of this
    implementation is adapted and copied from.
    """

    def __init__(self, reference_tokens: List[str]) -> None:
        """Initialize _LevenshteinEditDistance object.

        Args:
            reference_tokens:
                A tokenized reference sentence.
        """
        self.reference_tokens = reference_tokens
        self.reference_len = len(reference_tokens)

        self.cache: Dict[str, Tuple[int, str]] = {}
        self.cache_size = 0

    def __call__(self, prediction_tokens: List[str]) -> Tuple[int, Tuple[_EditOperations, ...]]:
        """Calculate edit distance between self._words_ref and the hypothesis. Uses cache to skip some computations.

        Args:
            prediction_tokens: A tokenized predicted sentence.

        Return:
            A tuple of a calculated edit distance and a trace of executed operations.
        """
        # Use cached edit distance for already computed words
        start_position, cached_edit_distance = self._find_cache(prediction_tokens)
        # Calculate the rest of the edit distance matrix
        edit_distance_int, edit_distance, trace = self._levenshtein_edit_distance(
            prediction_tokens, start_position, cached_edit_distance
        )
        # Update our cache with the newly calculated rows
        self._add_cache(prediction_tokens, edit_distance)

        return edit_distance_int, trace

    def _levenshtein_edit_distance(
        self,
        prediction_tokens: List[str],
        prediction_start: int,
        cache: List[List[Tuple[int, _EditOperations]]],
    ) -> Tuple[int, List[List[Tuple[int, _EditOperations]]], Tuple[_EditOperations, ...]]:
        """Dynamic programming algorithm to compute the Levenhstein edit distance.

        Args:
            prediction_tokens: A tokenized predicted sentence.
            prediction_start: An index where a predicted sentence to be considered from.
            cache: A cached Levenshtein edit distance.

        Returns:
            Edit distance between the predicted sentence and the reference sentence
        """
        prediction_len = len(prediction_tokens)

        empty_rows: List[List[Tuple[int, _EditOperations]]] = [
            list(self._get_empty_row(self.reference_len)) for _ in range(prediction_len - prediction_start)
        ]
        edit_distance: List[List[Tuple[int, _EditOperations]]] = cache + empty_rows
        length_ratio = self.reference_len / prediction_len if prediction_tokens else 1.0

        # Ensure to not end up with zero overlaip with previous role
        beam_width = math.ceil(length_ratio / 2 + _BEAM_WIDTH) if length_ratio / 2 > _BEAM_WIDTH else _BEAM_WIDTH

        # Calculate the Levenshtein distance
        for i in range(prediction_start + 1, prediction_len + 1):
            pseudo_diag = math.floor(i * length_ratio)
            min_j = max(0, pseudo_diag - beam_width)
            max_j = (
                self.reference_len + 1 if i == prediction_len else min(self.reference_len + 1, pseudo_diag + beam_width)
            )

            for j in range(min_j, max_j):
                if j == 0:
                    edit_distance[i][j] = (
                        edit_distance[i - 1][j][0] + _EditOperationsCost.OP_DELETE,
                        _EditOperations.OP_DELETE,
                    )
                else:
                    if prediction_tokens[i - 1] == self.reference_tokens[j - 1]:
                        cost_substitute = _EditOperationsCost.OP_NOTHING
                        operation_substitute = _EditOperations.OP_NOTHING
                    else:
                        cost_substitute = _EditOperationsCost.OP_SUBSTITUTE
                        operation_substitute = _EditOperations.OP_SUBSTITUTE

                    # Tercom prefers no-op/sub, then insertion, then deletion. But since we flip the trace and compute
                    # the alignment from the inverse, we need to swap order of insertion and  deletion in the
                    # preference.
                    # Copied from: https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/ter.py.
                    operations = (
                        (edit_distance[i - 1][j - 1][0] + cost_substitute, operation_substitute),
                        (edit_distance[i - 1][j][0] + _EditOperationsCost.OP_DELETE, _EditOperations.OP_DELETE),
                        (edit_distance[i][j - 1][0] + _EditOperationsCost.OP_INSERT, _EditOperations.OP_INSERT),
                    )

                    for operation_cost, operation_name in operations:
                        if edit_distance[i][j][0] > operation_cost:
                            edit_distance[i][j] = operation_cost, operation_name

        trace = self._get_trace(prediction_len, edit_distance)

        return edit_distance[-1][-1][0], edit_distance[len(cache) :], trace

    def _get_trace(
        self, prediction_len: int, edit_distance: List[List[Tuple[int, _EditOperations]]]
    ) -> Tuple[_EditOperations, ...]:
        """Get a trace of executed operations from the edit distance matrix.

        Args:
            prediction_len: A length of a tokenized predicted sentence.
            edit_distance:
                A matrix of the Levenshtedin edit distance. The element part of the matrix is a tuple of an edit
                operation cost and an edit operation itself.

        Return:
            A trace of executed operations returned as a tuple of `_EDIT_OPERATIONS` enumerates.

        Raises:
            ValueError:
                If an unknown operation has been applied.
        """
        trace: Tuple[_EditOperations, ...] = ()
        i = prediction_len
        j = self.reference_len

        while i > 0 or j > 0:
            operation = edit_distance[i][j][1]
            trace = (operation, *trace)
            if operation in (_EditOperations.OP_SUBSTITUTE, _EditOperations.OP_NOTHING):
                i -= 1
                j -= 1
            elif operation == _EditOperations.OP_INSERT:
                j -= 1
            elif operation == _EditOperations.OP_DELETE:
                i -= 1
            else:
                raise ValueError(f"Unknown operation {operation!r}")

        return trace

    def _add_cache(self, prediction_tokens: List[str], edit_distance: List[List[Tuple[int, _EditOperations]]]) -> None:
        """Add newly computed rows to cache.

        Since edit distance is only calculated on the hypothesis suffix that was not in cache, the number of rows in
        `edit_distance` matrx may be shorter than hypothesis length. In that case we skip over these initial words.

        Args:
            prediction_tokens: A tokenized predicted sentence.
            edit_distance:
                A matrix of the Levenshtedin edit distance. The element part of the matrix is a tuple of an edit
                operation cost and an edit operation itself.
        """
        if self.cache_size >= _MAX_CACHE_SIZE:
            return

        node = self.cache

        # how many initial words to skip
        skip_num = len(prediction_tokens) - len(edit_distance)

        # Jump through the cache to the current position
        for i in range(skip_num):
            node = node[prediction_tokens[i]][0]  # type: ignore

        # Update cache with newly computed rows
        for word, row in zip(prediction_tokens[skip_num:], edit_distance):
            if word not in node:
                node[word] = ({}, tuple(row))  # type: ignore
                self.cache_size += 1
            value = node[word]
            node = value[0]  # type: ignore

    def _find_cache(self, prediction_tokens: List[str]) -> Tuple[int, List[List[Tuple[int, _EditOperations]]]]:
        """Find the already calculated rows of the Levenshtein edit distance matric.

        Args:
            prediction_tokens: A tokenized predicted sentence.

        Return:
            A tuple of a start hypothesis position and `edit_distance` matrix.

            prediction_start: An index where a predicted sentence to be considered from.
            edit_distance:
                A matrix of the cached Levenshtedin edit distance. The element part of the matrix is a tuple of an edit
                operation cost and an edit operation itself.
        """
        node = self.cache
        start_position = 0
        edit_distance: List[List[Tuple[int, _EditOperations]]] = [self._get_initial_row(self.reference_len)]
        for word in prediction_tokens:
            if word in node:
                start_position += 1
                node, row = node[word]  # type: ignore
                edit_distance.append(row)  # type: ignore
            else:
                break

        return start_position, edit_distance

    @staticmethod
    def _get_empty_row(length: int) -> List[Tuple[int, _EditOperations]]:
        """Precomputed empty matrix row for Levenhstein edit distance.

        Args:
            length: A length of a tokenized sentence.

        Return:
            A list of tuples containing infinite edit operation costs and yet undefined edit operations.
        """
        return [(int(_EditOperationsCost.OP_UNDEFINED), _EditOperations.OP_UNDEFINED)] * (length + 1)

    @staticmethod
    def _get_initial_row(length: int) -> List[Tuple[int, _EditOperations]]:
        """First row corresponds to insertion operations of the reference, so we do 1 edit operation per reference word.

        Args:
            length: A length of a tokenized sentence.

        Return:
            A list of tuples containing edit operation costs of insert and insert edit operations.
        """
        return [(i * _EditOperationsCost.OP_INSERT, _EditOperations.OP_INSERT) for i in range(length + 1)]


def _validate_inputs(
    ref_corpus: Union[Sequence[str], Sequence[Sequence[str]]],
    hypothesis_corpus: Union[str, Sequence[str]],
) -> Tuple[Sequence[Sequence[str]], Sequence[str]]:
    """Check and update (if needed) the format of reference and hypothesis corpora for various text evaluation metrics.

    Args:
        ref_corpus: An iterable of iterables of reference corpus.
        hypothesis_corpus: An iterable of hypothesis corpus.

    Return:
        ref_corpus: An iterable of iterables of reference corpus.
        hypothesis_corpus: An iterable of hypothesis corpus.

    Raises:
        ValueError:
            If length of `ref_corpus` and `hypothesis_corpus` differs.
    """
    if isinstance(hypothesis_corpus, str):
        hypothesis_corpus = [hypothesis_corpus]

    # Ensure reference corpus is properly of a type Sequence[Sequence[str]]
    if all(isinstance(ref, str) for ref in ref_corpus):
        ref_corpus = [ref_corpus] if len(hypothesis_corpus) == 1 else [[ref] for ref in ref_corpus]  # type: ignore

    if hypothesis_corpus and all(ref for ref in ref_corpus) and len(ref_corpus) != len(hypothesis_corpus):
        raise ValueError(f"Corpus has different size {len(ref_corpus)} != {len(hypothesis_corpus)}")

    return ref_corpus, hypothesis_corpus


def _edit_distance(prediction_tokens: List[str], reference_tokens: List[str]) -> int:
    """Dynamic programming algorithm to compute the edit distance.

    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence
    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


def _flip_trace(trace: Tuple[_EditOperations, ...]) -> Tuple[_EditOperations, ...]:
    """Flip the trace of edit operations.

    Instead of rewriting a->b, get a recipe for rewriting b->a. Simply flips insertions and deletions.

    Args:
        trace: A tuple of edit operations.

    Return:
        inverted_trace:
            A tuple of inverted edit operations.
    """
    _flip_operations: Dict[_EditOperations, _EditOperations] = {
        _EditOperations.OP_INSERT: _EditOperations.OP_DELETE,
        _EditOperations.OP_DELETE: _EditOperations.OP_INSERT,
    }

    def _replace_operation_or_retain(
        operation: _EditOperations, _flip_operations: Dict[_EditOperations, _EditOperations]
    ) -> _EditOperations:
        if operation in _flip_operations:
            return _flip_operations.get(operation)  # type: ignore
        return operation

    return tuple(_replace_operation_or_retain(operation, _flip_operations) for operation in trace)


def _trace_to_alignment(trace: Tuple[_EditOperations, ...]) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Transform trace of edit operations into an alignment of the sequences.

    Args:
        trace: A trace of edit operations as a tuple of `_EDIT_OPERATIONS` enumerates.

    Return:
        alignments: A dictionary mapping aligned positions between a reference and a hypothesis.
        reference_errors: A list of error positions in a reference.
        hypothesis_errors: A list of error positions in a hypothesis.

    Raises:
        ValueError:
            If an unknown operation is
    """
    reference_position = hypothesis_position = -1
    reference_errors: List[int] = []
    hypothesis_errors: List[int] = []
    alignments: Dict[int, int] = {}

    # we are rewriting a into b
    for operation in trace:
        if operation == _EditOperations.OP_NOTHING:
            hypothesis_position += 1
            reference_position += 1
            alignments[reference_position] = hypothesis_position
            reference_errors.append(0)
            hypothesis_errors.append(0)
        elif operation == _EditOperations.OP_SUBSTITUTE:
            hypothesis_position += 1
            reference_position += 1
            alignments[reference_position] = hypothesis_position
            reference_errors.append(1)
            hypothesis_errors.append(1)
        elif operation == _EditOperations.OP_INSERT:
            hypothesis_position += 1
            hypothesis_errors.append(1)
        elif operation == _EditOperations.OP_DELETE:
            reference_position += 1
            alignments[reference_position] = hypothesis_position
            reference_errors.append(1)
        else:
            raise ValueError(f"Unknown operation {operation!r}.")

    return alignments, reference_errors, hypothesis_errors
