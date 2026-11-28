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
import torch

from torchmetrics.functional.text import bert_score, infolm
from torchmetrics.text import BERTScore, InfoLM

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

PREDS = ["cat dog bird", "fish", "dog cat"]
TARGETS = ["cat bird", "fish dog cat", "bird"]


@pytest.fixture(scope="module")
def local_masked_lm(tmp_path_factory):
    """Create an offline model with real tokenization and nontrivial contextual embeddings."""
    path = tmp_path_factory.mktemp("masked-lm")
    vocab = {
        token: idx
        for idx, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cat", "dog", "bird", "fish"])
    }
    backend = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab, unk_token="[UNK]"))  # noqa: S106
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",  # noqa: S106
        unk_token="[UNK]",  # noqa: S106
        cls_token="[CLS]",  # noqa: S106
        sep_token="[SEP]",  # noqa: S106
        mask_token="[MASK]",  # noqa: S106
        model_max_length=8,
    )
    tokenizer.save_pretrained(path)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = transformers.BertForMaskedLM(
            transformers.BertConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=8,
            )
        )
    model.save_pretrained(path)
    return path


def _score(family, preds, targets, model_path, batch_size=2, idf=False):
    """Score the original pairs through the public functions."""
    kwargs = {
        "model_name_or_path": str(model_path),
        "max_length": 8,
        "batch_size": batch_size,
        "idf": idf,
        "verbose": False,
    }
    if family == "bert":
        return {key: value.reshape(-1) for key, value in bert_score(preds, targets, num_layers=1, **kwargs).items()}
    _, scores = infolm(preds, targets, information_measure="l2_distance", return_sentence_level_score=True, **kwargs)
    return {"distance": scores}


@pytest.mark.parametrize("family", ["bert", "infolm"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_embedding_metrics_preserve_pairs(local_masked_lm, family, batch_size):
    """A three-cycle length sort must not change which prediction is paired with each target."""
    scores = _score(family, PREDS, TARGETS, local_masked_lm, batch_size)
    pairs = [_score(family, [pred], [target], local_masked_lm) for pred, target in zip(PREDS, TARGETS)]
    for key, values in scores.items():
        expected = torch.cat([pair[key] for pair in pairs])
        torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["bert", "infolm"])
@pytest.mark.parametrize("already_sorted", [False, True])
def test_embedding_metric_order_controls(local_masked_lm, family, already_sorted):
    """Keep identity and two-item permutations working as before."""
    if already_sorted:
        preds = sorted(PREDS, key=lambda text: len(text.split()))
        targets = sorted(TARGETS, key=lambda text: len(text.split()))
    else:
        preds, targets = PREDS[:2], TARGETS[:2]
    scores = _score(family, preds, targets, local_masked_lm)
    pairs = [_score(family, [pred], [target], local_masked_lm) for pred, target in zip(preds, targets)]
    for key, values in scores.items():
        torch.testing.assert_close(values, torch.cat([pair[key] for pair in pairs]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["bert", "infolm"])
@pytest.mark.parametrize("idf", [False, True])
def test_embedding_metrics_follow_input_permutation(local_masked_lm, family, idf):
    """Reordering complete pairs must reorder their scores, including corpus-level IDF weights."""
    order = [2, 0, 1]
    scores = _score(family, PREDS, TARGETS, local_masked_lm, batch_size=3, idf=idf)
    reordered = _score(
        family, [PREDS[i] for i in order], [TARGETS[i] for i in order], local_masked_lm, batch_size=3, idf=idf
    )
    for key, values in scores.items():
        torch.testing.assert_close(reordered[key], values[order], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["bert", "infolm"])
def test_embedding_metric_updates_preserve_pairs(local_masked_lm, family):
    """Accumulating two updates must preserve the same pairs as independent evaluations."""
    kwargs = {
        "model_name_or_path": str(local_masked_lm),
        "max_length": 8,
        "batch_size": 2,
        "idf": False,
        "verbose": False,
    }
    if family == "bert":
        metric = BERTScore(num_layers=1, **kwargs)
    else:
        metric = InfoLM(information_measure="l2_distance", return_sentence_level_score=True, **kwargs)
    metric.update(PREDS[:1], TARGETS[:1])
    metric.update(PREDS[1:], TARGETS[1:])
    scores = metric.compute()
    if family == "infolm":
        scores = {"distance": scores[1]}
    pairs = [_score(family, [pred], [target], local_masked_lm) for pred, target in zip(PREDS, TARGETS)]
    for key, values in scores.items():
        torch.testing.assert_close(values, torch.cat([pair[key] for pair in pairs]), rtol=1e-5, atol=1e-6)


def test_bertscore_preserves_reference_groups(local_masked_lm):
    """Restoring sentence order must happen before taking the best score in each reference group."""
    targets = [[TARGETS[0], "cat"], [TARGETS[1]], [TARGETS[2], "dog bird fish"]]
    scores = _score("bert", PREDS, targets, local_masked_lm)
    pairs = [[_score("bert", [pred], [ref], local_masked_lm) for ref in refs] for pred, refs in zip(PREDS, targets)]
    for key, values in scores.items():
        expected = torch.stack([torch.cat([pair[key] for pair in group]).max() for group in pairs])
        torch.testing.assert_close(values, expected, rtol=1e-5, atol=1e-6)
