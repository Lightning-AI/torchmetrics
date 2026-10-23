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

from torchmetrics.functional.text import infolm
from torchmetrics.text import InfoLM

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

PREDS = ["cat", "dog bird", "fish dog cat"]
TARGETS = ["bird", "cat fish", "dog bird fish"]


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


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("idf", [False, True])
@pytest.mark.parametrize("information_measure", ["kl_divergence", "l1_distance"])
def test_infolm_batch_size_invariance(local_masked_lm, batch_size, idf, information_measure):
    """Short batches must use the same corpus IDF weights as a single full batch."""
    kwargs = {
        "model_name_or_path": str(local_masked_lm),
        "max_length": 8,
        "idf": idf,
        "information_measure": information_measure,
        "verbose": False,
        "return_sentence_level_score": True,
    }
    expected = infolm(PREDS, TARGETS, batch_size=len(PREDS), **kwargs)
    actual = infolm(PREDS, TARGETS, batch_size=batch_size, **kwargs)
    for actual_value, expected_value in zip(actual, expected):
        assert torch.isfinite(actual_value).all()
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_infolm_idf_across_updates(local_masked_lm, batch_size):
    """The class metric must keep corpus IDF after accumulating unequal-length updates."""
    kwargs = {
        "model_name_or_path": str(local_masked_lm),
        "max_length": 8,
        "verbose": False,
        "return_sentence_level_score": True,
    }
    expected = infolm(PREDS, TARGETS, batch_size=len(PREDS), **kwargs)
    metric = InfoLM(batch_size=batch_size, **kwargs)
    metric.update(PREDS[:1], TARGETS[:1])
    metric.update(PREDS[1:], TARGETS[1:])
    for actual_value, expected_value in zip(metric.compute(), expected):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-5, atol=1e-6)


def test_infolm_default_batch_size_with_long_tail(local_masked_lm):
    """Default IDF and batch size must handle a corpus whose longest sentence is in the last batch."""
    preds = [PREDS[0]] * 64 + ["fish dog fish"]
    targets = [TARGETS[0]] * 64 + ["cat fish dog"]
    kwargs = {
        "model_name_or_path": str(local_masked_lm),
        "max_length": 8,
        "verbose": False,
        "return_sentence_level_score": True,
    }
    expected = infolm(preds, targets, batch_size=len(preds), **kwargs)
    actual = infolm(preds, targets, **kwargs)
    for actual_value, expected_value in zip(actual, expected):
        assert torch.isfinite(actual_value).all()
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-5, atol=1e-6)
