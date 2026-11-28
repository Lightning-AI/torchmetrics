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

from torchmetrics.functional.text import bert_score
from torchmetrics.text import BERTScore

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


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


@pytest.mark.parametrize("api", ["functional", "class"])
@pytest.mark.parametrize("all_layers", [False, True])
@pytest.mark.parametrize("references", [["cat bird"], ["cat bird", "fish dog"]])
def test_single_prediction_reference_group(local_masked_lm, api, all_layers, references):
    """A single reference group must keep each layer and match flat pair scoring."""
    kwargs = {
        "model_name_or_path": str(local_masked_lm),
        "num_layers": 1,
        "all_layers": all_layers,
        "max_length": 8,
        "verbose": False,
    }
    flat = bert_score(["cat dog"] * len(references), references, **kwargs)
    if api == "functional":
        actual = bert_score(["cat dog"], [references], **kwargs)
    else:
        metric = BERTScore(**kwargs)
        metric.update(["cat dog"], [references])
        actual = metric.compute()
    for key, value in actual.items():
        expected = flat[key].reshape(-1, len(references)).max(dim=-1).values
        expected = expected.unsqueeze(-1) if all_layers else expected
        if api == "class" and expected.numel() == 1:
            expected = expected.squeeze()
        torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("all_layers", [False, True])
def test_reference_group_maxima(all_layers):
    """Keep maxima within each reference group and each model layer."""
    from torchmetrics.functional.text.bert import _postprocess_multiple_references

    scores = torch.tensor([[0.2, 0.7, 0.3], [0.9, 0.1, 0.8]]) if all_layers else torch.tensor([0.2, 0.7, 0.3])
    expected = torch.tensor([[0.7, 0.3], [0.9, 0.8]]) if all_layers else torch.tensor([0.7, 0.3])
    for actual in _postprocess_multiple_references(scores, scores, scores, [(0, 2), (2, 3)], all_layers):
        torch.testing.assert_close(actual, expected)
