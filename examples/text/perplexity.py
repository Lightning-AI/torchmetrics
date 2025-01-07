"""
Perplexity
===============================

Perplexity is a measure of how well a probabilistic model predicts a sample.

In the context of language modeling, perplexity equals the exponential of the cross-entropy loss. A lower perplexity score indicates that the model is more certain about its predictions.
Since Perplexity measures token probabilities, it is not suitable for evaluating decoding tasks like text generation or machine translation. Instead, it is commonly used to evaluate the logits of generative language models.
"""

# %%
# Here's a hypothetical Python example demonstrating the usage of Perplexity to evaluate a generative language model

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

from torchmetrics.text import Perplexity

# %%
# Load the GPT-2 model and tokenizer

model = AutoModelWithLMHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%
# Generate token logits for a sample text

sample_text = "The quick brown fox jumps over the lazy dog"
sample_input_ids = tokenizer.encode(sample_text, return_tensors="pt")

with torch.no_grad():
    sample_outputs = model(sample_input_ids, labels=sample_input_ids)
logits = sample_outputs.logits

# %%
# We can now calculate the perplexity of the logits

perplexity = Perplexity()
score = perplexity(preds=logits, target=sample_input_ids)
print(f"Perplexity, unshifted: {score.item()}")

# %%
# This perplexity score is suspiciously high. The cause of this is that the model labels need to be shifted to the right by one position. We can fix this by removing the first token from the logits and the last token from the target

score = perplexity(preds=logits[:, :-1], target=sample_input_ids[:, 1:])
print(f"Perplexity, shifted: {score.item()}")

# %%
# Since the perplexity equates to the exponential of the cross-entropy loss, we can verify the perplexity calculation by comparing it to the loss

cross_entropy = score
perplexity = sample_outputs.loss.exp()
print(torch.allclose(perplexity, cross_entropy))

# %%
# Be aware that sequences are often padded to ensure equal length. In such cases, the padding tokens should be ignored when calculating the perplexity. This can be achieved by specifying the `ignore_index` argument in the `Perplexity` metric

tokenizer.pad_token_id = tokenizer.eos_token_id
sample_input_ids = tokenizer.encode(sample_text, return_tensors="pt", padding="max_length", max_length=20)
with torch.no_grad():
    sample_outputs = model(sample_input_ids, labels=sample_input_ids)
logits = sample_outputs.logits

perplexity = Perplexity(ignore_index=None)
score = perplexity(preds=logits[:, :-1], target=sample_input_ids[:, 1:])
print(f"Perplexity, including padding: {score.item()}")

perplexity = Perplexity(ignore_index=tokenizer.pad_token_id)
score = perplexity(preds=logits[:, :-1], target=sample_input_ids[:, 1:])
print(f"Perplexity, ignoring padding: {score.item()}")
