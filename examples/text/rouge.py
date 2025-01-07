"""ROUGE
===============================

The ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric used to evaluate the quality of generated text compared to a reference text. It does so by computing the overlap between two texts, for which a subsequent precision and recall value can be computed. The ROUGE score is often used in the context of generative tasks such as text summarization and machine translation.

A major difference with Perplexity comes from the fact that ROUGE evaluates actual text, whereas Perplexity evaluates logits.
"""

# %%
# Here's a hypothetical Python example demonstrating the usage of unigram ROUGE F-score to evaluate a generative language model:

from transformers import AutoTokenizer, pipeline

from torchmetrics.text import ROUGEScore

pipe = pipeline("text-generation", model="openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# %%
# Define the prompt and target texts

prompt = "The quick brown fox"
target_text = "The quick brown fox jumps over the lazy dog."

# %%
# Generate a sample text using the GPT-2 model

sample_text = pipe(prompt, max_length=20, do_sample=True, temperature=0.1, pad_token_id=tokenizer.eos_token_id)[0][
    "generated_text"
]
print(sample_text)

# %%
# Calculate the ROUGE of the generated text

rouge = ROUGEScore()
rouge(preds=[sample_text], target=[target_text])

# %%
# By default, the ROUGE score is calculated using a whitespace tokenizer. You can also calculate the ROUGE for the tokens directly:
token_rouge = ROUGEScore(tokenizer=lambda text: tokenizer.tokenize(text))
token_rouge(preds=[sample_text], target=[target_text])

# %%
# Since ROUGE is a text-based metric, it can be used to benchmark decoding strategies. For example, you can compare temperature settings:

import matplotlib.pyplot as plt  # noqa: E402

temperatures = [x * 0.1 for x in range(1, 10)]  # Generate temperature values from 0 to 1 with a step of 0.1
n_samples = 100  # Note that a real benchmark typically requires more data

average_scores = []

for temperature in temperatures:
    sample_text = pipe(
        prompt, max_length=20, do_sample=True, temperature=temperature, pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]
    scores = [rouge(preds=[sample_text], target=[target_text])["rouge1_fmeasure"] for _ in range(n_samples)]
    average_scores.append(sum(scores) / n_samples)

# Plot the average ROUGE score for each temperature
plt.plot(temperatures, average_scores)
plt.xlabel("Generation temperature")
plt.ylabel("Average unigram ROUGE F-Score")
plt.title("ROUGE for varying temperature settings")
plt.show()
