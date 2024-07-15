"""BERTScore
===============================

BERTScore is a text generation metric to compute the similarity between a generated text and a reference text using a pre-trained BERT model. Instead of relying on exact token matches, BERTScore leverages contextual embeddings to capture the semantic similarity between the texts. This makes BERTScore robust to paraphrasing and word order variations. BERTScore has been shown to correlate well with human judgments and is widely used in evaluating text generation models.

Let's consider a use case in natural language processing where BERTScore is used to evaluate the quality of a text generation model.
"""

from torchmetrics.text import BERTScore, ROUGEScore
from transformers import AutoTokenizer, pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# %%
# Define the prompt and target texts

prompt = "The quick brown fox jumps"
target_text = " over the lazy dog."

# %%
# Generate a sample text using the GPT-2 model

generated_text = pipe(prompt, max_new_tokens=20, do_sample=False, temperature=0, pad_token_id=tokenizer.eos_token_id)[
    0
]["generated_text"][len(prompt) :]

# %%
# Calculate the BERTScore of the generated text

bertscore = BERTScore(model_name_or_path="roberta-base")
score = bertscore(preds=[generated_text], target=[target_text])

print(f"Prompt: {prompt}")
print(f"Target Text: {target_text}")
print(f"Generated Text: {generated_text}")
print(f"BERTScore: {score['f1']}")

# %%
# In addition, to illustrate BERTScore's robustness to paraphrasing, let's consider two candidate texts that are variations of the reference text.
reference = "the weather is freezing"
candidate_good = "it is cold today"
candidate_bad = "it is warm outside"

rouge = ROUGEScore()
bertscore = BERTScore(model_name_or_path="roberta-base")

print("ROUGE for candidate_good:", rouge(preds=[candidate_good], target=[reference])["rouge1_fmeasure"])
print("ROUGE for candidate_bad:", rouge(preds=[candidate_bad], target=[reference])["rouge1_fmeasure"])
print("BERTScore for candidate_good:", bertscore(preds=[candidate_good], target=[reference])["f1"])
print("BERTScore for candidate_bad:", bertscore(preds=[candidate_bad], target=[reference])["f1"])
