"""BERTScore
===============================

BERTScore is a text generation metric to compute the similarity between a generated text and a reference text using a pre-trained BERT model. Instead of relying on exact token matches, BERTScore leverages contextual embeddings to capture the semantic similarity between the texts. This makes BERTScore robust to paraphrasing and word order variations. BERTScore has been shown to correlate well with human judgments and is widely used in evaluating text generation models.

Let's consider a use case in natural language processing where BERTScore is used to evaluate the quality of a text generation model. In this case we are imaging that we are developing a automated news summarization system. The goal is to create concise summaries of news articles that accurately capture the key points of the original articles. To evaluate the performance of your summarization system, you need a metric that can compare the generated summaries to human-written summaries. This is where the BERTScore can be used.
"""

from transformers import AutoTokenizer, pipeline

from torchmetrics.text import BERTScore, ROUGEScore

pipe = pipeline("text-generation", model="openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# %%
# Define the prompt and target texts

prompt = "Economic recovery is underway with a 3.5% GDP growth and a decrease in unemployment. Experts forecast continued improvement with boosts from consumer spending and government projects. In summary: "
target_summary = "the recession is ending."

# %%
# Generate a sample text using the GPT-2 model

generated_summary = pipe(prompt, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0][
    "generated_text"
][len(prompt) :].strip()

# %%
# Calculate the BERTScore of the generated text

bertscore = BERTScore(model_name_or_path="roberta-base")
score = bertscore(preds=[generated_summary], target=[target_summary])

print(f"Prompt: {prompt}")
print(f"Target summary: {target_summary}")
print(f"Generated summary: {generated_summary}")
print(f"BERTScore: {score['f1']:.4f}")

# %%
# In addition, to illustrate BERTScore's robustness to paraphrasing, let's consider two candidate texts that are variations of the reference text.
reference = "the weather is freezing"
candidate_good = "it is cold today"
candidate_bad = "it is warm outside"

# %%
# Here we see that using the BERTScore we are able to differentiate between the candidate texts based on their similarity to the reference text, whereas the ROUGE scores for the same text pairs are identical.
rouge = ROUGEScore()

print(f"ROUGE for candidate_good: {rouge(preds=[candidate_good], target=[reference])['rouge1_fmeasure'].item()}")
print(f"ROUGE for candidate_bad: {rouge(preds=[candidate_bad], target=[reference])['rouge1_fmeasure'].item()}")
print(f"BERTScore for candidate_good: {bertscore(preds=[candidate_good], target=[reference])['f1'].item():.4f}")
print(f"BERTScore for candidate_bad: {bertscore(preds=[candidate_bad], target=[reference])['f1'].item():.4f}")
