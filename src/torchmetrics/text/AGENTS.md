# AGENTS.md — Text Metrics

## Scope

- Textual generation and classification metrics (e.g., BLEU, ROUGE, perplexity-like proxies).

## Implementation Guidelines

- Tokenization:
  - Keep metric independent of tokenizer where possible; accept pre-tokenized inputs when relevant.
- Unicode & Locale:
  - Normalize text consistently; document lowercasing/stemming rules as applicable.
- Batching:
  - Accept lists of strings or token id tensors; document accepted types.

## Dependencies

- Install extras: pip install torchmetrics[text]
- Optional NLP libraries (e.g., nltk) should be imported lazily with informative errors if missing.

## Modules/Functional

- Functional: stateless, one-shot computation.
- Module: aggregate counts/statistics and compute at the end; support DDP.

## Tests

- Gold references with small, deterministic examples.
- Edge cases: empty strings, punctuation, Unicode.

## Lint & Style

- ruff check --fix ., black .
- Docstrings include input types, normalization steps, and references.
