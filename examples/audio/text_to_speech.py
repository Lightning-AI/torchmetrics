"""
Perceptual Evaluation of Text-to-Speech with PESQ
==================================================

Consider a use case where we want to find the highest-quality speaker signal based on an example target voice. Using a text-to-speech model, we generate speech for five different synthetic speakers, each with unique speaker embeddings. We then compare each generated voice to a reference speaker using Perceptual Evaluation of Speech Quality (PESQ), a metric that assesses how closely the generated audio matches the target.

By ranking the PESQ scores, we identify which synthetic speaker sounds most natural and which performs the worst, providing insights into improving speech synthesis quality.
"""

# %%
# Import necessary libraries
import json
import os

import numpy as np
import torch
from ipykernel import get_connection_file
from IPython.display import Audio
from transformers import pipeline

from torchmetrics.audio import PerceptualEvaluationSpeechQuality

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Define the test string and number of speakers
TEST_STRING = "Hello, my dog is cooler than you!"
n_speakers = 5

# Generate random speaker embeddings
speaker_embeddings = [torch.randn(1, 512) for _ in range(n_speakers)]
speaker_embeddings = [e / e.norm() for e in speaker_embeddings]  # Normalize the embeddings

# %%
# Load the text-to-speech pipeline
pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts")

# Placeholder for storing audio data
audio_fragments = []

# %%
# Synthesize speech for each speaker
for idx, e in enumerate(speaker_embeddings):
    speech = pipe(TEST_STRING, forward_params={"speaker_embeddings": e})
    audio_fragments.append((speech["audio"], speech["sampling_rate"]))
    print(f"Generated speech for speaker {idx + 1}")

# %%
# Get the target audio using the speaker embedding from the separate file

# Get the directory of the current script to locate the JSON file
json_path = os.path.join(os.path.dirname(get_connection_file()), "target_embedding.json")
with open(json_path) as f:
    embedding_list = json.load(f)
# Target speaker embedding (512-dimensional X-vector)
TARGET_EMBEDDING = torch.Tensor(embedding_list)
# Generate target audio using the target speaker embedding
target_audio = torch.Tensor(pipe(TEST_STRING, forward_params={"speaker_embeddings": TARGET_EMBEDDING})["audio"])

# %%
# Initialize PESQ metrics for wideband (16 kHz)
pesq_wb = PerceptualEvaluationSpeechQuality(16000, "wb")


# %%
# Evaluate PESQ for each generated audio fragment
pesq_results = []
audio_metadata = []

for audio, _sr in audio_fragments:
    # Pad or truncate to match the target length
    audio_tensor = torch.tensor(audio[: len(target_audio)])
    if len(audio_tensor) < len(target_audio):
        audio_tensor = torch.cat([audio_tensor, torch.zeros(len(target_audio) - len(audio_tensor))])

    # Compute PESQ
    pesq_results.append(pesq_wb(audio_tensor, target_audio).item())
    audio_metadata.append((audio, pesq_results[-1]))

# %%
# Find the best and worst PESQ scores
best_idx = np.argmax(pesq_results)
worst_idx = np.argmin(pesq_results)

best_audio, best_pesq = audio_metadata[best_idx]
worst_audio, worst_pesq = audio_metadata[worst_idx]

print(f"Best PESQ: {best_pesq} (Speaker {best_idx + 1})")
print(f"Worst PESQ: {worst_pesq} (Speaker {worst_idx + 1})")

# %%
# Display target audio playback
print("Target audio:")
Audio(target_audio, rate=16000)

# %%
# Display audio playback for the best PESQ score
print(f"Audio fragment with highest PESQ: {best_pesq}")
Audio(best_audio, rate=16000)

# %%
# Display audio playback for the worst PESQ score
print(f"Audio fragment with lowest PESQ: {worst_pesq}")
Audio(worst_audio, rate=16000)
