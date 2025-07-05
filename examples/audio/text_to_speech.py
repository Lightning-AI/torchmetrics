"""
Perceptual Evaluation of Text-to-Speech with PESQ
==================================================

Consider a use case where we want to find the highest-quality speaker signal based on an example target voice. Using a text-to-speech model, we generate speech for five different synthetic speakers, each with unique speaker embeddings. We then compare each generated voice to a reference speaker using Perceptual Evaluation of Speech Quality (PESQ), a metric that assesses how closely the generated audio matches the target.

By ranking the PESQ scores, we identify which synthetic speaker sounds most natural and which performs the worst, providing insights into improving speech synthesis quality.
"""

# %%
# Import necessary libraries

import numpy as np
import torch
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
# Generate target audio using the target speaker embedding (512-dimensional X-vector)

# fmt: off
TARGET_EMBEDDING = torch.Tensor([
  [
    -0.075, 0.003, 0.037, 0.035, -0.005, -0.034, -0.087, 0.028, 0.041, 0.015, -0.076, -0.096, 0.052, 0.042, 0.042,
    0.054, 0.017, 0.033, 0.009, 0.02, 0.03, 0.01, -0.012, -0.033, -0.063, -0.008, -0.061, -0.011, 0.04, 0.039, -0.004,
    0.065, 0.035, -0.002, 0.053, -0.047, 0.007, 0.052, 0.002, -0.058, 0.006, -0.004, 0.041, 0.048, 0.024, -0.115,
    -0.018, 0.012, -0.07, 0.045, 0.01, 0.028, 0.034, 0.044, -0.108, -0.057, -0.009, 0.013, 0.023, 0.021, 0.002, -0.007,
    -0.016, -0.02, 0.029, 0.031, 0.031, -0.042, -0.074, -0.059, 0.005, 0.01, 0.024, 0.007, 0.027, 0.038, 0.033, -0.003,
    -0.086, -0.085, -0.07, -0.06, -0.052, -0.059, -0.032, -0.076, -0.066, 0.032, 0.032, -0.034, 0.029, -0.06, 0.02,
    -0.079, 0.05, -0.033, 0.049, 0.028, -0.078, -0.061, 0.047, -0.055, -0.107, 0.021, 0.047, 0.024, 0.07, 0.03, 0.03,
    0.038, -0.088, -0.011, 0.081, 0.008, 0.034, 0.065, -0.058, 0.02, -0.05, 0.036, 0.035, -0.059, 0.012, 0.054, -0.06,
    0.046, -0.074, 0.041, 0.035, 0.049, -0.016, 0.029, 0.029, 0.055, 0.014, -0.073, -0.061, 0.038, -0.066, -0.015,
    0.022, 0.002, -0.046, 0.058, -0.085, 0.024, 0.018, -0.021, 0.004, -0.106, 0.03, -0.05, -0.078, 0.008, 0.037, 0.041,
    0.049, -0.092, -0.073, 0.039, 0.034, 0.033, 0.025, 0.01, -0.039, 0.004, 0.013, 0.017, 0.033, 0.039, 0.012, -0.07,
    0.017, -0.074, -0.027, 0.011, -0.045, 0.016, 0.054, -0.085, 0.028, -0.057, 0.013, 0.006, -0.077, -0.012, 0.04,
    0.026, -0.07, -0.06, 0.041, 0.022, -0.066, 0.016, 0.026, 0.013, 0.032, 0.019, 0.045, -0.024, 0.046, 0.038, -0.061,
    0.013, 0.016, 0.013, 0.033, 0.027, 0.037, 0.022, 0.003, -0.065, -0.062, 0.043, -0.056, 0.042, 0.024, -0.059, 0.033,
    0.029, -0.059, -0.003, -0.069, -0.058, -0.055, 0.041, 0.058, 0.077, 0.063, 0.03, -0.025, 0.048, 0.047, -0.02, 0.028,
    -0.009, 0.05, -0.002, 0.004, 0.054, -0.07, 0.02, -0.087, 0.004, -0.068, 0.029, 0.042, 0.032, 0.033, 0.035, 0.05,
    0.013, 0.007, -0.06, 0.015, 0.041, 0.033, 0.037, -0.066, 0.069, 0.007, -0.059, 0.059, 0.027, -0.001, 0.046, 0.032,
    0.043, 0.029, 0.01, 0.029, 0.001, -0.027, 0.013, -0.079, 0.024, 0.026, 0.041, -0.064, -0.048, -0.009, 0.024, 0.041,
    -0.079, 0.029, 0.052, 0.006, 0.033, -0.104, 0.004, 0.019, 0.012, 0.045, -0.055, 0.034, 0.002, 0.028, -0.026, 0.03,
    0.025, -0.039, 0.047, 0.022, -0.074, 0.012, 0.039, 0.014, 0.02, 0.035, 0.048, 0.032, 0.021, -0.005, 0.033, -0.088,
    -0.058, -0.019, 0.01, -0.067, 0.045, -0.044, 0.027, -0.035, 0.008, 0.034, -0.074, 0.038, 0.049, -0.044, -0.093,
    -0.046, 0.004, 0.021, 0.041, -0.066, 0.05, 0.044, 0.005, -0.025, 0.03, 0.016, -0.05, 0.015, 0.015, -0.067, 0.029,
    0.051, 0.028, -0.062, -0.067, -0.054, 0.009, -0.056, 0.099, 0.024, -0.045, -0.005, 0.038, -0.043, 0.033, -0.097,
    0.025, -0.002, 0.041, 0.048, 0.017, -0.063, 0.003, 0.01, 0.026, 0.006, 0.036, -0.058, 0.026, -0.015, -0.002, 0.042,
    0.022, 0.041, 0.03, -0.073, -0.113, 0.047, 0.017, 0.02, 0.017, 0.034, -0.056, 0.028, 0.065, 0.02, 0.026, -0.023,
    0.051, -0.004, -0.013, 0.038, -0.071, -0.001, -0.01, 0.027, -0.046, -0.032, 0.009, 0.005, 0.01, 0.005, -0.059,
    -0.047, -0.081, -0.049, 0.024, 0.001, -0.01, 0.038, -0.054, -0.004, -0.081, -0.134, -0.02, -0.065, 0.003, 0.024,
    -0.01, -0.062, 0.038, 0.06, 0.035, 0.015, -0.043, -0.041, -0.011, -0.021, 0.031, 0.026, 0.017, 0.052, 0.02, 0.028,
    -0.077, 0.025, 0.029, 0.032, 0.002, -0.033, 0.008, 0.03, 0.005, -0.01, -0.01, 0.048, 0.036, 0.027, 0.026, 0.013,
    0.029, 0.02, -0.072, -0.052, 0.02, -0.011, 0.007, 0.059, 0.06, -0.079, 0.047, 0.032, -0.04, 0.04, 0.044, -0.002,
    0.009, 0.02, 0.005, -0.043, -0.068, 0.006, -0.005, 0.048, 0.065, -0.062, -0.061, 0.006, 0.035, 0.035, 0.042, -0.053,
    0.047, -0.057, -0.011, -0.039, 0.044, -0.04, 0.019, -0.005, 0.004, -0.056, -0.015, -0.071, -0.063, 0.008, 0.064,
    -0.069, 0.055, 0.04, -0.014, -0.031, 0.027, 0.029, -0.028, 0.025, -0.074
  ]
])
# fmt: on
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
