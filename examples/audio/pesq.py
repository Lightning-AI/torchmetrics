"""
Evaluating Speech Quality with PESQ metric
==============================================

This notebook will guide you through calculating the Perceptual Evaluation of Speech Quality (PESQ) score,
 a key metric in assessing how effective noise reduction and enhancement techniques are in improving speech quality.
 PESQ is widely adopted in industries such as telecommunications, VoIP, and audio processing.
 It provides an objective way to measure the perceived quality of speech signals from a human listener's perspective.

Imagine being on a noisy street, trying to have a phone call. The technology behind the scenes aims
 to clean up your voice and make it sound clearer on the other end. But how do engineers measure that improvement?
 This is where PESQ comes in. In this notebook, we will simulate a similar scenario, applying a simple noise reduction
 technique and using the PESQ score to evaluate how much the speech quality improves.
"""

# %%
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from torchmetrics.audio import PerceptualEvaluationSpeechQuality

# %%
# Generate Synthetic Clean and Noisy Audio Signals
# We'll generate a clean sine wave (representing a clean speech signal) and add white noise to simulate the noisy version.


def generate_sine_wave(frequency, duration, sample_rate, amplitude: float = 0.5):
    """Generate a clean sine wave at a given frequency."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    return amplitude * torch.sin(2 * np.pi * frequency * t)


def add_noise(waveform: torch.Tensor, noise_factor: float = 0.05) -> torch.Tensor:
    """Add white noise to a waveform."""
    noise = noise_factor * torch.randn(waveform.size())
    return waveform + noise


# Parameters for the synthetic audio
sample_rate = 16000  # 16 kHz typical for speech
duration = 3  # 3 seconds of audio
frequency = 440  # A4 note, can represent a simple speech-like tone

# Generate the clean sine wave
clean_waveform = generate_sine_wave(frequency, duration, sample_rate)

# Generate the noisy waveform by adding white noise
noisy_waveform = add_noise(clean_waveform)


# %%
# Apply Basic Noise Reduction Technique
# In this step, we apply a simple spectral gating method for noise reduction using torchaudio's
# `spectrogram` method. This is to simulate the enhancement of noisy speech.


def reduce_noise(noisy_signal: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    """Basic noise reduction using spectral gating."""
    # Compute the spectrogram
    spec = torchaudio.transforms.Spectrogram()(noisy_signal)

    # Apply threshold-based gating: values below the threshold will be zeroed out
    spec_denoised = spec * (spec > threshold)

    # Convert back to the waveform
    return torchaudio.transforms.GriffinLim()(spec_denoised)


# Apply noise reduction to the noisy waveform
enhanced_waveform = reduce_noise(noisy_waveform)

# %%
# Initialize the PESQ Metric
# PESQ can be computed in two modes: 'wb' (wideband) or 'nb' (narrowband).
# Here, we are using 'wb' mode for wideband speech quality evaluation.
pesq_metric = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode="wb")

# %%
# Compute PESQ Scores
# We will calculate the PESQ scores for both the noisy and enhanced versions compared to the clean signal.
# The PESQ scores give us a numerical evaluation of how well the enhanced speech
# compares to the clean speech. Higher scores indicate better quality.

pesq_noisy = pesq_metric(clean_waveform, noisy_waveform)
pesq_enhanced = pesq_metric(clean_waveform, enhanced_waveform)

print(f"PESQ Score for Noisy Audio: {pesq_noisy.item():.4f}")
print(f"PESQ Score for Enhanced Audio: {pesq_enhanced.item():.4f}")

# %%
# Visualize the waveforms
# We can visualize the waveforms of the clean, noisy, and enhanced audio to see the differences.
fig, axs = plt.subplots(3, 1, figsize=(12, 9))

# Plot clean waveform
axs[0].plot(clean_waveform.numpy())
axs[0].set_title("Clean Audio Waveform (Sine Wave)")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Amplitude")

# Plot noisy waveform
axs[1].plot(noisy_waveform.numpy(), color="orange")
axs[1].set_title(f"Noisy Audio Waveform (PESQ: {pesq_noisy.item():.4f})")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Amplitude")

# Plot enhanced waveform
axs[2].plot(enhanced_waveform.numpy(), color="green")
axs[2].set_title(f"Enhanced Audio Waveform (PESQ: {pesq_enhanced.item():.4f})")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Amplitude")

# Adjust layout for better visualization
fig.tight_layout()
plt.show()
