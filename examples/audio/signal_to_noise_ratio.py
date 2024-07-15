"""Signal-to-Noise Ratio
===============================

The Signal-to-Noise Ratio (SNR) is a metric used to evaluate the quality of a signal by comparing the power of the signal to the power of background noise. In audio processing, SNR can be used to measure the quality of a reconstructed audio signal by comparing it to the original clean signal.
"""

# %%
# Here's a hypothetical Python example demonstrating the usage of the Signal-to-Noise Ratio to evaluate an audio reconstruction task:

from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.audio import SignalNoiseRatio

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# %%
# Create a clean signal (sine wave)
def generate_clean_signal(length: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a clean signal (sine wave)"""
    t = np.linspace(0, 1, length)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    return t, signal


# %%
# Add Gaussian noise to the signal
def add_noise(signal: np.ndarray, noise_level: float = 0.5) -> np.ndarray:
    """Add Gaussian noise to the signal."""
    noise = noise_level * np.random.randn(signal.shape[0])
    return signal + noise


# %%
# Generate and plot clean and noisy signals
length = 1000
t, clean_signal = generate_clean_signal(length)
noisy_signal = add_noise(clean_signal, noise_level=0.5)

plt.figure(figsize=(12, 4))
plt.plot(t, noisy_signal, label="Noisy Signal", color="blue", alpha=0.7)
plt.plot(t, clean_signal, label="Clean Signal", color="red", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Clean Signal vs. Noisy Signal")
plt.legend()
plt.show()


# %%
# Convert the signals to PyTorch tensors and calculate the SNR
clean_signal_tensor = torch.tensor(clean_signal).float()
noisy_signal_tensor = torch.tensor(noisy_signal).float()

snr = SignalNoiseRatio()
score = snr(preds=noisy_signal_tensor, target=clean_signal_tensor)

# %%
# To show the effect of different noise levels on the SNR, we can create an animation that iterates over different noise levels and updates the plot accordingly:
fig, ax = plt.subplots(figsize=(12, 4))
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def update(num: int) -> tuple:
    """Update the plot for each frame."""
    t, clean_signal = generate_clean_signal(length)
    noisy_signal = add_noise(clean_signal, noise_level=noise_levels[num])

    clean_signal_tensor = torch.tensor(clean_signal).float()
    noisy_signal_tensor = torch.tensor(noisy_signal).float()
    score = snr(preds=noisy_signal_tensor, target=clean_signal_tensor)

    ax.clear()
    (clean,) = plt.plot(t, noisy_signal, label="Noisy Signal", color="blue", alpha=0.7)
    (noisy,) = plt.plot(t, clean_signal, label="Clean Signal", color="red", linewidth=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"SNR: {score:.2f} - Noise level: {noise_levels[num]}")
    ax.legend()
    ax.set_ylim(-3, 3)
    return clean, noisy


ani = animation.FuncAnimation(fig, update, frames=len(noise_levels), interval=500)
plt.show()
