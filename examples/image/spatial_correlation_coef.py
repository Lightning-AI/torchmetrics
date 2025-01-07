"""
Spatial Correlation Coefficient
===============================

The Spatial Correlation Coefficient can be applied to compare the spatial structure of two images, which can be valuable in various domains such as medical imaging, remote sensing, and quality assessment in manufacturing or design processes.

Let's consider a use case in medical imaging where Spatial Correlation Coefficient is used to compare the spatial correlation between a reference image and a reconstructed medical scan.
This can be particularly relevant in evaluating the accuracy of image reconstruction techniques or assessing the quality of medical imaging data.
"""

# %%
# Here's a hypothetical Python example demonstrating the usage of the Spatial Correlation Coefficient to compare two medical images:

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import iradon, radon, rescale

from torchmetrics.image import SpatialCorrelationCoefficient

# %%
# Create a Shepp-Logan phantom image
phantom = shepp_logan_phantom()
phantom = rescale(phantom, scale=512 / 400)  # Rescaling to 512x512

# %%
# Simulate projection data (sinogram) using Radon transform
theta = np.linspace(0.0, 180.0, max(phantom.shape), endpoint=False)
sinogram = radon(phantom, theta=theta)

# %%
# Perform reconstruction using the inverse Radon transform
reconstruction = iradon(sinogram, theta=theta, circle=True)

# %%
# Display the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
ax1.set_title("Original")
ax1.imshow(phantom, cmap=plt.cm.Greys_r)
ax2.set_title("Radon transform (Sinogram)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect="equal")
ax3.set_title("Reconstruction from sinogram")
ax3.imshow(reconstruction, cmap=plt.cm.Greys_r)
fig.tight_layout()

# %%
# Convert the images to PyTorch tensors
phantom_tensor = torch.from_numpy(phantom).float().unsqueeze(0).unsqueeze(0)
reconstructed_tensor = torch.from_numpy(reconstruction).float().unsqueeze(0).unsqueeze(0)

# %%
# Calculating the Spatial Correlation Coefficient
scc = SpatialCorrelationCoefficient()
score = scc(preds=reconstructed_tensor, target=phantom_tensor)

print(f"Spatial Correlation Coefficient between the images: {score}")
fig.suptitle(f"Spatial Correlation Coefficient: {score:.5}", y=-0.01)
