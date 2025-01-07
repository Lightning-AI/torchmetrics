"""
CLIPScore
===============================

The CLIPScore is a model-based image captioning metric that correlates well with human judgments.

The benefit of CLIPScore is that it does not require reference captions for evaluation.
"""

# %%
# Here's a hypothetical Python example demonstrating the usage of the CLIPScore metric to evaluate image captions:
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.table import Table
from skimage.data import astronaut, cat, coffee

from torchmetrics.multimodal import CLIPScore

# %%
# Get sample images

images = {
    "astronaut": astronaut(),
    "cat": cat(),
    "coffee": coffee(),
}

# %%
# Define a hypothetical captions for the images

captions = [
    "A photo of an astronaut.",
    "A photo of a cat.",
    "A photo of a cup of coffee.",
]

# %%
# Define the models for CLIPScore

models = [
    "openai/clip-vit-base-patch16",
    # "openai/clip-vit-base-patch32",
    # "openai/clip-vit-large-patch14-336",
    "openai/clip-vit-large-patch14",
]

# %%
# Collect scores for each image-caption pair

score_results = []
for model in models:
    clip_score = CLIPScore(model_name_or_path=model)
    for key, img in images.items():
        img_tensor = torch.tensor(np.array(img))
        caption_scores = {caption: clip_score(img_tensor, caption) for caption in captions}
        score_results.append({"scores": caption_scores, "image": key, "model": model})

# %%
# Create an animation to display the scores

fig, (ax_img, ax_table) = plt.subplots(1, 2, figsize=(10, 5))


def update(num: int) -> tuple:
    """Update the image and table with the scores for the given model."""
    results = score_results[num]
    scores, image, model = results["scores"], results["image"], results["model"]

    fig.suptitle(f"Model: {model.split('/')[-1]}", fontsize=16, fontweight="bold")

    # Update image
    ax_img.imshow(images[image])
    ax_img.axis("off")

    # Update table
    table = Table(ax_table, bbox=[0, 0, 1, 1])
    header1 = table.add_cell(0, 0, text="Caption", width=3, height=1)
    header2 = table.add_cell(0, 1, text="Score", width=1, height=1)
    header1.get_text().set_weight("bold")
    header2.get_text().set_weight("bold")
    for i, (caption, score) in enumerate(scores.items()):
        table.add_cell(i + 1, 0, text=caption, width=3, height=1)
        table.add_cell(i + 1, 1, text=f"{score:.2f}", width=1, height=1)
    ax_table.add_table(table)
    ax_table.axis("off")
    return ax_img, ax_table


ani = animation.FuncAnimation(fig, update, frames=len(score_results), interval=3000)
