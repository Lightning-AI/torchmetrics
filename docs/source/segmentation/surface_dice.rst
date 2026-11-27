.. customcarditem::
   :header: Surface Dice Score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/text_classification.svg
   :tags: segmentation

.. include:: ../links.rst

##################
Surface Dice Score
##################

Surface Dice Score, often reported as normalized surface Dice (NSD), is useful when boundary agreement matters more
than region overlap. Unlike volumetric Dice or IoU, it measures how much of the predicted and reference boundaries lie
within an acceptable class-specific tolerance.

When to use
___________

Surface Dice Score is a strong fit for segmentation settings where small contour errors matter, such as medical
imaging, microscopy, and remote sensing. Region-overlap metrics can remain high even when boundaries are visibly
misaligned, while Surface Dice Score focuses directly on contour agreement.

Behavior notes
______________

- ``class_thresholds`` can be a single shared tolerance or one tolerance per evaluated class.
- Thresholds are interpreted in pixels or voxels when ``spacing`` is not provided.
- When ``spacing`` is provided, thresholds are interpreted in the same physical units as the spacing values.
- 3D computations currently rely on SciPy-backed distance transforms for the closest-surface lookup.
- Classes that have no boundary elements in either prediction or target return ``nan`` in per-class outputs.
- When reducing across classes, these absent classes are ignored instead of lowering the reduced score.

Limitations
___________

- Euclidean surface distance only in this first implementation
- Thresholds are absolute, not ratio-based
- No spacing-aware threshold normalization beyond direct physical-unit spacing support
- Surface overlap and average surface distance are not included yet

Future work
___________

- AverageSurfaceDistance
- surface overlap at tolerance
- ratio-based or organ-specific threshold helpers
- comparison examples with Dice, IoU, HausdorffDistance, and Surface Dice Score

Module Interface
________________

.. autoclass:: torchmetrics.segmentation.SurfaceDiceScore
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.segmentation.surface_dice_score
