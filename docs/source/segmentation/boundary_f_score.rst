.. customcarditem::
   :header: Boundary F-score
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/text_classification.svg
   :tags: segmentation

.. include:: ../links.rst

################
Boundary F-score
################

Boundary F-score is useful when contour alignment matters more than region overlap. Unlike Dice or IoU, it evaluates
whether predicted and target boundaries match within a configurable pixel tolerance.

When to use
___________

Boundary F-score is a good fit when small contour shifts matter. Region-overlap metrics such as Dice or IoU can stay
high even when object outlines are visibly misaligned, while Boundary F-score focuses directly on boundary agreement.

Behavior notes
______________

- ``boundary_width`` is an integer pixel tolerance for 2D masks and an integer voxel tolerance for 3D volumes.
- Classes that have no boundary pixels in either prediction or target return ``nan`` in per-class outputs.
- When reducing across classes, these absent classes are ignored instead of lowering the reduced score.

Limitations
___________

- integer pixel/voxel tolerance only
- no spacing-aware tolerance yet
- no ratio-based tolerance yet
- no BoundaryIoU yet

Future work
___________

- BoundaryIoU
- ratio-based tolerance
- spacing-aware tolerance
- comparison examples with Dice, IoU, HausdorffDistance, and Boundary F-score

Module Interface
________________

.. autoclass:: torchmetrics.segmentation.BoundaryFScore
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.segmentation.boundary_f_score
