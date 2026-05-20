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

Module Interface
________________

.. autoclass:: torchmetrics.segmentation.BoundaryFScore
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.segmentation.boundary_f_score
