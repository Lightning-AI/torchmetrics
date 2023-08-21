.. customcarditem::
   :header: Generalized Intersection Over Union (gIoU)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/object_detection.svg
   :tags: Detection

.. include:: ../links.rst

##########################################
Generalized Intersection Over Union (gIoU)
##########################################

Module Interface
________________

.. autoclass:: torchmetrics.detection.giou.GeneralizedIntersectionOverUnion
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.detection.giou.generalized_intersection_over_union
