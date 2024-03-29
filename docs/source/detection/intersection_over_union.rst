.. customcarditem::
   :header: Intersection Over Union (IoU)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/object_detection.svg
   :tags: Detection

.. include:: ../links.rst

#############################
Intersection Over Union (IoU)
#############################

Module Interface
________________

.. autoclass:: torchmetrics.detection.iou.IntersectionOverUnion
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.detection.iou.intersection_over_union
