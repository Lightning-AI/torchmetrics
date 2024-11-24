.. customcarditem::
   :header: Mean Intersection over Union (mIoU)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/object_detection.svg
   :tags: Segmentation

###################################
Mean Intersection over Union (mIoU)
###################################

Module Interface
________________

.. autoclass:: torchmetrics.segmentation.MeanIoU
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.segmentation.mean_iou
