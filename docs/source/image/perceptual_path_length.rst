.. customcarditem::
   :header: Perceptual Path Length
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/image_classification.svg
   :tags: Image

.. include:: ../links.rst

############################
Perceptual Path Length (PPL)
############################

Module Interface
________________

.. autoclass:: torchmetrics.image.perceptual_path_length.PerceptualPathLength
    :exclude-members: update, compute

.. autoclass:: torchmetrics.image.perceptual_path_length.GeneratorType

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.image.perceptual_path_length.perceptual_path_length
