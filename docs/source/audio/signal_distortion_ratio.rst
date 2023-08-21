.. customcarditem::
   :header: Signal to Distortion Ratio (SDR)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/audio_classification.svg
   :tags: Audio

.. include:: ../links.rst

################################
Signal to Distortion Ratio (SDR)
################################

Module Interface
________________

.. autoclass:: torchmetrics.audio.SignalDistortionRatio
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.audio.signal_distortion_ratio
