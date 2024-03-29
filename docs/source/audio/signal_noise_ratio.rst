.. customcarditem::
   :header: Signal-to-Noise Ratio (SNR)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/audio_classification.svg
   :tags: Audio

.. include:: ../links.rst

###########################
Signal-to-Noise Ratio (SNR)
###########################

Module Interface
________________

.. autoclass:: torchmetrics.audio.SignalNoiseRatio
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.audio.signal_noise_ratio
