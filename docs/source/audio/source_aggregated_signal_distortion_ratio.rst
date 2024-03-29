.. customcarditem::
   :header: Source Aggregated Signal-to-Distortion Ratio (SA-SDR)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/audio_classification.svg
   :tags: Audio

.. include:: ../links.rst

#####################################################
Source Aggregated Signal-to-Distortion Ratio (SA-SDR)
#####################################################

Module Interface
________________

.. autoclass:: torchmetrics.audio.sdr.SourceAggregatedSignalDistortionRatio
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.audio.sdr.source_aggregated_signal_distortion_ratio
