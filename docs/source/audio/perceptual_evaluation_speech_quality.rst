.. customcarditem::
   :header: Perceptual Evaluation of Speech Quality (PESQ)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/audio_classification.svg
   :tags: Audio

.. include:: ../links.rst

##############################################
Perceptual Evaluation of Speech Quality (PESQ)
##############################################

Module Interface
________________

.. autoclass:: torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.audio.pesq.perceptual_evaluation_speech_quality
