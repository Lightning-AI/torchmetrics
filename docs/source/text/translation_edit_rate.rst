.. customcarditem::
   :header: Translation Edit Rate (TER)
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/summarization.svg
   :tags: Text

.. include:: ../links.rst

###########################
Translation Edit Rate (TER)
###########################

Module Interface
________________

.. autoclass:: torchmetrics.text.TranslationEditRate
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.text.translation_edit_rate
