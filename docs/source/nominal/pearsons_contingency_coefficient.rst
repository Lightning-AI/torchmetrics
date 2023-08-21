.. customcarditem::
   :header: Pearson's Contingency Coefficient
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Nominal

.. include:: ../links.rst

#################################
Pearson's Contingency Coefficient
#################################

Module Interface
________________

.. autoclass:: torchmetrics.nominal.PearsonsContingencyCoefficient
    :exclude-members: update, compute

Functional Interface
____________________

.. autofunction:: torchmetrics.functional.nominal.pearsons_contingency_coefficient

pearsons_contingency_coefficient_matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.nominal.pearsons_contingency_coefficient_matrix
