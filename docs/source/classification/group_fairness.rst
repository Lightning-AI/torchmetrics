.. customcarditem::
   :header: Group Fairness
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Classification

.. include:: ../links.rst

##############
Group Fairness
##############

Module Interface
________________

BinaryFairness
^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryFairness
    :exclude-members: update, compute

BinaryGroupStatRates
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchmetrics.classification.BinaryGroupStatRates
    :exclude-members: update, compute

Functional Interface
____________________

binary_fairness
^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_fairness

demographic_parity
^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.demographic_parity

equal_opportunity
^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.equal_opportunity

binary_groups_stat_rates
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: torchmetrics.functional.classification.binary_groups_stat_rates
