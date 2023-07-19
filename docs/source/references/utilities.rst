.. role:: hidden
    :class: hidden-section

######################
torchmetrics.utilities
######################

In the following is listed public utility functions that may be beneficial to use in your own code. These functions are
not part of the public API and may change at any time.

**********************************
torchmetrics.utilities.distributed
**********************************

The `distributed` utilities are used to help with syncronization of metrics across multiple processes.

EvaluationDistributedSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: torchmetrics.utilities.distributed.EvaluationDistributedSampler
    :noindex:

gather_all_tensors
~~~~~~~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.distributed.gather_all_tensors
    :noindex:

***************************
torchmetrics.utilities.data
***************************

The `data` utilities are used to help with data manipulation, such as converting labels in classification from format
to another.

select_topk
~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.select_topk
    :noindex:

to_categorical
~~~~~~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.to_categorical
    :noindex:

to_onehot
~~~~~~~~~

.. autofunction:: torchmetrics.utilities.data.to_onehot
    :noindex:
