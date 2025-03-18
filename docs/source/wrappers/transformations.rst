.. customcarditem::
   :header: Transformations
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/graph_classification.svg
   :tags: Wrappers

.. include:: ../links.rst

###############
Transformations
###############

Transformations allow for modifications to the input a metric receives by wrapping its `pred` and `target` arguments.
Transformations can be implemented by either subclassing the ``MetricInputTransformer`` base class and overriding the ``.transform_pred()`` and/or ``transform_target()`` functions, or by supplying a lambda function via the ``LambdaInputTransformer``.
A ``BinaryTargetTransformer`` which casts target labels to 0/1 given a threshold is provided for convenience.

Module Interface
________________

.. autoclass:: torchmetrics.wrappers.MetricInputTransformer

.. autoclass:: torchmetrics.wrappers.LambdaInputTransformer

.. autoclass:: torchmetrics.wrappers.BinaryTargetTransformer
