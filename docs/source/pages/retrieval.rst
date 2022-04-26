***********************
Using Retrieval Metrics
***********************

Input details
~~~~~~~~~~~~~

For the purposes of retrieval metrics, inputs (indexes, predictions and targets) must have the same size
(``N`` stands for the batch size) and the following types:

.. csv-table::
    :header: "indexes shape", "indexes dtype", "preds shape", "preds dtype", "target shape", "target dtype"
    :widths: 10, 10, 10, 10, 10, 10

    "(N,...)", "``long``", "(N,...)", "``float``", "(N,...)", "``long`` or ``bool``"

.. note::
    All dimensions are flattened at the beginning, so
    that, for example, a tensor of shape ``(N, M)`` is treated as ``(N * M, )``.

In Information Retrieval you have a query that is compared with a variable number of documents. For each pair ``(Q_i, D_j)``,
a score is computed that measures the relevance of document ``D`` w.r.t. query ``Q``. Documents are then sorted by score
and you hope that relevant documents are scored higher. ``target`` contains the labels for the documents (relevant or not).

Since a query may be compared with a variable number of documents, we use ``indexes`` to keep track of which scores belong to
the set of pairs ``(Q_i, D_j)`` having the same query ``Q_i``.

.. note::
    `Retrieval` metrics are only intended to be used globally. This means that the average of the metric over each batch can be quite different
    from the metric computed on the whole dataset. For this reason, we suggest to compute the metric only when all the examples
    has been provided to the metric. When using `Pytorch Lightning`, we suggest to use ``on_step=False``
    and ``on_epoch=True`` in ``self.log`` or to place the metric calculation in ``training_epoch_end``, ``validation_epoch_end`` or ``test_epoch_end``.

.. doctest::

    >>> from torchmetrics import RetrievalMAP
    >>> # functional version works on a single query at a time
    >>> from torchmetrics.functional import retrieval_average_precision

    >>> # the first query was compared with two documents, the second with three
    >>> indexes = torch.tensor([0, 0, 1, 1, 1])
    >>> preds = torch.tensor([0.8, -0.4, 1.0, 1.4, 0.0])
    >>> target = torch.tensor([0, 1, 0, 1, 1])

    >>> rmap = RetrievalMAP() # or some other retrieval metric
    >>> rmap(preds, target, indexes=indexes)
    tensor(0.6667)

    >>> # the previous instruction is roughly equivalent to
    >>> res = []
    >>> # iterate over indexes of first and second query
    >>> for indexes in ([0, 1], [2, 3, 4]):
    ...     res.append(retrieval_average_precision(preds[indexes], target[indexes]))
    >>> torch.stack(res).mean()
    tensor(0.6667)
