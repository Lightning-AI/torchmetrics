# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
import time
from copy import deepcopy

import pytest
import torch

from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    CohenKappa,
    ConfusionMatrix,
    F1Score,
    MatthewsCorrCoef,
    Metric,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.utilities.checks import _allclose_recursive
from unittests.helpers import seed_all
from unittests.helpers.testers import DummyMetricDiff, DummyMetricSum

seed_all(42)


def test_metric_collection(tmpdir):
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])

    # Test correct dict structure
    assert len(metric_collection) == 2
    assert metric_collection["DummyMetricSum"] == m1
    assert metric_collection["DummyMetricDiff"] == m2

    # Test correct initialization
    for name, metric in metric_collection.items():
        assert metric.x == 0, f"Metric {name} not initialized correctly"

    # Test every metric gets updated
    metric_collection.update(5)
    for name, metric in metric_collection.items():
        assert metric.x.abs() == 5, f"Metric {name} not updated correctly"

    # Test compute on each metric
    metric_collection.update(-5)
    metric_vals = metric_collection.compute()
    assert len(metric_vals) == 2
    for name, metric_val in metric_vals.items():
        assert metric_val == 0, f"Metric {name}.compute not called correctly"

    # Test that everything is reset
    for name, metric in metric_collection.items():
        assert metric.x == 0, f"Metric {name} not reset correctly"

    # Test pickable
    metric_pickled = pickle.dumps(metric_collection)
    metric_loaded = pickle.loads(metric_pickled)
    assert isinstance(metric_loaded, MetricCollection)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_device_and_dtype_transfer_metriccollection(tmpdir):
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])
    for _, metric in metric_collection.items():
        assert metric.x.is_cuda is False
        assert metric.x.dtype == torch.float32

    metric_collection = metric_collection.to(device="cuda")
    for _, metric in metric_collection.items():
        assert metric.x.is_cuda

    metric_collection = metric_collection.double()
    for _, metric in metric_collection.items():
        assert metric.x.dtype == torch.float64

    metric_collection = metric_collection.half()
    for _, metric in metric_collection.items():
        assert metric.x.dtype == torch.float16


def test_metric_collection_wrong_input(tmpdir):
    """Check that errors are raised on wrong input."""
    dms = DummyMetricSum()

    # Not all input are metrics (list)
    with pytest.raises(ValueError):
        _ = MetricCollection([dms, 5])

    # Not all input are metrics (dict)
    with pytest.raises(ValueError):
        _ = MetricCollection({"metric1": dms, "metric2": 5})

    # Same metric passed in multiple times
    with pytest.raises(ValueError, match="Encountered two metrics both named *."):
        _ = MetricCollection([dms, dms])

    # Not a list or dict passed in
    with pytest.warns(Warning, match=" which are not `Metric` so they will be ignored."):
        _ = MetricCollection(dms, [dms])


def test_metric_collection_args_kwargs(tmpdir):
    """Check that args and kwargs gets passed correctly in metric collection, Checks both update and forward
    method."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])

    # args gets passed to all metrics
    metric_collection.update(5)
    assert metric_collection["DummyMetricSum"].x == 5
    assert metric_collection["DummyMetricDiff"].x == -5
    metric_collection.reset()
    _ = metric_collection(5)
    assert metric_collection["DummyMetricSum"].x == 5
    assert metric_collection["DummyMetricDiff"].x == -5
    metric_collection.reset()

    # kwargs gets only passed to metrics that it matches
    metric_collection.update(x=10, y=20)
    assert metric_collection["DummyMetricSum"].x == 10
    assert metric_collection["DummyMetricDiff"].x == -20
    metric_collection.reset()
    _ = metric_collection(x=10, y=20)
    assert metric_collection["DummyMetricSum"].x == 10
    assert metric_collection["DummyMetricDiff"].x == -20


@pytest.mark.parametrize(
    "prefix, postfix",
    [
        [None, None],
        ["prefix_", None],
        [None, "_postfix"],
        ["prefix_", "_postfix"],
    ],
)
def test_metric_collection_prefix_postfix_args(prefix, postfix):
    """Test that the prefix arg alters the keywords in the output."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()
    names = ["DummyMetricSum", "DummyMetricDiff"]
    names = [prefix + n if prefix is not None else n for n in names]
    names = [n + postfix if postfix is not None else n for n in names]

    metric_collection = MetricCollection([m1, m2], prefix=prefix, postfix=postfix)

    # test forward
    out = metric_collection(5)
    for name in names:
        assert name in out, "prefix or postfix argument not working as intended with forward method"

    # test compute
    out = metric_collection.compute()
    for name in names:
        assert name in out, "prefix or postfix argument not working as intended with compute method"

    # test clone
    new_metric_collection = metric_collection.clone(prefix="new_prefix_")
    out = new_metric_collection(5)
    names = [n[len(prefix) :] if prefix is not None else n for n in names]  # strip away old prefix
    for name in names:
        assert f"new_prefix_{name}" in out, "prefix argument not working as intended with clone method"

    for k, _ in new_metric_collection.items():
        assert "new_prefix_" in k

    for k in new_metric_collection.keys():
        assert "new_prefix_" in k

    for k, _ in new_metric_collection.items(keep_base=True):
        assert "new_prefix_" not in k

    for k in new_metric_collection.keys(keep_base=True):
        assert "new_prefix_" not in k

    assert isinstance(new_metric_collection.keys(keep_base=True), type(new_metric_collection.keys(keep_base=False)))
    assert isinstance(new_metric_collection.items(keep_base=True), type(new_metric_collection.items(keep_base=False)))

    new_metric_collection = new_metric_collection.clone(postfix="_new_postfix")
    out = new_metric_collection(5)
    names = [n[: -len(postfix)] if postfix is not None else n for n in names]  # strip away old postfix
    for name in names:
        assert f"new_prefix_{name}_new_postfix" in out, "postfix argument not working as intended with clone method"


def test_metric_collection_repr():
    """Test MetricCollection."""

    class A(DummyMetricSum):
        pass

    class B(DummyMetricDiff):
        pass

    m1 = A()
    m2 = B()
    metric_collection = MetricCollection([m1, m2], prefix=None, postfix=None)

    expected = "MetricCollection(\n  (A): A()\n  (B): B()\n)"
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix="a", postfix=None)

    expected = "MetricCollection(\n  (A): A()\n  (B): B(),\n  prefix=a\n)"
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix=None, postfix="a")
    expected = "MetricCollection(\n  (A): A()\n  (B): B(),\n  postfix=a\n)"
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix="a", postfix="b")
    expected = "MetricCollection(\n  (A): A()\n  (B): B(),\n  prefix=a,\n  postfix=b\n)"
    assert metric_collection.__repr__() == expected


def test_metric_collection_same_order():
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()
    col1 = MetricCollection({"a": m1, "b": m2})
    col2 = MetricCollection({"b": m2, "a": m1})
    for k1, k2 in zip(col1.keys(), col2.keys()):
        assert k1 == k2


def test_collection_add_metrics():
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    collection = MetricCollection([m1])
    collection.add_metrics({"m1_": DummyMetricSum()})
    collection.add_metrics(m2)

    collection.update(5)
    results = collection.compute()
    assert results["DummyMetricSum"] == results["m1_"] and results["m1_"] == 5
    assert results["DummyMetricDiff"] == -5


def test_collection_check_arg():
    assert MetricCollection._check_arg(None, "prefix") is None
    assert MetricCollection._check_arg("sample", "prefix") == "sample"

    with pytest.raises(ValueError, match="Expected input `postfix` to be a string, but got"):
        MetricCollection._check_arg(1, "postfix")


def test_collection_filtering():
    """Test that collections works with the kwargs argument."""

    class DummyMetric(Metric):
        full_state_update = True

        def __init__(self):
            super().__init__()

        def update(self, *args, kwarg):
            print("Entered DummyMetric")

        def compute(self):
            return

    class MyAccuracy(Metric):
        full_state_update = True

        def __init__(self):
            super().__init__()

        def update(self, preds, target, kwarg2):
            print("Entered MyAccuracy")

        def compute(self):
            return

    mc = MetricCollection([Accuracy(), DummyMetric()])
    mc2 = MetricCollection([MyAccuracy(), DummyMetric()])
    mc(torch.tensor([0, 1]), torch.tensor([0, 1]), kwarg="kwarg")
    mc2(torch.tensor([0, 1]), torch.tensor([0, 1]), kwarg="kwarg", kwarg2="kwarg2")


# function for generating
_mc_preds = torch.randn(10, 3).softmax(dim=-1)
_mc_target = torch.randint(3, (10,))
_ml_preds = torch.rand(10, 3)
_ml_target = torch.randint(2, (10, 3))


@pytest.mark.parametrize(
    "metrics, expected, preds, target",
    [
        # single metric forms its own compute group
        (Accuracy(3), {0: ["Accuracy"]}, _mc_preds, _mc_target),
        # two metrics of same class forms a compute group
        ({"acc0": Accuracy(3), "acc1": Accuracy(3)}, {0: ["acc0", "acc1"]}, _mc_preds, _mc_target),
        # two metrics from registry froms a compute group
        ([Precision(3), Recall(3)], {0: ["Precision", "Recall"]}, _mc_preds, _mc_target),
        # two metrics from different classes gives two compute groups
        ([ConfusionMatrix(3), Recall(3)], {0: ["ConfusionMatrix"], 1: ["Recall"]}, _mc_preds, _mc_target),
        # multi group multi metric
        (
            [ConfusionMatrix(3), CohenKappa(3), Recall(3), Precision(3)],
            {0: ["ConfusionMatrix", "CohenKappa"], 1: ["Recall", "Precision"]},
            _mc_preds,
            _mc_target,
        ),
        # Complex example
        (
            {
                "acc": Accuracy(3),
                "acc2": Accuracy(3),
                "acc3": Accuracy(num_classes=3, average="macro"),
                "f1": F1Score(3),
                "recall": Recall(3),
                "confmat": ConfusionMatrix(3),
            },
            {0: ["acc", "acc2", "f1", "recall"], 1: ["acc3"], 2: ["confmat"]},
            _mc_preds,
            _mc_target,
        ),
        # With list states
        (
            [AUROC(average="macro", num_classes=3), AveragePrecision(average="macro", num_classes=3)],
            {0: ["AUROC", "AveragePrecision"]},
            _mc_preds,
            _mc_target,
        ),
        # Nested collections
        (
            [
                MetricCollection(
                    AUROC(average="micro", num_classes=3),
                    AveragePrecision(average="micro", num_classes=3),
                    postfix="_micro",
                ),
                MetricCollection(
                    AUROC(average="macro", num_classes=3),
                    AveragePrecision(average="macro", num_classes=3),
                    postfix="_macro",
                ),
            ],
            {0: ["AUROC_micro", "AveragePrecision_micro", "AUROC_macro", "AveragePrecision_macro"]},
            _ml_preds,
            _ml_target,
        ),
    ],
)
class TestComputeGroups:
    @pytest.mark.parametrize(
        "prefix, postfix",
        [
            [None, None],
            ["prefix_", None],
            [None, "_postfix"],
            ["prefix_", "_postfix"],
        ],
    )
    def test_check_compute_groups_correctness(self, metrics, expected, preds, target, prefix, postfix):
        """Check that compute groups are formed after initialization and that metrics are correctly computed."""
        if isinstance(metrics, MetricCollection):
            prefix, postfix = None, None  # disable for nested collections
        m = MetricCollection(deepcopy(metrics), prefix=prefix, postfix=postfix, compute_groups=True)
        # Construct without for comparison
        m2 = MetricCollection(deepcopy(metrics), prefix=prefix, postfix=postfix, compute_groups=False)

        assert len(m.compute_groups) == len(m)
        assert m2.compute_groups == {}

        for _ in range(2):  # repeat to emulate effect of multiple epochs
            m.update(preds, target)
            m2.update(preds, target)

            for _, member in m.items():
                assert member._update_called

            assert m.compute_groups == expected
            assert m2.compute_groups == {}

            # compute groups should kick in here
            m.update(preds, target)
            m2.update(preds, target)

            for _, member in m.items():
                assert member._update_called

            # compare results for correctness
            res_cg = m.compute()
            res_without_cg = m2.compute()
            for key in res_cg.keys():
                assert torch.allclose(res_cg[key], res_without_cg[key])

            m.reset()
            m2.reset()

    @pytest.mark.parametrize("method", ["items", "values", "keys"])
    def test_check_compute_groups_items_and_values(self, metrics, expected, preds, target, method):
        """Check that whenever user call a methods that give access to the indivitual metric that state are copied
        instead of just passed by reference."""
        m = MetricCollection(deepcopy(metrics), compute_groups=True)
        m2 = MetricCollection(deepcopy(metrics), compute_groups=False)

        for _ in range(2):  # repeat to emulate effect of multiple epochs
            for _ in range(2):  # repeat to emulate effect of multiple batches
                m.update(preds, target)
                m2.update(preds, target)

            def _compare(m1, m2):
                for state in m1._defaults:
                    assert _allclose_recursive(getattr(m1, state), getattr(m2, state))
                # if states are still by reference the reset will make following metrics fail
                m1.reset()
                m2.reset()

            if method == "items":
                for (name_cg, metric_cg), (name_no_cg, metric_no_cg) in zip(m.items(), m2.items()):
                    assert name_cg == name_no_cg
                    _compare(metric_cg, metric_no_cg)
            if method == "values":
                for metric_cg, metric_no_cg in zip(m.values(), m2.values()):
                    _compare(metric_cg, metric_no_cg)
            if method == "keys":
                for key in m.keys():
                    metric_cg, metric_no_cg = m[key], m2[key]
                    _compare(metric_cg, metric_no_cg)


@pytest.mark.parametrize(
    "metrics",
    [
        {"acc0": Accuracy(3), "acc1": Accuracy(3)},
        [Precision(3), Recall(3)],
        [ConfusionMatrix(3), CohenKappa(3), Recall(3), Precision(3)],
        {
            "acc": Accuracy(3),
            "acc2": Accuracy(3),
            "acc3": Accuracy(num_classes=3, average="macro"),
            "f1": F1Score(3),
            "recall": Recall(3),
            "confmat": ConfusionMatrix(3),
        },
    ],
)
@pytest.mark.parametrize("steps", [100, 1000])
def test_check_compute_groups_is_faster(metrics, steps):
    """Check that compute groups are formed after initialization."""
    m = MetricCollection(deepcopy(metrics), compute_groups=True)
    # Construct without for comparison
    m2 = MetricCollection(deepcopy(metrics), compute_groups=False)

    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))

    start = time.time()
    for _ in range(steps):
        m.update(preds, target)
    time_cg = time.time() - start

    start = time.time()
    for _ in range(steps):
        m2.update(preds, target)
    time_no_cg = time.time() - start

    assert time_cg < time_no_cg, "using compute groups were not faster"


def test_compute_group_define_by_user():
    """Check that user can provide compute groups."""
    m = MetricCollection(
        ConfusionMatrix(3), Recall(3), Precision(3), compute_groups=[["ConfusionMatrix"], ["Recall", "Precision"]]
    )

    # Check that we are not going to check the groups in the first update
    assert m._groups_checked
    assert m.compute_groups == {0: ["ConfusionMatrix"], 1: ["Recall", "Precision"]}

    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    m.update(preds, target)
    assert m.compute()


def test_compute_on_different_dtype():
    """Check that extraction of compute groups are robust towards difference in dtype."""
    m = MetricCollection(
        [
            ConfusionMatrix(num_classes=3),
            MatthewsCorrCoef(num_classes=3),
        ]
    )
    assert not m._groups_checked
    assert m.compute_groups == {0: ["ConfusionMatrix"], 1: ["MatthewsCorrCoef"]}
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    for _ in range(2):
        m.update(preds, target)
    assert m.compute_groups == {0: ["ConfusionMatrix", "MatthewsCorrCoef"]}
    assert m.compute()


def test_error_on_wrong_specified_compute_groups():
    """Test that error is raised if user mis-specify the compute groups."""
    with pytest.raises(ValueError, match="Input Accuracy in `compute_groups`.*"):
        MetricCollection(
            ConfusionMatrix(3), Recall(3), Precision(3), compute_groups=[["ConfusionMatrix"], ["Recall", "Accuracy"]]
        )


@pytest.mark.parametrize(
    "input_collections",
    [
        [
            MetricCollection(
                [Accuracy(num_classes=3, average="macro"), Precision(num_classes=3, average="macro")], prefix="macro_"
            ),
            MetricCollection(
                [Accuracy(num_classes=3, average="micro"), Precision(num_classes=3, average="micro")], prefix="micro_"
            ),
        ],
        {
            "macro": MetricCollection(
                [Accuracy(num_classes=3, average="macro"), Precision(num_classes=3, average="macro")]
            ),
            "micro": MetricCollection(
                [Accuracy(num_classes=3, average="micro"), Precision(num_classes=3, average="micro")]
            ),
        },
    ],
)
def test_nested_collections(input_collections):
    """Test that nested collections gets flattened to a single collection."""
    metrics = MetricCollection(input_collections, prefix="valmetrics/")
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    val = metrics(preds, target)
    assert "valmetrics/macro_Accuracy" in val
    assert "valmetrics/macro_Precision" in val
    assert "valmetrics/micro_Accuracy" in val
    assert "valmetrics/micro_Precision" in val
