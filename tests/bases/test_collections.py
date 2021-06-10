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

import pytest
import torch

from tests.helpers import seed_all
from tests.helpers.testers import DummyMetricDiff, DummyMetricSum
from torchmetrics.collections import MetricCollection

seed_all(42)


def test_metric_collection(tmpdir):
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])

    # Test correct dict structure
    assert len(metric_collection) == 2
    assert metric_collection['DummyMetricSum'] == m1
    assert metric_collection['DummyMetricDiff'] == m2

    # Test correct initialization
    for name, metric in metric_collection.items():
        assert metric.x == 0, f'Metric {name} not initialized correctly'

    # Test every metric gets updated
    metric_collection.update(5)
    for name, metric in metric_collection.items():
        assert metric.x.abs() == 5, f'Metric {name} not updated correctly'

    # Test compute on each metric
    metric_collection.update(-5)
    metric_vals = metric_collection.compute()
    assert len(metric_vals) == 2
    for name, metric_val in metric_vals.items():
        assert metric_val == 0, f'Metric {name}.compute not called correctly'

    # Test that everything is reset
    for name, metric in metric_collection.items():
        assert metric.x == 0, f'Metric {name} not reset correctly'

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

    metric_collection = metric_collection.to(device='cuda')
    for _, metric in metric_collection.items():
        assert metric.x.is_cuda

    metric_collection = metric_collection.double()
    for _, metric in metric_collection.items():
        assert metric.x.dtype == torch.float64

    metric_collection = metric_collection.half()
    for _, metric in metric_collection.items():
        assert metric.x.dtype == torch.float16


def test_metric_collection_wrong_input(tmpdir):
    """ Check that errors are raised on wrong input """
    dms = DummyMetricSum()

    # Not all input are metrics (list)
    with pytest.raises(ValueError):
        _ = MetricCollection([dms, 5])

    # Not all input are metrics (dict)
    with pytest.raises(ValueError):
        _ = MetricCollection({'metric1': dms, 'metric2': 5})

    # Same metric passed in multiple times
    with pytest.raises(ValueError, match='Encountered two metrics both named *.'):
        _ = MetricCollection([dms, dms])

    # Not a list or dict passed in
    with pytest.warns(Warning, match=' which are not `Metric` so they will be ignored.'):
        _ = MetricCollection(dms, [dms])


def test_metric_collection_args_kwargs(tmpdir):
    """ Check that args and kwargs gets passed correctly in metric collection,
        Checks both update and forward method
    """
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])

    # args gets passed to all metrics
    metric_collection.update(5)
    assert metric_collection['DummyMetricSum'].x == 5
    assert metric_collection['DummyMetricDiff'].x == -5
    metric_collection.reset()
    _ = metric_collection(5)
    assert metric_collection['DummyMetricSum'].x == 5
    assert metric_collection['DummyMetricDiff'].x == -5
    metric_collection.reset()

    # kwargs gets only passed to metrics that it matches
    metric_collection.update(x=10, y=20)
    assert metric_collection['DummyMetricSum'].x == 10
    assert metric_collection['DummyMetricDiff'].x == -20
    metric_collection.reset()
    _ = metric_collection(x=10, y=20)
    assert metric_collection['DummyMetricSum'].x == 10
    assert metric_collection['DummyMetricDiff'].x == -20


@pytest.mark.parametrize(
    "prefix, postfix", [
        [None, None],
        ['prefix_', None],
        [None, '_postfix'],
        ['prefix_', '_postfix'],
    ]
)
def test_metric_collection_prefix_postfix_args(prefix, postfix):
    """ Test that the prefix arg alters the keywords in the output"""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()
    names = ['DummyMetricSum', 'DummyMetricDiff']
    names = [prefix + n if prefix is not None else n for n in names]
    names = [n + postfix if postfix is not None else n for n in names]

    metric_collection = MetricCollection([m1, m2], prefix=prefix, postfix=postfix)

    # test forward
    out = metric_collection(5)
    for name in names:
        assert name in out, 'prefix or postfix argument not working as intended with forward method'

    # test compute
    out = metric_collection.compute()
    for name in names:
        assert name in out, 'prefix or postfix argument not working as intended with compute method'

    # test clone
    new_metric_collection = metric_collection.clone(prefix='new_prefix_')
    out = new_metric_collection(5)
    names = [n[len(prefix):] if prefix is not None else n for n in names]  # strip away old prefix
    for name in names:
        assert f"new_prefix_{name}" in out, 'prefix argument not working as intended with clone method'

    for k, _ in new_metric_collection.items():
        assert 'new_prefix_' in k

    for k in new_metric_collection.keys():
        assert 'new_prefix_' in k

    for k, _ in new_metric_collection.items(keep_base=True):
        assert 'new_prefix_' not in k

    for k in new_metric_collection.keys(keep_base=True):
        assert 'new_prefix_' not in k

    assert isinstance(new_metric_collection.keys(keep_base=True), type(new_metric_collection.keys(keep_base=False)))
    assert isinstance(new_metric_collection.items(keep_base=True), type(new_metric_collection.items(keep_base=False)))

    new_metric_collection = new_metric_collection.clone(postfix='_new_postfix')
    out = new_metric_collection(5)
    names = [n[:-len(postfix)] if postfix is not None else n for n in names]  # strip away old postfix
    for name in names:
        assert f"new_prefix_{name}_new_postfix" in out, 'postfix argument not working as intended with clone method'


def test_metric_collection_repr():
    """
    Test MetricCollection
    """

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

    expected = 'MetricCollection(\n  (A): A()\n  (B): B(),\n  prefix=a\n)'
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix=None, postfix="a")
    expected = 'MetricCollection(\n  (A): A()\n  (B): B(),\n  postfix=a\n)'
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix="a", postfix="b")
    expected = 'MetricCollection(\n  (A): A()\n  (B): B(),\n  prefix=a,\n  postfix=b\n)'
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
    collection.add_metrics({'m1_': DummyMetricSum()})
    collection.add_metrics(m2)

    collection.update(5)
    results = collection.compute()
    assert results['DummyMetricSum'] == results['m1_'] and results['m1_'] == 5
    assert results['DummyMetricDiff'] == -5


def test_collection_check_arg():
    assert MetricCollection._check_arg(None, 'prefix') is None
    assert MetricCollection._check_arg('sample', 'prefix') == 'sample'

    with pytest.raises(ValueError, match="Expected input `postfix` to be a string, but got"):
        MetricCollection._check_arg(1, 'postfix')
