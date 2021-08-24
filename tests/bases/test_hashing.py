from tests.helpers.testers import DummyMetric, DummyNonTensorMetric
import pytest

pytest.mark.parametrize(
    "metric_cls",
    [
        (DummyMetric,),
        (DummyNonTensorMetric,),
    ],
)
def test_metric_hashing(metric_cls):
    """
    Tests that hases are different. 
    See the Metric's hash function for details on why this is required.
    """
    instance_1 = metric_cls()
    instance_2 = metric_cls()

    assert hash(instance_1) != hash(instance_2)
    assert id(instance_1) != id(instance_2)