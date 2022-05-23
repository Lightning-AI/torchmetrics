import os.path

from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, NUM_PROCESSES, DummyMetric, MetricTester  # noqa: F401

_PATH_TESTS = os.path.dirname(__file__)
_PATH_ROOT = os.path.dirname(_PATH_TESTS)
