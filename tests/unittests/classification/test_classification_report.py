# Copyright The Lightning team.
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
"""Tests for ClassificationReport metric using collection-style wrapper."""

import numpy as np
import pytest
import torch
from sklearn import datasets
from sklearn.metrics import classification_report as sklearn_classification_report
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from torchmetrics.classification import (
    BinaryClassificationReport,
    ClassificationReport,
    MulticlassClassificationReport,
    MultilabelClassificationReport,
)
from unittests._helpers import seed_all

seed_all(42)


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC."""
    if dataset is None:
        dataset = datasets.load_iris()

    x = dataset.data
    y = dataset.target

    if binary:
        x, y = x[y < 2], y[y < 2]

    n_samples, n_features = x.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    x, y = x[p], y[p]
    half = int(n_samples / 2)

    rng = np.random.RandomState(0)
    x = np.c_[x, rng.randn(n_samples, 200 * n_features)]

    clf = SVC(kernel="linear", probability=True, random_state=0)
    y_pred_proba = clf.fit(x[:half], y[:half]).predict_proba(x[half:])

    if binary:
        y_pred_proba = y_pred_proba[:, 1]

    y_pred = clf.predict(x[half:])
    y_true = y[half:]
    return y_true, y_pred, y_pred_proba


class TestBinaryClassificationReport:
    """Tests for BinaryClassificationReport."""

    def test_basic_usage(self):
        """Test basic binary classification report."""
        target = torch.tensor([0, 1, 0, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = BinaryClassificationReport()
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, str)
        assert "precision" in result
        assert "recall" in result
        assert "f1-score" in result
        assert "accuracy" in result

    def test_output_dict(self):
        """Test binary classification report with output_dict=True."""
        target = torch.tensor([0, 1, 0, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = BinaryClassificationReport(output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, dict)
        assert "0" in result
        assert "1" in result
        assert "accuracy" in result
        assert "macro avg" in result
        assert "weighted avg" in result

        # Check accuracy value
        assert abs(result["accuracy"] - 0.75) < 0.01

    def test_custom_metrics(self):
        """Test binary classification report with custom metrics."""
        target = torch.tensor([0, 1, 0, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = BinaryClassificationReport(metrics=["precision", "specificity"], output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert "precision" in result["0"]
        assert "specificity" in result["0"]
        assert "recall" not in result["0"]
        assert "f1-score" not in result["0"]

    def test_target_names(self):
        """Test binary classification report with custom target names."""
        target = torch.tensor([0, 1, 0, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = BinaryClassificationReport(target_names=["negative", "positive"], output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert "negative" in result
        assert "positive" in result

    def test_ignore_index(self):
        """Test binary classification report with ignore_index."""
        target = torch.tensor([0, 1, -1, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = BinaryClassificationReport(ignore_index=-1, output_dict=True)
        report.update(preds, target)
        result = report.compute()

        # Total support should be 3 (excluding ignored sample)
        total_support = result["weighted avg"]["support"]
        assert total_support == 3

    def test_matches_sklearn(self):
        """Test that binary classification report matches sklearn."""
        y_true, y_pred, _ = make_prediction(binary=True)

        target = torch.tensor(y_true)
        preds = torch.tensor(y_pred)

        report = BinaryClassificationReport(output_dict=True)
        report.update(preds, target)
        tm_result = report.compute()

        sk_result = sklearn_classification_report(y_true, y_pred, output_dict=True)

        # Compare accuracy
        assert abs(tm_result["accuracy"] - sk_result["accuracy"]) < 0.01

        # Compare macro avg
        assert abs(tm_result["macro avg"]["precision"] - sk_result["macro avg"]["precision"]) < 0.01
        assert abs(tm_result["macro avg"]["recall"] - sk_result["macro avg"]["recall"]) < 0.01
        assert abs(tm_result["macro avg"]["f1-score"] - sk_result["macro avg"]["f1-score"]) < 0.01


class TestMulticlassClassificationReport:
    """Tests for MulticlassClassificationReport."""

    def test_basic_usage(self):
        """Test basic multiclass classification report."""
        target = torch.tensor([0, 1, 2, 2, 2])
        preds = torch.tensor([0, 0, 2, 2, 1])

        report = MulticlassClassificationReport(num_classes=3)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, str)
        assert "precision" in result
        assert "recall" in result
        assert "f1-score" in result

    def test_output_dict(self):
        """Test multiclass classification report with output_dict=True."""
        target = torch.tensor([0, 1, 2, 2, 2])
        preds = torch.tensor([0, 0, 2, 2, 1])

        report = MulticlassClassificationReport(num_classes=3, output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, dict)
        assert "0" in result
        assert "1" in result
        assert "2" in result
        assert "accuracy" in result
        assert "macro avg" in result
        assert "weighted avg" in result

        # Check accuracy value (3 correct out of 5)
        assert abs(result["accuracy"] - 0.60) < 0.01

    def test_custom_metrics(self):
        """Test multiclass classification report with custom metrics."""
        target = torch.tensor([0, 1, 2, 2, 2])
        preds = torch.tensor([0, 0, 2, 2, 1])

        report = MulticlassClassificationReport(num_classes=3, metrics=["precision", "specificity"], output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert "precision" in result["0"]
        assert "specificity" in result["0"]
        assert "recall" not in result["0"]

    def test_top_k(self):
        """Test multiclass classification report with top_k."""
        target = torch.tensor([0, 1, 2])
        # Probabilities where correct class is in top-2
        preds = torch.tensor([
            [0.1, 0.6, 0.3],  # True: 0, Pred top-2: [1, 2] - wrong at k=1, wrong at k=2
            [0.4, 0.5, 0.1],  # True: 1, Pred top-2: [1, 0] - correct at k=1
            [0.1, 0.3, 0.6],  # True: 2, Pred top-2: [2, 1] - correct at k=1
        ])

        report_k1 = MulticlassClassificationReport(num_classes=3, top_k=1, output_dict=True)
        report_k1.update(preds, target)
        result_k1 = report_k1.compute()

        report_k2 = MulticlassClassificationReport(num_classes=3, top_k=2, output_dict=True)
        report_k2.update(preds, target)
        result_k2 = report_k2.compute()

        # Accuracy should be higher or equal with higher k
        assert result_k2["accuracy"] >= result_k1["accuracy"]

    def test_matches_sklearn(self):
        """Test that multiclass classification report matches sklearn."""
        y_true, y_pred, _ = make_prediction(binary=False)

        target = torch.tensor(y_true)
        preds = torch.tensor(y_pred)

        report = MulticlassClassificationReport(num_classes=3, output_dict=True)
        report.update(preds, target)
        tm_result = report.compute()

        sk_result = sklearn_classification_report(y_true, y_pred, output_dict=True)

        # Compare accuracy
        assert abs(tm_result["accuracy"] - sk_result["accuracy"]) < 0.01

        # Compare macro avg
        assert abs(tm_result["macro avg"]["precision"] - sk_result["macro avg"]["precision"]) < 0.01
        assert abs(tm_result["macro avg"]["recall"] - sk_result["macro avg"]["recall"]) < 0.01


class TestMultilabelClassificationReport:
    """Tests for MultilabelClassificationReport."""

    def test_basic_usage(self):
        """Test basic multilabel classification report."""
        target = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        preds = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]])

        report = MultilabelClassificationReport(num_labels=3)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, str)
        assert "precision" in result
        assert "recall" in result
        assert "micro avg" in result
        assert "samples avg" in result

    def test_output_dict(self):
        """Test multilabel classification report with output_dict=True."""
        target = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        preds = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0]])

        report = MultilabelClassificationReport(num_labels=3, output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, dict)
        assert "0" in result
        assert "1" in result
        assert "2" in result
        assert "micro avg" in result
        assert "macro avg" in result
        assert "weighted avg" in result
        assert "samples avg" in result

    def test_custom_metrics(self):
        """Test multilabel classification report with custom metrics."""
        target = torch.tensor([[1, 0, 1], [0, 1, 0]])
        preds = torch.tensor([[1, 0, 1], [0, 1, 1]])

        report = MultilabelClassificationReport(num_labels=3, metrics=["precision", "accuracy"], output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert "precision" in result["0"]
        assert "accuracy" in result["0"]
        assert "recall" not in result["0"]


class TestClassificationReportWrapper:
    """Tests for ClassificationReport task wrapper."""

    def test_binary_task(self):
        """Test ClassificationReport with binary task."""
        target = torch.tensor([0, 1, 0, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = ClassificationReport(task="binary", output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_multiclass_task(self):
        """Test ClassificationReport with multiclass task."""
        target = torch.tensor([0, 1, 2])
        preds = torch.tensor([0, 1, 1])

        report = ClassificationReport(task="multiclass", num_classes=3, output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_multilabel_task(self):
        """Test ClassificationReport with multilabel task."""
        target = torch.tensor([[1, 0], [0, 1]])
        preds = torch.tensor([[1, 0], [0, 1]])

        report = ClassificationReport(task="multilabel", num_labels=2, output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert isinstance(result, dict)
        assert "micro avg" in result

    def test_invalid_task(self):
        """Test ClassificationReport with invalid task raises error."""
        with pytest.raises(ValueError, match="Invalid"):
            ClassificationReport(task="invalid")

    def test_missing_num_classes(self):
        """Test that multiclass task requires num_classes."""
        with pytest.raises(ValueError, match="num_classes"):
            ClassificationReport(task="multiclass")

    def test_missing_num_labels(self):
        """Test that multilabel task requires num_labels."""
        with pytest.raises(ValueError, match="num_labels"):
            ClassificationReport(task="multilabel")


class TestMetricAliases:
    """Tests for metric name aliases."""

    def test_f1_alias(self):
        """Test that 'f1' is an alias for 'f1-score'."""
        target = torch.tensor([0, 1, 0, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = BinaryClassificationReport(metrics=["f1"], output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert "f1-score" in result["0"]

    def test_f_measure_alias(self):
        """Test that 'f-measure' is an alias for 'f1-score'."""
        target = torch.tensor([0, 1, 0, 1])
        preds = torch.tensor([0, 1, 1, 1])

        report = BinaryClassificationReport(metrics=["f-measure"], output_dict=True)
        report.update(preds, target)
        result = report.compute()

        assert "f1-score" in result["0"]

    def test_invalid_metric_name(self):
        """Test that invalid metric name raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            BinaryClassificationReport(metrics=["invalid_metric"])


class TestZeroSupportClassHandling:
    """Tests for handling classes with zero support."""

    def test_zero_support_class_metrics(self):
        """Test that zero-support classes are handled correctly."""
        # Only classes 0, 1, 2 have samples; class 3 has zero support
        preds = torch.tensor([0, 1, 2, 2, 1, 0])
        target = torch.tensor([0, 1, 2, 2, 1, 0])
        num_classes = 4

        report = MulticlassClassificationReport(num_classes=num_classes, output_dict=True)
        report.update(preds, target)
        result = report.compute()

        # Class 3 should have zero support
        assert result["3"]["support"] == 0

        # Other classes should have correct support
        assert result["0"]["support"] == 2
        assert result["1"]["support"] == 2
        assert result["2"]["support"] == 2

        # All predictions correct, so precision/recall for present classes should be 1.0
        assert abs(result["0"]["precision"] - 1.0) < 0.01
        assert abs(result["1"]["precision"] - 1.0) < 0.01
        assert abs(result["2"]["precision"] - 1.0) < 0.01


class TestResetAndMultipleUpdates:
    """Tests for reset and multiple updates."""

    def test_multiple_updates(self):
        """Test that multiple updates accumulate correctly."""
        report = BinaryClassificationReport(output_dict=True)

        # First batch
        report.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        # Second batch
        report.update(torch.tensor([1, 0]), torch.tensor([1, 0]))

        result = report.compute()

        # All predictions correct, accuracy should be 1.0
        assert abs(result["accuracy"] - 1.0) < 0.01

    def test_reset(self):
        """Test that reset clears accumulated state."""
        report = BinaryClassificationReport(output_dict=True)

        report.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
        report.reset()
        report.update(torch.tensor([0, 0]), torch.tensor([0, 1]))

        result = report.compute()

        # Only second batch counts, 1 correct out of 2
        assert abs(result["accuracy"] - 0.5) < 0.01
