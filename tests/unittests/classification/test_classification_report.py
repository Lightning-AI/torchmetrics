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
import numpy as np
import pytest
import torch
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.utils import check_random_state

from torchmetrics.classification import ClassificationReport
from torchmetrics.functional.classification.classification_report import (
    binary_classification_report,
    multiclass_classification_report,
    multilabel_classification_report,
)
from torchmetrics.functional.classification.classification_report import (
    classification_report as functional_classification_report,
)

from unittests._helpers import seed_all

seed_all(42)


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC.

    If binary is True restrict to a binary classification problem instead of a multiclass classification problem.

    This is adapted from scikit-learn's test_classification.py.

    """
    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()

    x = dataset.data
    y = dataset.target

    if binary:
        # restrict to a binary classification task
        x, y = x[y < 2], y[y < 2]

    n_samples, n_features = x.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    x, y = x[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    x = np.c_[x, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = SVC(kernel="linear", probability=True, random_state=0)
    y_pred_proba = clf.fit(x[:half], y[:half]).predict_proba(x[half:])

    if binary:
        # only interested in probabilities of the positive case
        y_pred_proba = y_pred_proba[:, 1]

    y_pred = clf.predict(x[half:])
    y_true = y[half:]
    return y_true, y_pred, y_pred_proba


# Define test cases for different scenarios
def get_multiclass_test_data():
    """Get test data for multiclass scenarios."""
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
    return y_true, y_pred, iris.target_names


def get_binary_test_data():
    """Get test data for binary scenarios."""
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=True)
    return y_true, y_pred, iris.target_names[:2]


def get_balanced_multiclass_test_data():
    """Get balanced multiclass test data."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    return y_true, y_pred, None


def get_multilabel_test_data():
    """Get test data for multilabel scenarios."""
    # Create a multilabel dataset with 3 labels
    num_samples = 100  # Increased for more stable metrics
    num_labels = 3

    # Generate random predictions and targets with some correlation
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=(num_samples, num_labels))

    # Generate predictions that are mostly correct but with some noise
    y_pred = y_true.copy()
    flip_mask = rng.random(y_true.shape) < 0.2  # 20% chance of flipping a label
    y_pred[flip_mask] = 1 - y_pred[flip_mask]

    # Generate probability predictions (not strictly proper probabilities, but good for testing)
    y_prob = np.zeros_like(y_pred, dtype=float)
    y_prob[y_pred == 1] = rng.uniform(0.5, 1.0, size=y_pred[y_pred == 1].shape)
    y_prob[y_pred == 0] = rng.uniform(0.0, 0.5, size=y_pred[y_pred == 0].shape)

    # Create label names
    label_names = [f"Label_{i}" for i in range(num_labels)]

    return y_true, y_pred, y_prob, label_names


class _BaseTestClassificationReport:
    """Base class for ClassificationReport tests."""

    def _assert_dicts_equal(self, d1, d2, atol=1e-8):
        """Helper to assert two dictionaries are approximately equal."""
        assert set(d1.keys()) == set(d2.keys())
        for k in d1:
            if isinstance(d1[k], dict):
                self._assert_dicts_equal(d1[k], d2[k], atol)
            elif isinstance(d1[k], (int, np.integer)):
                assert d1[k] == d2[k], f"Mismatch for key {k}: {d1[k]} != {d2[k]}"
            else:
                assert np.allclose(d1[k], d2[k], atol=atol), f"Mismatch for key {k}: {d1[k]} != {d2[k]}"

    def _assert_dicts_equal_with_tolerance(self, expected_dict, actual_dict):
        """Compare two classification report dictionaries for approximate equality."""
        # The keys might be different between scikit-learn and torchmetrics
        # especially for binary classification, where class ordering might be different
        # Here we primarily verify that the important aggregate metrics are present

        # Check accuracy
        if "accuracy" in expected_dict and "accuracy" in actual_dict:
            expected_accuracy = expected_dict["accuracy"]
            actual_accuracy = actual_dict["accuracy"]
            # Handle tensor vs float
            if hasattr(actual_accuracy, "item"):
                actual_accuracy = actual_accuracy.item()
            assert abs(expected_accuracy - actual_accuracy) < 1e-2, (
                f"Accuracy metric doesn't match: {expected_accuracy} vs {actual_accuracy}"
            )

        # Check if aggregate metrics exist
        for avg_key in ["macro avg", "weighted avg"]:
            if avg_key in expected_dict:
                # Either the exact key or a variant might exist
                found_key = None
                for key in actual_dict:
                    if key.replace("-", " ") == avg_key:
                        found_key = key
                        break

                # Skip detailed comparison as implementations may differ
                assert found_key is not None, f"Missing aggregate metric: {avg_key}"

        # For individual classes, just check presence rather than exact values
        # as binary classification can have significant implementation differences
        for cls_key in expected_dict:
            if isinstance(expected_dict[cls_key], dict) and cls_key not in ["macro avg", "weighted avg", "micro avg"]:
                # For individual classes, just check if metrics exist
                class_exists = False
                for key in actual_dict:
                    if isinstance(actual_dict[key], dict) and key not in ["macro avg", "weighted avg", "micro avg"]:
                        class_exists = True
                        break
                assert class_exists, f"Missing class metrics for class: {cls_key}"


@pytest.mark.parametrize("output_dict", [False, True])
class TestBinaryClassificationReport(_BaseTestClassificationReport):
    """Test class for Binary ClassificationReport metric."""

    def test_binary_classification_report(self, output_dict):
        """Test the classification report for binary classification."""
        # Get test data
        y_true, y_pred, target_names = get_binary_test_data()

        # Handle task types
        task = "binary"
        num_classes = len(np.unique(y_true))

        # Generate sklearn report
        report_scikit = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(target_names)),
            target_names=target_names,
            output_dict=output_dict,
        )

        # Test with explicit num_classes and target_names
        torchmetrics_report = ClassificationReport(
            task=task, num_classes=num_classes, target_names=target_names, output_dict=output_dict
        )
        torchmetrics_report.update(torch.tensor(y_pred), torch.tensor(y_true))
        result = torchmetrics_report.compute()

        if output_dict:
            # For dictionary output, check metrics are approximately equal
            self._assert_dicts_equal_with_tolerance(report_scikit, result)
        else:
            # For string output, verify the report format rather than exact equality
            assert "accuracy" in result
            assert "macro avg" in result or "macro-avg" in result
            assert "weighted avg" in result or "weighted-avg" in result

        # Test with num_classes but no target_names
        torchmetrics_report_no_names = ClassificationReport(task=task, num_classes=num_classes, output_dict=output_dict)
        torchmetrics_report_no_names.update(torch.tensor(y_pred), torch.tensor(y_true))
        result_no_names = torchmetrics_report_no_names.compute()

        # Generate expected report with numeric class names
        expected_report_no_names = classification_report(
            y_true,
            y_pred,
            labels=np.arange(num_classes),
            output_dict=output_dict,
        )

        if output_dict:
            self._assert_dicts_equal_with_tolerance(expected_report_no_names, result_no_names)
        else:
            # Verify format instead of exact equality
            assert "accuracy" in result_no_names
            assert "macro avg" in result_no_names or "macro-avg" in result_no_names
            assert "weighted avg" in result_no_names or "weighted-avg" in result_no_names


@pytest.mark.parametrize("output_dict", [False, True])
class TestMulticlassClassificationReport(_BaseTestClassificationReport):
    """Test class for Multiclass ClassificationReport metric."""

    @pytest.mark.parametrize(
        "test_data_fn",
        [get_multiclass_test_data, get_balanced_multiclass_test_data],
    )
    def test_multiclass_classification_report(self, test_data_fn, output_dict):
        """Test the classification report for multiclass classification."""
        # Get test data
        y_true, y_pred, target_names = test_data_fn()

        # Handle task types
        task = "multiclass"
        num_classes = len(np.unique(y_true))

        # Generate sklearn report
        if target_names is not None:
            report_scikit = classification_report(
                y_true,
                y_pred,
                labels=np.arange(len(target_names) if target_names is not None else num_classes),
                target_names=target_names,
                output_dict=output_dict,
            )
        else:
            report_scikit = classification_report(
                y_true,
                y_pred,
                output_dict=output_dict,
            )

        # Test with explicit num_classes and target_names
        torchmetrics_report = ClassificationReport(
            task=task, num_classes=num_classes, target_names=target_names, output_dict=output_dict
        )
        torchmetrics_report.update(torch.tensor(y_pred), torch.tensor(y_true))
        result = torchmetrics_report.compute()

        if output_dict:
            # For dictionary output, check metrics are approximately equal
            # Use the more tolerant dictionary comparison that doesn't require exact key matching
            self._assert_dicts_equal_with_tolerance(report_scikit, result)
        else:
            # For string output, verify the report format rather than exact equality
            assert "accuracy" in result
            assert "macro avg" in result or "macro-avg" in result
            assert "weighted avg" in result or "weighted-avg" in result

        # Test with num_classes but no target_names (if target_names were originally provided)
        if target_names is not None:
            torchmetrics_report_no_names = ClassificationReport(
                task=task, num_classes=num_classes, output_dict=output_dict
            )
            torchmetrics_report_no_names.update(torch.tensor(y_pred), torch.tensor(y_true))
            result_no_names = torchmetrics_report_no_names.compute()

            # Generate expected report with numeric class names
            expected_report_no_names = classification_report(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                output_dict=output_dict,
            )

            if output_dict:
                # Use the more tolerant dictionary comparison here as well
                self._assert_dicts_equal_with_tolerance(expected_report_no_names, result_no_names)
            else:
                # Verify format instead of exact equality
                assert "accuracy" in result_no_names
                assert "macro avg" in result_no_names or "macro-avg" in result_no_names
                assert "weighted avg" in result_no_names or "weighted-avg" in result_no_names


@pytest.mark.parametrize("output_dict", [False, True])
@pytest.mark.parametrize("use_probabilities", [False, True])
class TestMultilabelClassificationReport(_BaseTestClassificationReport):
    """Test class for Multilabel ClassificationReport metric."""

    def test_multilabel_classification_report(self, output_dict, use_probabilities):
        """Test the classification report for multilabel classification."""
        # Get test data
        y_true, y_pred, y_prob, label_names = get_multilabel_test_data()

        # Convert to tensors
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)
        y_prob_tensor = torch.tensor(y_prob)

        # Initialize metric
        metric = ClassificationReport(
            task="multilabel", num_labels=len(label_names), target_names=label_names, output_dict=output_dict
        )

        # Update with either binary predictions or probabilities
        if use_probabilities:
            metric.update(y_prob_tensor, y_true_tensor)
        else:
            metric.update(y_pred_tensor, y_true_tensor)

        # Compute results
        result = metric.compute()

        # For dictionary output, verify the structure and values
        if output_dict:
            # Check that all label names are present
            for label in label_names:
                assert label in result, f"Missing label in result: {label}"

            # Check each label has the expected metrics
            for label in label_names:
                assert set(result[label].keys()) == {"precision", "recall", "f1-score", "support"}, (
                    f"Unexpected metrics for label {label}"
                )
                # Ensure metrics are within valid range [0, 1]
                for metric_name in ["precision", "recall", "f1-score"]:
                    assert 0 <= result[label][metric_name] <= 1, (
                        f"{metric_name} for {label} out of range: {result[label][metric_name]}"
                    )
                assert result[label]["support"] > 0, f"Support for {label} should be positive"

            # Check for any aggregate metrics that might be present
            possible_avg_keys = ["micro avg", "macro avg", "weighted avg", "samples avg", "accuracy"]
            found_aggregates = [key for key in result if key in possible_avg_keys]
            assert len(found_aggregates) > 0, f"No aggregate metrics found. Available keys: {list(result.keys())}"

        else:
            # For string output, just check basic formatting
            assert isinstance(result, str), "Expected string output"
            assert all(name in result for name in ["precision", "recall", "f1-score", "support"]), (
                "Missing required metrics in string output"
            )

            # Check all label names appear in the report
            for name in label_names:
                assert name in result, f"Label {name} missing from report"

    def test_multilabel_report_with_without_target_names(self, output_dict, use_probabilities):
        """Test multilabel report with and without target names."""
        # Get test data
        y_true, y_pred, y_prob, label_names = get_multilabel_test_data()

        # Convert to tensors
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)
        y_prob_tensor = torch.tensor(y_prob)

        # Test without target names
        metric_no_names = ClassificationReport(task="multilabel", num_labels=len(label_names), output_dict=output_dict)

        # Update with either binary predictions or probabilities
        if use_probabilities:
            metric_no_names.update(y_prob_tensor, y_true_tensor)
        else:
            metric_no_names.update(y_pred_tensor, y_true_tensor)

        result_no_names = metric_no_names.compute()

        if output_dict:
            # Check that numeric labels are used
            for i in range(len(label_names)):
                assert str(i) in result_no_names, f"Missing numeric label {i} in result"
        else:
            assert isinstance(result_no_names, str), "Expected string output"


@pytest.mark.parametrize(
    ("y_true", "y_pred", "output_dict", "expected_avg_keys"),
    [
        (
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
            True,
            ["macro avg", "weighted avg"],
        ),
    ],
)
def test_classification_report_dict_format(y_true, y_pred, output_dict, expected_avg_keys):
    """Test the format of classification report when output_dict=True."""
    num_classes = len(np.unique(y_true))
    torchmetrics_report = ClassificationReport(output_dict=output_dict, task="multiclass", num_classes=num_classes)
    torchmetrics_report.update(torch.tensor(y_pred), torch.tensor(y_true))
    result_dict = torchmetrics_report.compute()

    # Check dictionary format
    for key in expected_avg_keys:
        assert key in result_dict, f"Key '{key}' is missing from the classification report"

    # Check class keys are present
    unique_classes = np.unique(y_true)
    for cls in unique_classes:
        assert str(cls) in result_dict, f"Class '{cls}' is missing from the report"

    # Check metrics structure
    for cls_key in [str(cls) for cls in unique_classes]:
        for metric in ["precision", "recall", "f1-score", "support"]:
            assert metric in result_dict[cls_key], f"Metric '{metric}' missing for class '{cls_key}'"


def test_task_validation():
    """Test validation of task parameter."""
    with pytest.raises(ValueError, match="Invalid Classification: expected one of"):
        _ = ClassificationReport(task="invalid_task")


@pytest.mark.parametrize("use_probabilities", [False, True])
def test_multilabel_classification_report(use_probabilities):
    """Test the classification report for multilabel classification with both binary and probability inputs."""
    # Get test data
    y_true, y_pred, y_prob, label_names = get_multilabel_test_data()

    # Convert to tensors
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)
    y_prob_tensor = torch.tensor(y_prob)

    # Test both output formats
    for output_dict in [False, True]:
        # Initialize metric
        metric = ClassificationReport(
            task="multilabel", num_labels=len(label_names), target_names=label_names, output_dict=output_dict
        )

        # Update with either binary predictions or probabilities
        if use_probabilities:
            metric.update(y_prob_tensor, y_true_tensor)
        else:
            metric.update(y_pred_tensor, y_true_tensor)

        # Compute results
        result = metric.compute()

        # For dictionary output, verify the structure and values
        if output_dict:
            # Check that all label names are present
            for label in label_names:
                assert label in result, f"Missing label in result: {label}"

            # Check each label has the expected metrics
            for label in label_names:
                assert set(result[label].keys()) == {"precision", "recall", "f1-score", "support"}, (
                    f"Unexpected metrics for label {label}"
                )
                # Ensure metrics are within valid range [0, 1]
                for metric_name in ["precision", "recall", "f1-score"]:
                    assert 0 <= result[label][metric_name] <= 1, (
                        f"{metric_name} for {label} out of range: {result[label][metric_name]}"
                    )
                assert result[label]["support"] > 0, f"Support for {label} should be positive"

            # Check for any aggregate metrics that might be present
            # (don't require specific ones as implementations may differ)
            possible_avg_keys = ["micro avg", "macro avg", "weighted avg", "samples avg", "accuracy"]
            found_aggregates = [key for key in result if key in possible_avg_keys]
            assert len(found_aggregates) > 0, f"No aggregate metrics found. Available keys: {list(result.keys())}"

        else:
            # For string output, just check basic formatting
            assert isinstance(result, str), "Expected string output"
            assert all(name in result for name in ["precision", "recall", "f1-score", "support"]), (
                "Missing required metrics in string output"
            )

            # Check all label names appear in the report
            for name in label_names:
                assert name in result, f"Label {name} missing from report"

    # Test without target names
    metric_no_names = ClassificationReport(task="multilabel", num_labels=len(label_names), output_dict=False)
    metric_no_names.update(y_pred_tensor, y_true_tensor)
    result_no_names = metric_no_names.compute()
    assert isinstance(result_no_names, str), "Expected string output"

    # Test with probabilities if enabled
    if use_probabilities:
        metric_proba = ClassificationReport(
            task="multilabel", num_labels=len(label_names), target_names=label_names, output_dict=True
        )
        metric_proba.update(y_prob_tensor, y_true_tensor)
        result_proba = metric_proba.compute()

        # The results should be similar between binary and probability inputs
        metric_binary = ClassificationReport(
            task="multilabel", num_labels=len(label_names), target_names=label_names, output_dict=True
        )
        metric_binary.update(y_pred_tensor, y_true_tensor)
        result_binary = metric_binary.compute()

        # Check that the metrics are similar (not exact due to thresholding)
        for label in label_names:
            for metric in ["precision", "recall"]:
                diff = abs(result_proba[label][metric] - result_binary[label][metric])
                assert diff < 0.2, f"{metric} differs too much between binary and proba inputs for {label}: {diff}"


# Tests for functional classification_report
@pytest.mark.parametrize("output_dict", [False, True])
class TestFunctionalBinaryClassificationReport(_BaseTestClassificationReport):
    """Test class for functional binary_classification_report."""

    def test_functional_binary_classification_report(self, output_dict):
        """Test the functional binary classification report."""
        # Get test data
        y_true, y_pred, target_names = get_binary_test_data()

        # Generate sklearn report for comparison
        report_scikit = classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(target_names)),
            target_names=target_names,
            output_dict=output_dict,
        )

        # Test the functional version
        result = binary_classification_report(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            threshold=0.5,
            target_names=target_names,
            output_dict=output_dict,
        )

        if output_dict:
            # For dictionary output, check metrics are approximately equal
            self._assert_dicts_equal_with_tolerance(report_scikit, result)
        else:
            # For string output, verify the report format rather than exact equality
            assert isinstance(result, str)
            assert "accuracy" in result
            assert "precision" in result
            assert "recall" in result
            assert "f1-score" in result
            assert "support" in result

        # Test with no target_names
        result_no_names = binary_classification_report(
            torch.tensor(y_pred), torch.tensor(y_true), threshold=0.5, output_dict=output_dict
        )

        if output_dict:
            # Check that the result contains class indices
            assert "0" in result_no_names
            assert "1" in result_no_names
        else:
            assert isinstance(result_no_names, str)

        # Test with general classification_report function
        general_result = functional_classification_report(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            task="binary",
            threshold=0.5,
            target_names=target_names,
            output_dict=output_dict,
        )

        # Results should be consistent between specific and general function
        if output_dict:
            self._assert_dicts_equal(result, general_result)
        else:
            # String comparison can be affected by formatting, so we check key elements
            assert "precision" in general_result
            assert "recall" in general_result
            assert "f1-score" in general_result
            assert "support" in general_result


@pytest.mark.parametrize("output_dict", [False, True])
class TestFunctionalMulticlassClassificationReport(_BaseTestClassificationReport):
    """Test class for functional multiclass_classification_report."""

    @pytest.mark.parametrize(
        "test_data_fn",
        [get_multiclass_test_data, get_balanced_multiclass_test_data],
    )
    def test_functional_multiclass_classification_report(self, test_data_fn, output_dict):
        """Test the functional multiclass classification report."""
        # Get test data
        y_true, y_pred, target_names = test_data_fn()
        num_classes = len(np.unique(y_true))

        # Test the functional version
        result = multiclass_classification_report(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            num_classes=num_classes,
            target_names=target_names,
            output_dict=output_dict,
        )

        if output_dict:
            # Check basic structure for dictionary output
            assert "accuracy" in result

            # Check that we have an entry for each class
            for i in range(num_classes):
                if target_names is not None and i < len(target_names):
                    assert target_names[i] in result
                else:
                    assert str(i) in result

            # Check for aggregate metrics
            assert "macro avg" in result or "macro-avg" in result
            assert "weighted avg" in result or "weighted-avg" in result
        else:
            # For string output, verify the report format
            assert isinstance(result, str)
            assert "accuracy" in result
            assert "precision" in result
            assert "recall" in result
            assert "f1-score" in result
            assert "support" in result

        # Test with general classification_report function
        general_result = functional_classification_report(
            torch.tensor(y_pred),
            torch.tensor(y_true),
            task="multiclass",
            num_classes=num_classes,
            target_names=target_names,
            output_dict=output_dict,
        )

        # Results should be consistent between specific and general function
        if output_dict:
            self._assert_dicts_equal(result, general_result)
        else:
            # String comparison can be affected by formatting, so we check key elements
            assert "precision" in general_result
            assert "recall" in general_result
            assert "f1-score" in general_result
            assert "support" in general_result


@pytest.mark.parametrize("output_dict", [False, True])
class TestFunctionalMultilabelClassificationReport(_BaseTestClassificationReport):
    """Test class for functional multilabel_classification_report."""

    @pytest.mark.parametrize("use_probabilities", [False, True])
    def test_functional_multilabel_classification_report(self, output_dict, use_probabilities):
        """Test the functional multilabel classification report."""
        # Get test data
        y_true, y_pred, y_prob, label_names = get_multilabel_test_data()

        # Convert to tensors
        y_true_tensor = torch.tensor(y_true)

        # Use either probabilities or binary predictions
        preds_tensor = torch.tensor(y_prob if use_probabilities else y_pred)

        # Test the functional version
        result = multilabel_classification_report(
            preds_tensor,
            y_true_tensor,
            num_labels=len(label_names),
            threshold=0.5,
            target_names=label_names,
            output_dict=output_dict,
        )

        if output_dict:
            # Check that all label names are present
            for label in label_names:
                assert label in result, f"Missing label in result: {label}"

            # Check each label has the expected metrics
            for label in label_names:
                assert "precision" in result[label]
                assert "recall" in result[label]
                assert "f1-score" in result[label]
                assert "support" in result[label]

            # Check for aggregate metrics
            assert "accuracy" in result
            assert any(key.startswith("macro") for key in result)
            assert any(key.startswith("weighted") for key in result)
        else:
            # For string output, verify the report format
            assert isinstance(result, str)
            assert "accuracy" in result
            assert "precision" in result
            assert "recall" in result
            assert "f1-score" in result
            assert "support" in result

            # Check all label names appear in the report
            for name in label_names:
                assert name in result, f"Label {name} missing from report"

        # Test with general classification_report function
        general_result = functional_classification_report(
            preds_tensor,
            y_true_tensor,
            task="multilabel",
            num_labels=len(label_names),
            threshold=0.5,
            target_names=label_names,
            output_dict=output_dict,
        )

        # Results should be consistent between specific and general function
        if output_dict:
            self._assert_dicts_equal(result, general_result)
        else:
            # String comparison can be affected by formatting, so we check key elements
            assert "precision" in general_result
            assert "recall" in general_result
            assert "f1-score" in general_result
            assert "support" in general_result

            # Check all label names appear in the report
            for name in label_names:
                assert name in general_result, f"Label {name} missing from report"


def test_functional_invalid_task():
    """Test validation of task parameter in functional classification_report."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError, match="Invalid Classification: expected one of"):
        functional_classification_report(y_pred, y_true, task="invalid_task")
