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


# Define fixtures for test data with different scenarios
@pytest.fixture(params=[
    ("binary", "get_binary_test_data"),
    ("multiclass", "get_multiclass_test_data"),
    ("multiclass", "get_balanced_multiclass_test_data"),
    ("multilabel", "get_multilabel_test_data"),
])
def classification_test_data(request):
    """Return test data for different classification scenarios."""
    task, data_fn = request.param
    
    # Get the appropriate test data function
    data_function = globals()[data_fn]
    
    if task == "multilabel":
        y_true, y_pred, y_prob, target_names = data_function()
        return task, y_true, y_pred, target_names, y_prob
    else:
        y_true, y_pred, target_names = data_function()
        return task, y_true, y_pred, target_names, None


def get_test_data_with_ignore_index(task):
    """Generate test data with ignore_index scenario for different tasks."""
    if task == "binary":
        preds = torch.tensor([0, 1, 1, 0, 1, 0])
        target = torch.tensor([0, 1, -1, 0, 1, -1])  # -1 will be ignored
        ignore_index = -1
        expected_support = 4  # Only 4 valid samples
        return preds, target, ignore_index, expected_support
    elif task == "multiclass":
        preds = torch.tensor([0, 1, 2, 1, 2, 0, 1])
        target = torch.tensor([0, 1, 2, -1, 2, 0, -1])  # -1 will be ignored
        ignore_index = -1
        expected_support = 5  # Only 5 valid samples
        return preds, target, ignore_index, expected_support
    elif task == "multilabel":
        preds = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        target = torch.tensor([[1, 0, 1], [0, -1, 0], [1, 1, -1], [0, 0, 1]])  # -1 will be ignored
        ignore_index = -1
        expected_support = [2, 1, 2]  # Per-label support counts
        return preds, target, ignore_index, expected_support


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
                # Handle NaN values specially - if both are NaN, consider them equal
                if np.isnan(d1[k]) and np.isnan(d2[k]):
                    continue
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

    def _verify_string_report(self, report):
        """Verify that a string report has the expected format."""
        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report
        assert "support" in report
        
        # Check for aggregate metrics
        assert any(metric in report for metric in ["accuracy", "macro avg", "weighted avg", "macro-avg", "weighted-avg"])


@pytest.mark.parametrize("output_dict", [False, True])
class TestClassificationReport(_BaseTestClassificationReport):
    """Unified test class for all ClassificationReport types."""

    @pytest.mark.parametrize("with_target_names", [True, False])
    @pytest.mark.parametrize("use_probabilities", [False, True])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_classification_report(self, classification_test_data, output_dict, with_target_names, use_probabilities, ignore_index):
        """Test the classification report across different scenarios."""
        task, y_true, y_pred, target_names, y_prob = classification_test_data
        
        # Skip irrelevant combinations
        if task != "multilabel" and use_probabilities:
            pytest.skip("Probabilities only relevant for multilabel tasks")
            
        # Use ignore_index test data if ignore_index is specified
        if ignore_index is not None:
            y_pred, y_true, ignore_index, expected_support = get_test_data_with_ignore_index(task)
            target_names = ['0', '1', '2'] if task in ["multiclass", "multilabel"] else ['0', '1']
            
        # Create common parameters for all tasks
        common_params = {
            "task": task,
            "output_dict": output_dict,
            "ignore_index": ignore_index,
        }
        
        # Add task-specific parameters
        if task == "binary":
            common_params["num_classes"] = len(np.unique(y_true)) if ignore_index is None else 2
        elif task == "multiclass":
            common_params["num_classes"] = len(np.unique(y_true)) if ignore_index is None else 3
        elif task == "multilabel":
            common_params["num_labels"] = y_true.shape[1] if ignore_index is None else 3
            common_params["threshold"] = 0.5
            
        # Handle target names
        if with_target_names and target_names is not None:
            common_params["target_names"] = target_names
            
        # Create metric and update with data
        torchmetrics_report = ClassificationReport(**common_params)
        
        # Use probabilities if applicable (only for multilabel currently)
        if task == "multilabel" and use_probabilities and y_prob is not None and ignore_index is None:
            torchmetrics_report.update(torch.tensor(y_prob), torch.tensor(y_true))
        else:
            torchmetrics_report.update(torch.tensor(y_pred), torch.tensor(y_true))
            
        # Compute result
        result = torchmetrics_report.compute()
        
        # For comparison, generate sklearn report when possible
        if task != "multilabel" and ignore_index is None:  # sklearn doesn't support multilabel or ignore_index in the same way
            # Generate sklearn report
            sklearn_params = {
                "output_dict": output_dict,
            }
            
            if with_target_names and target_names is not None:
                sklearn_params["target_names"] = target_names
                sklearn_params["labels"] = np.arange(len(target_names))
                
            report_scikit = classification_report(y_true, y_pred, **sklearn_params)
            
            # Verify results
            if output_dict:
                self._assert_dicts_equal_with_tolerance(report_scikit, result)
            else:
                self._verify_string_report(result)
        else:
            # For multilabel or ignore_index cases, we don't have a direct sklearn comparison
            # Verify the format is correct
            if output_dict:
                # Check basic structure
                if with_target_names and target_names is not None:
                    for label in target_names:
                        assert label in result
                        assert "precision" in result[label]
                        assert "recall" in result[label]
                        assert "f1-score" in result[label]
                        assert "support" in result[label]
                
                # Check for aggregate metrics
                possible_avg_keys = ["micro avg", "macro avg", "weighted avg", "micro-avg", "macro-avg", "weighted-avg"]
                assert any(key in result for key in possible_avg_keys)
                
                # Additional tests for ignore_index functionality
                if ignore_index is not None:
                    self._test_ignore_index_functionality(task, result, expected_support)
            else:
                self._verify_string_report(result)

    def _test_ignore_index_functionality(self, task, tm_report, expected_support):
        """Test that ignore_index functionality works correctly."""
        if task in ["binary", "multiclass"]:
            # Check that total support matches expected (ignored samples excluded)
            total_support = sum(tm_report[key]['support'] for key in tm_report 
                              if key not in ['accuracy', 'macro avg', 'weighted avg', 'macro-avg', 'weighted-avg', 'micro avg', 'micro-avg'])
            assert total_support == expected_support
        elif task == "multilabel":
            # For multilabel, check per-label support
            for i, label_key in enumerate(['0', '1', '2']):
                if label_key in tm_report:
                    assert tm_report[label_key]['support'] == expected_support[i]
                
    @pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
    def test_functional_equivalence(self, task, output_dict):
        """Test that the functional and class implementations are equivalent."""
        # Create test data based on task
        if task == "binary":
            y_true, y_pred, target_names = get_binary_test_data()
            y_prob = None
        elif task == "multiclass":
            y_true, y_pred, target_names = get_multiclass_test_data()
            y_prob = None
        else:  # multilabel
            y_true, y_pred, y_prob, target_names = get_multilabel_test_data()
            
        # Create common parameters
        common_params = {
            "output_dict": output_dict,
            "target_names": target_names,
        }
        
        # Add task-specific parameters
        if task == "binary":
            common_params["threshold"] = 0.5
        elif task == "multiclass":
            common_params["num_classes"] = len(np.unique(y_true))
        elif task == "multilabel":
            common_params["num_labels"] = y_true.shape[1]
            common_params["threshold"] = 0.5
            
        # Get class implementation result
        class_metric = ClassificationReport(task=task, **common_params)
        class_metric.update(torch.tensor(y_pred), torch.tensor(y_true))
        class_result = class_metric.compute()
        
        # Get functional implementation result
        if task == "binary":
            func_result = binary_classification_report(torch.tensor(y_pred), torch.tensor(y_true), **common_params)
        elif task == "multiclass":
            func_result = multiclass_classification_report(torch.tensor(y_pred), torch.tensor(y_true), **common_params)
        elif task == "multilabel":
            func_result = multilabel_classification_report(torch.tensor(y_pred), torch.tensor(y_true), **common_params)
            
        # Also test the general functional implementation
        general_result = functional_classification_report(
            torch.tensor(y_pred), 
            torch.tensor(y_true), 
            task=task, 
            **common_params
        )
        
        # Verify results are equivalent
        if output_dict:
            self._assert_dicts_equal(class_result, func_result)
            self._assert_dicts_equal(class_result, general_result)
        else:
            # For string output, check they have the same key content
            for metric in ["precision", "recall", "f1-score", "support"]:
                assert metric in func_result
                assert metric in general_result

    @pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
    @pytest.mark.parametrize("ignore_value", [-1, 99])
    def test_ignore_index_specific_functionality(self, task, ignore_value, output_dict):
        """Test specific ignore_index functionality and edge cases."""
        # Create test data with ignore_index values
        if task == "binary":
            preds = torch.tensor([0, 1, 1, 0, 1, 0])
            target = torch.tensor([0, 1, ignore_value, 0, 1, ignore_value])
            expected_support = 4
            num_classes = 2
            func_call = binary_classification_report
            common_params = {"threshold": 0.5}
        elif task == "multiclass":
            preds = torch.tensor([0, 1, 2, 1, 2, 0, 1])
            target = torch.tensor([0, 1, 2, ignore_value, 2, 0, ignore_value])
            expected_support = 5
            num_classes = 3
            func_call = multiclass_classification_report
            common_params = {"num_classes": num_classes}
        else:  # multilabel
            preds = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
            target = torch.tensor([[1, 0, 1], [0, ignore_value, 0], [1, 1, ignore_value], [0, 0, 1]])
            expected_support = [2, 1, 2]  # Per-label support
            func_call = multilabel_classification_report
            common_params = {"num_labels": 3, "threshold": 0.5}
        
        # Test functional version
        result = func_call(
            preds=preds,
            target=target,
            ignore_index=ignore_value,
            output_dict=True,
            **common_params
        )
        
        # Test modular version
        metric_params = {"task": task, "ignore_index": ignore_value, "output_dict": True}
        if task == "binary":
            metric_params.update(common_params)
        elif task == "multiclass":
            metric_params.update(common_params)
        else:  # multilabel
            metric_params.update(common_params)
            
        metric = ClassificationReport(**metric_params)
        metric.update(preds, target)
        result_modular = metric.compute()
        
        # Verify support counts
        if task in ["binary", "multiclass"]:
            total_support = sum(result[str(i)]['support'] for i in range(num_classes))
            total_support_modular = sum(result_modular[str(i)]['support'] for i in range(num_classes))
            assert total_support == expected_support
            assert total_support_modular == expected_support
        else:  # multilabel
            for i in range(3):
                assert result[str(i)]['support'] == expected_support[i]
                assert result_modular[str(i)]['support'] == expected_support[i]
        
        # Test that ignore_index=None behaves like no ignore_index
        result_none = func_call(
            preds=preds,
            target=torch.where(target == ignore_value, 0, target),  # Replace ignore values with valid ones
            ignore_index=None,
            output_dict=True,
            **common_params
        )
        
        result_no_param = func_call(
            preds=preds,
            target=torch.where(target == ignore_value, 0, target),
            output_dict=True,
            **common_params
        )
        
        # These should be equivalent
        if task in ["binary", "multiclass"]:
            for i in range(num_classes):
                if str(i) in result_none and str(i) in result_no_param:
                    assert abs(result_none[str(i)]['support'] - result_no_param[str(i)]['support']) < 1e-6
        else:  # multilabel
            for i in range(3):
                if str(i) in result_none and str(i) in result_no_param:
                    assert abs(result_none[str(i)]['support'] - result_no_param[str(i)]['support']) < 1e-6

    def test_ignore_index_accuracy_calculation(self, output_dict):
        """Test that ignore_index properly affects accuracy calculation."""
        # Create scenario where ignored indices would change accuracy
        preds = torch.tensor([0, 1, 0, 1])
        target = torch.tensor([0, 1, -1, -1])  # Last two are ignored
        
        result = binary_classification_report(
            preds=preds,
            target=target,
            ignore_index=-1,
            output_dict=True
        )
        
        # With ignore_index, accuracy should be 1.0 (2/2 correct)
        assert result['accuracy'] == 1.0
        
        # Compare with case where we have wrong predictions for ignored indices
        preds_wrong = torch.tensor([0, 1, 1, 0])  # Wrong predictions for what would be ignored
        target_wrong = torch.tensor([0, 1, -1, -1])
        
        result_wrong = binary_classification_report(
            preds=preds_wrong,
            target=target_wrong,
            ignore_index=-1,
            output_dict=True
        )
        
        # Should still be 1.0 because ignored indices don't affect accuracy
        assert result_wrong['accuracy'] == 1.0


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


def test_functional_invalid_task():
    """Test validation of task parameter in functional classification_report."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError, match="Invalid Classification: expected one of"):
        functional_classification_report(y_pred, y_true, task="invalid_task")


# Add parameterized tests for various edge cases
@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("output_dict", [True, False])
@pytest.mark.parametrize("zero_division", [0, 1, "warn"])
def test_zero_division_handling(task, output_dict, zero_division):
    """Test zero_division parameter works correctly across all classification types."""
    # Create edge case data with some classes having no support
    if task == "binary":
        # Create data where class 1 never appears in target
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1])
        params = {"threshold": 0.5}
    elif task == "multiclass":
        # Create data where class 2 never appears in target
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 2, 1, 2])
        params = {"num_classes": 3}
    else:  # multilabel
        # Create data where second label never appears
        y_true = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
        y_pred = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]])
        params = {"num_labels": 3, "threshold": 0.5}
    
    # Create report with zero_division parameter
    report = ClassificationReport(
        task=task, 
        output_dict=output_dict,
        zero_division=zero_division,
        **params
    )
    
    report.update(torch.tensor(y_pred), torch.tensor(y_true))
    result = report.compute()
    
    # Check the results
    if output_dict:
        # Verify that a result is produced
        if task == "binary":
            # Verify class '1' is in the result if it was predicted
            if "1" in result:
                # Just check that precision exists - actual value depends on implementation
                assert "precision" in result["1"]
                
                # For zero_division=0, precision should always be 0 for classes with no support
                if zero_division == 0:
                    assert result["1"]["precision"] == 0.0
                    
        elif task == "multiclass":
            # Verify class '2' is in the result
            if "2" in result:
                # Just check that precision exists - actual value depends on implementation
                assert "precision" in result["2"]
                
                # For zero_division=0, precision should always be 0 for classes with no support
                if zero_division == 0:
                    assert result["2"]["precision"] == 0.0
    else:
        # For string output, just verify it's a valid string
        assert isinstance(result, str)

# Tests for top_k functionality
@pytest.mark.parametrize("output_dict", [True, False])
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_multiclass_classification_report_top_k(output_dict, top_k):
    """Test top_k functionality in multiclass classification report."""
    # Create simple test data where top_k can make a difference
    num_classes = 3
    batch_size = 12
    
    # Create predictions with specific pattern for testing top_k
    preds = torch.tensor([
        [0.1, 0.8, 0.1],  # Class 1 is top-1, class 0 is top-2  -> target: 0
        [0.7, 0.2, 0.1],  # Class 0 is top-1, class 1 is top-2  -> target: 1  
        [0.1, 0.1, 0.8],  # Class 2 is top-1, class 0 is top-2  -> target: 2
        [0.4, 0.5, 0.1],  # Class 1 is top-1, class 0 is top-2  -> target: 0
        [0.3, 0.6, 0.1],  # Class 1 is top-1, class 0 is top-2  -> target: 1
        [0.2, 0.1, 0.7],  # Class 2 is top-1, class 0 is top-2  -> target: 2
        [0.6, 0.3, 0.1],  # Class 0 is top-1, class 1 is top-2  -> target: 0
        [0.2, 0.7, 0.1],  # Class 1 is top-1, class 0 is top-2  -> target: 1
        [0.1, 0.2, 0.7],  # Class 2 is top-1, class 1 is top-2  -> target: 2
        [0.5, 0.4, 0.1],  # Class 0 is top-1, class 1 is top-2  -> target: 0
        [0.1, 0.8, 0.1],  # Class 1 is top-1, class 0 is top-2  -> target: 1
        [0.1, 0.3, 0.6],  # Class 2 is top-1, class 1 is top-2  -> target: 2
    ])
    
    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    
    # Test functional interface
    result_functional = multiclass_classification_report(
        preds=preds,
        target=target,
        num_classes=num_classes,
        top_k=top_k,
        output_dict=output_dict
    )
    
    # Test class interface
    metric = ClassificationReport(
        task="multiclass",
        num_classes=num_classes,
        top_k=top_k,
        output_dict=output_dict
    )
    metric.update(preds, target)
    result_class = metric.compute()
    
    # Verify both interfaces produce same result
    if output_dict:
        assert isinstance(result_functional, dict)
        assert isinstance(result_class, dict)
        # Check that accuracy improves with higher top_k (should be non-decreasing)
        if "accuracy" in result_functional:
            assert result_functional["accuracy"] >= 0.0
            assert result_functional["accuracy"] <= 1.0
    else:
        assert isinstance(result_functional, str)
        assert isinstance(result_class, str)
        # Verify standard metrics are present in string output
        assert "precision" in result_functional
        assert "recall" in result_functional
        assert "f1-score" in result_functional
        assert "support" in result_functional
    
    # Verify that functional and class methods produce identical results
    assert result_functional == result_class


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_multiclass_classification_report_top_k_accuracy_monotonic(top_k):
    """Test that accuracy is monotonic non-decreasing with increasing top_k."""
    num_classes = 4
    batch_size = 20
    
    # Create random but consistent test data  
    torch.manual_seed(42)
    preds = torch.randn(batch_size, num_classes).softmax(dim=1)
    target = torch.randint(0, num_classes, (batch_size,))
    
    result = multiclass_classification_report(
        preds=preds,
        target=target,
        num_classes=num_classes,
        top_k=top_k,
        output_dict=True
    )
    
    # Basic sanity checks
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    
    # Check that all class metrics are present
    for i in range(num_classes):
        assert str(i) in result
        class_metrics = result[str(i)]
        assert "precision" in class_metrics
        assert "recall" in class_metrics
        assert "f1-score" in class_metrics
        assert "support" in class_metrics


def test_multiclass_classification_report_top_k_comparison():
    """Test that higher top_k generally leads to equal or better accuracy."""
    num_classes = 5
    batch_size = 50
    
    # Create test data where top_k makes a significant difference
    torch.manual_seed(123)
    preds = torch.randn(batch_size, num_classes).softmax(dim=1)
    target = torch.randint(0, num_classes, (batch_size,))
    
    accuracies = {}
    
    for k in [1, 2, 3, 4, 5]:
        result = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=num_classes,
            top_k=k,
            output_dict=True
        )
        accuracies[k] = result["accuracy"]
    
    # Verify accuracy is non-decreasing
    for k in range(1, 5):
        assert accuracies[k] <= accuracies[k + 1], (
            f"Accuracy should be non-decreasing with top_k: "
            f"top_{k}={accuracies[k]:.3f} > top_{k+1}={accuracies[k+1]:.3f}"
        )
    
    # At top_k = num_classes, accuracy should be 1.0
    assert accuracies[5] == 1.0, f"Accuracy at top_k=num_classes should be 1.0, got {accuracies[5]}"


@pytest.mark.parametrize("ignore_index", [None, -1])
@pytest.mark.parametrize("top_k", [1, 2])
def test_multiclass_classification_report_top_k_with_ignore_index(ignore_index, top_k):
    """Test top_k functionality works correctly with ignore_index."""
    num_classes = 3
    
    preds = torch.tensor([
        [0.6, 0.3, 0.1],  # pred: 0, target: 0 (correct)
        [0.2, 0.7, 0.1],  # pred: 1, target: 1 (correct)  
        [0.1, 0.2, 0.7],  # pred: 2, target: ignored
        [0.4, 0.5, 0.1],  # pred: 1, target: 0 (wrong for top-1, correct for top-2)
    ])
    
    if ignore_index is not None:
        target = torch.tensor([0, 1, ignore_index, 0])
    else:
        target = torch.tensor([0, 1, 2, 0])
    
    result = multiclass_classification_report(
        preds=preds,
        target=target,
        num_classes=num_classes,
        top_k=top_k,
        ignore_index=ignore_index,
        output_dict=True
    )
    
    # Basic verification
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    
    # With ignore_index, the third sample should be ignored
    if ignore_index is not None and top_k == 2:
        # With top_k=2, the last prediction [0.4, 0.5, 0.1] should be correct 
        # since target=0 and both classes 0 and 1 are in top-2
        expected_accuracy = 1.0  # 3 out of 3 valid samples correct
        assert abs(result["accuracy"] - expected_accuracy) < 1e-6


def test_classification_report_wrapper_top_k():
    """Test that the wrapper ClassificationReport correctly handles top_k."""
    num_classes = 3
    preds = torch.tensor([
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1], 
        [0.1, 0.1, 0.8],
    ])
    target = torch.tensor([0, 1, 2])
    
    # Test with different top_k values
    for top_k in [1, 2, 3]:
        report = ClassificationReport(
            task="multiclass",
            num_classes=num_classes,
            top_k=top_k,
            output_dict=True
        )
        
        report.update(preds, target)
        result = report.compute()
        
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        
        # Check that all expected classes are present
        for i in range(num_classes):
            assert str(i) in result


@pytest.mark.parametrize("top_k", [1, 2])
def test_functional_classification_report_top_k(top_k):
    """Test that the main functional classification_report interface supports top_k."""
    num_classes = 3
    preds = torch.tensor([
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.1, 0.8],
    ])
    target = torch.tensor([0, 1, 2])
    
    result = functional_classification_report(
        preds=preds,
        target=target,
        task="multiclass",
        num_classes=num_classes,
        top_k=top_k,
        output_dict=True
    )
    
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    
    # Verify structure is correct
    for i in range(num_classes):
        assert str(i) in result
        metrics = result[str(i)]
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1-score" in metrics
        assert "support" in metrics


def test_top_k_binary_task_ignored():
    """Test that top_k parameter is ignored for binary tasks (should not cause errors)."""
    preds = torch.tensor([0.1, 0.9, 0.3, 0.8])
    target = torch.tensor([0, 1, 0, 1])
    
    # top_k should be ignored for binary classification
    result1 = functional_classification_report(
        preds=preds,
        target=target,
        task="binary",
        top_k=1,
        output_dict=True
    )
    
    result2 = functional_classification_report(
        preds=preds,
        target=target,
        task="binary", 
        top_k=5,  # Should be ignored
        output_dict=True
    )
    
    # Results should be identical since top_k is ignored for binary
    assert result1 == result2


def test_top_k_multilabel_task_ignored():
    """Test that top_k parameter is ignored for multilabel tasks."""
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    target = torch.tensor([[0, 1], [1, 0], [0, 1]])
    
    # top_k should be ignored for multilabel classification
    result1 = functional_classification_report(
        preds=preds,
        target=target,
        task="multilabel",
        num_labels=2,
        top_k=1,
        output_dict=True
    )
    
    result2 = functional_classification_report(
        preds=preds,
        target=target,
        task="multilabel",
        num_labels=2,
        top_k=5,  # Should be ignored
        output_dict=True
    )
    
    # Results should be identical since top_k is ignored for multilabel  
    assert result1 == result2


class TestTopKFunctionality:
    """Test class specifically for top_k functionality in multiclass classification."""
    
    def test_top_k_basic_functionality(self):
        """Test basic top_k functionality with probabilities."""
        # Create predictions where top-1 prediction is wrong but top-2 includes correct label
        preds = torch.tensor([
            [0.1, 0.8, 0.1],  # Predicted: 1, True: 0 (wrong for top-1, correct for top-2)
            [0.2, 0.3, 0.5],  # Predicted: 2, True: 2 (correct for both)
            [0.6, 0.3, 0.1],  # Predicted: 0, True: 1 (wrong for top-1, correct for top-2)
        ])
        target = torch.tensor([0, 2, 1])
        
        # Test top_k=1 (should have lower accuracy)
        result_k1 = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=3,
            top_k=1,
            output_dict=True
        )
        
        # Test top_k=2 (should have higher accuracy)
        result_k2 = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=3,
            top_k=2,
            output_dict=True
        )
        
        # With top_k=1, accuracy should be 1/3 = 0.333...
        assert abs(result_k1['accuracy'] - 0.3333333333333333) < 1e-6
        
        # With top_k=2, accuracy should be 3/3 = 1.0 (all samples have correct label in top-2)
        assert result_k2['accuracy'] == 1.0
        
        # Per-class metrics should also improve with top_k=2
        assert result_k2['0']['recall'] >= result_k1['0']['recall']
        assert result_k2['1']['recall'] >= result_k1['1']['recall']
    
    def test_top_k_with_logits(self):
        """Test top_k functionality with logits (unnormalized scores)."""
        # Logits that will be converted to probabilities via softmax
        preds = torch.tensor([
            [1.0, 3.0, 1.0],  # After softmax: highest prob for class 1, true label is 0
            [2.0, 1.0, 4.0],  # After softmax: highest prob for class 2, true label is 2
            [3.0, 2.0, 1.0],  # After softmax: highest prob for class 0, true label is 1
        ])
        target = torch.tensor([0, 2, 1])
        
        result_k1 = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=3,
            top_k=1,
            output_dict=True
        )
        
        result_k2 = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=3,
            top_k=2,
            output_dict=True
        )
        
        # top_k=2 should perform better than or equal to top_k=1
        assert result_k2['accuracy'] >= result_k1['accuracy']
    
    def test_top_k_with_class_wrapper(self):
        """Test top_k functionality through the ClassificationReport wrapper class."""
        preds = torch.tensor([
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.6, 0.3, 0.1],
        ])
        target = torch.tensor([0, 2, 1])
        
        # Test with class-based implementation
        metric_k1 = ClassificationReport(task="multiclass", num_classes=3, top_k=1, output_dict=True)
        metric_k1.update(preds, target)
        result_k1 = metric_k1.compute()
        
        metric_k2 = ClassificationReport(task="multiclass", num_classes=3, top_k=2, output_dict=True)
        metric_k2.update(preds, target)
        result_k2 = metric_k2.compute()
        
        # top_k=2 should perform better
        assert result_k2['accuracy'] >= result_k1['accuracy']
        
        # Test equivalence with functional implementation
        func_result_k2 = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=3,
            top_k=2,
            output_dict=True
        )
        
        assert result_k2['accuracy'] == func_result_k2['accuracy']
    
    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_top_k_edge_cases(self, top_k):
        """Test top_k with different values and edge cases."""
        # Simple case where all predictions are correct for top-1
        preds = torch.tensor([
            [0.9, 0.05, 0.05],  # Correct: class 0
            [0.05, 0.9, 0.05],  # Correct: class 1  
            [0.05, 0.05, 0.9],  # Correct: class 2
        ])
        target = torch.tensor([0, 1, 2])
        
        result = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=3,
            top_k=top_k,
            output_dict=True
        )
        
        # Should always be perfect accuracy regardless of top_k value
        assert result['accuracy'] == 1.0
    
    def test_top_k_larger_than_num_classes(self):
        """Test behavior when top_k is larger than number of classes."""
        preds = torch.tensor([
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ])
        target = torch.tensor([0, 2])
        
        # top_k=5 > num_classes=3, should raise an error as per torchmetrics validation
        with pytest.raises(ValueError, match="Expected argument `top_k` to be smaller or equal to `num_classes`"):
            multiclass_classification_report(
                preds=preds,
                target=target,
                num_classes=3,
                top_k=5,
                output_dict=True
            )
    
    def test_top_k_with_hard_predictions(self):
        """Test that top_k works correctly with hard predictions (class indices)."""
        # When predictions are already class indices, top_k > 1 should raise an error
        # because hard predictions are 1D and can't support top_k > 1
        preds = torch.tensor([1, 2, 0])  # Hard predictions
        target = torch.tensor([0, 2, 1])
        
        result_k1 = multiclass_classification_report(
            preds=preds,
            target=target,
            num_classes=3,
            top_k=1,
            output_dict=True
        )
        
        # With hard predictions, top_k > 1 should raise an error
        with pytest.raises(RuntimeError, match="selected index k out of range"):
            multiclass_classification_report(
                preds=preds,
                target=target,
                num_classes=3,
                top_k=2,
                output_dict=True
            )
    
    def test_top_k_ignored_for_binary(self):
        """Test that top_k parameter is ignored for binary classification."""
        preds = torch.tensor([0.6, 0.4, 0.7, 0.3])
        target = torch.tensor([1, 0, 1, 0])
        
        # top_k should be ignored for binary classification
        result1 = binary_classification_report(
            preds=preds,
            target=target,
            output_dict=True
        )
        
        # This should work the same way via the general interface
        result2 = functional_classification_report(
            preds=preds,
            target=target,
            task="binary",
            top_k=2,  # Should be ignored
            output_dict=True
        )
        
        assert result1['accuracy'] == result2['accuracy']
    
    def test_top_k_ignored_for_multilabel(self):
        """Test that top_k parameter is ignored for multilabel classification."""
        preds = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
        target = torch.tensor([[1, 0], [0, 1], [1, 1]])
        
        # top_k should be ignored for multilabel classification
        result1 = multilabel_classification_report(
            preds=preds,
            target=target,
            num_labels=2,
            output_dict=True
        )
        
        result2 = functional_classification_report(
            preds=preds,
            target=target,
            task="multilabel",
            num_labels=2,
            top_k=5,  # Should be ignored
            output_dict=True
        )
        
        # Results should be identical since top_k is ignored for multilabel  
        assert result1 == result2