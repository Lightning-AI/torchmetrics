import pytest
import torch

from torchmetrics import Accuracy, ClasswiseWrapper, MetricCollection, Recall


def test_raises_error_on_wrong_input():
    """Test that errors are raised on wrong input."""
    with pytest.raises(ValueError, match="Expected argument `metric` to be an instance of `torchmetrics.Metric` but.*"):
        ClasswiseWrapper([])

    with pytest.raises(ValueError, match="Expected argument `labels` to either be `None` or a list of strings.*"):
        ClasswiseWrapper(Accuracy(), "hest")


def test_output_no_labels():
    """Test that wrapper works with no label input."""
    metric = ClasswiseWrapper(Accuracy(num_classes=3, average=None))
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    val = metric(preds, target)
    assert isinstance(val, dict)
    assert len(val) == 3
    for i in range(3):
        assert f"accuracy_{i}" in val


def test_output_with_labels():
    """Test that wrapper works with label input."""
    labels = ["horse", "fish", "cat"]
    metric = ClasswiseWrapper(Accuracy(num_classes=3, average=None), labels=labels)
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    val = metric(preds, target)
    assert isinstance(val, dict)
    assert len(val) == 3
    for lab in labels:
        assert f"accuracy_{lab}" in val


@pytest.mark.parametrize("prefix", [None, "pre_"])
@pytest.mark.parametrize("postfix", [None, "_post"])
def test_using_metriccollection(prefix, postfix):
    """Test wrapper in combination with metric collection."""
    labels = ["horse", "fish", "cat"]
    metric = MetricCollection(
        {
            "accuracy": ClasswiseWrapper(Accuracy(num_classes=3, average=None), labels=labels),
            "recall": ClasswiseWrapper(Recall(num_classes=3, average=None), labels=labels),
        },
        prefix=prefix,
        postfix=postfix,
    )
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    val = metric(preds, target)
    assert isinstance(val, dict)
    assert len(val) == 6

    def _get_correct_name(base):
        name = base if prefix is None else prefix + base
        name = name if postfix is None else name + postfix
        return name

    for lab in labels:
        name = _get_correct_name(f"accuracy_{lab}")
        assert name in val
        name = _get_correct_name(f"recall_{lab}")
        assert name in val
