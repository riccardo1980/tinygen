from typing import List

import pytest

from tinygen.metrics import PerClassPrecision, PerClassRecall


def test_per_class_precision_init() -> None:
    precision = PerClassPrecision(class_id=0)
    assert precision.class_id == 0


# fmt: off
@pytest.mark.parametrize(
    "y_true, y_pred, precisions",
    [
        ([[0.1, 0.9], [0.9, 0.1]],  # 1 0
         [[0.1, 0.9], [0.9, 0.1]],  # 1 0
         [1.0, 1.0]),
        ([[0.1, 0.9], [0.9, 0.1]],  # 1 0
         [[0.1, 0.9], [0.1, 0.9]],  # 1 1
         [0.0, 0.5]),
        (
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],  # 1 0 0
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],  # 1 0 0
            [1.0, 1.0],
        ),
        (
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],  # 1 0 0
            [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],  # 1 0 1
            [1.0, 0.5],
        ),
    ],
)
# fmt: on
def test_per_class_precision(y_true: List, y_pred: List, precisions: List) -> None:
    num_classes = len(y_true[0])
    precision_metrics = [PerClassPrecision(class_id=i) for i in range(num_classes)]
    for i in range(num_classes):
        precision_metrics[i].update_state(y_true, y_pred)
        assert precision_metrics[i].result().numpy() == precisions[i]


def test_per_class_recall_init() -> None:
    recall = PerClassRecall(class_id=0)
    assert recall.class_id == 0


# fmt: off
@pytest.mark.parametrize(
    "y_true, y_pred, recalls",
    [
        ([[0.1, 0.9], [0.9, 0.1]],  # 1 0
         [[0.1, 0.9], [0.9, 0.1]],  # 1 0
         [1.0, 1.0]),
        ([[0.1, 0.9], [0.9, 0.1]],  # 1 0
         [[0.1, 0.9], [0.1, 0.9]],  # 1 1
         [0.0, 1.0]),
        (
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],  # 1 0 0
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],  # 1 0 0
            [1.0, 1.0],
        ),
        (
            [[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],  # 1 0 0
            [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],  # 1 0 1
            [0.5, 1.0],
        ),
    ],
)
# fmt: on
def test_per_class_recall(y_true: List, y_pred: List, recalls: List) -> None:
    num_classes = len(y_true[0])
    recall_metrics = [PerClassRecall(class_id=i) for i in range(num_classes)]
    for i in range(num_classes):
        recall_metrics[i].update_state(y_true, y_pred)
        assert recall_metrics[i].result().numpy() == recalls[i]
