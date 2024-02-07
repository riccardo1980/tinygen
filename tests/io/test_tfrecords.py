from typing import Any, Callable, Dict, List

import pytest
import tensorflow as tf

from tinygen.io.tfrecords import do_pipeline, make_formatter, read_tfrecord
from tinygen.preprocess import make_class_filter


@pytest.mark.parametrize(
    "records,label_to_index,expected",
    [
        (
            [
                {"label": "ham", "text": "message 1"},
                {"label": "ham", "text": "message 2"},
                {"label": "spam", "text": "message 3"},
                {"label": "pepperoni", "text": "message 4"},
            ],
            {"ham": 0, "spam": 1},
            [
                {"label": "ham", "text": "message 1"},
                {"label": "ham", "text": "message 2"},
                {"label": "spam", "text": "message 3"},
            ],
        )
    ],
)
def test_filter(
    records: List[Dict[str, str]],
    label_to_index: Dict[str, int],
    expected: List[Dict[str, Any]],
) -> None:
    allowed_labels = set(label_to_index.keys())
    got = list(filter(lambda entry: entry["label"] in allowed_labels, records))

    assert got == expected


@pytest.mark.parametrize(
    "records,label_to_index,expected",
    [
        (
            [
                {"label": "ham", "text": "message 1"},
                {"label": "ham", "text": "message 2"},
                {"label": "spam", "text": "message 3"},
            ],
            {"ham": 0, "spam": 1},
            [
                {"label": 0, "text": "message 1"},
                {"label": 0, "text": "message 2"},
                {"label": 1, "text": "message 3"},
            ],
        )
    ],
)
def test_formatter(
    records: List[Dict[str, str]],
    label_to_index: Dict[str, int],
    expected: List[Dict[str, Any]],
) -> None:
    formatter = make_formatter(label_to_index)
    got = list(map(formatter, records))
    assert got == expected


@pytest.mark.parametrize(
    "records,class_mapping",
    [
        (
            [
                {"label": "ham", "text": "message 1"},
                {"label": "ham", "text": "message 2 longer than the others"},
                {"label": "spam", "text": "message 3"},
            ],
            {"ham": 0, "spam": 1},
        )
    ],
)
def test_roundtrip(
    records: List[Dict[str, str]], class_mapping: Dict[str, int]
) -> None:
    # entries to encoded tfrecords
    encoded = do_pipeline(class_mapping, records)

    # decode to examples
    decoded = map(read_tfrecord, encoded)

    # tensor to numpy
    tensor_to_numpy: Callable[
        [tuple[tf.Tensor, tf.Tensor]], tuple[bytes, int]
    ] = lambda x: (
        x[0].numpy(),
        x[1].numpy(),
    )
    numpys = map(tensor_to_numpy, decoded)

    # change labels to string from int
    # change text to string from bytes
    index_to_label: Dict[int, str] = {v: k for k, v in class_mapping.items()}
    deformatter: Callable[[tuple[bytes, int]], tuple[str, str]] = lambda x: (
        x[0].decode("utf-8"),
        index_to_label[x[1]],
    )

    decoded = map(deformatter, numpys)

    expected = map(lambda x: (x["text"], x["label"]), records)

    assert list(decoded) == list(expected)


@pytest.mark.parametrize(
    "allowed_classes,input_records,expected_records",
    [
        (
            ["a", "b"],
            [
                {"label": "a", "text": "aaa aaaa"},
                {"label": "b", "text": "bbb bbb"},
                {"label": "c", "text": "ccc ccc"},
            ],
            [
                {"label": "a", "text": "aaa aaaa"},
                {"label": "b", "text": "bbb bbb"},
            ],
        ),
        (
            ["d", "e"],
            [
                {"label": "a", "text": "aaa aaaa"},
                {"label": "b", "text": "bbb bbb"},
                {"label": "c", "text": "ccc ccc"},
            ],
            [],
        ),
    ],
)
def test_class_filter(
    allowed_classes: List[str],
    input_records: List[Dict],
    expected_records: List[Dict],
) -> None:
    """
    Test for the application of class filter
    """
    filter = make_class_filter(allowed_classes)
    got = list(filter(input_records))

    assert expected_records == got
