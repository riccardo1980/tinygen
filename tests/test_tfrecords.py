from typing import Any, Callable, Dict, List

import pytest
import tensorflow as tf

from tinygen.io.tfrecords import do_pipeline, make_formatter, read_tfrecord


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
    "records,label_to_index",
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
    records: List[Dict[str, str]], label_to_index: Dict[str, int]
) -> None:
    # entries to encoded tfrecords
    encoded = do_pipeline(label_to_index, records)

    # decode to examples
    decoded = map(read_tfrecord, encoded)

    # tensor to numpy
    tensor_to_numpy: Callable[
        [tuple[tf.Tensor, tf.Tensor]], tuple[int, bytes]
    ] = lambda x: (
        x[0].numpy(),
        x[1].numpy(),
    )
    numpys = map(tensor_to_numpy, decoded)

    # change labels to string from int
    # change text to string from bytes
    index_to_label: Dict[int, str] = {v: k for k, v in label_to_index.items()}
    deformatter: Callable[[tuple[int, bytes]], tuple[str, str]] = lambda x: (
        index_to_label[x[0]],
        x[1].decode("utf-8"),
    )

    decoded = map(deformatter, numpys)

    expected = map(lambda x: (x["label"], x["text"]), records)

    assert list(decoded) == list(expected)
