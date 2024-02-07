from typing import Any, Callable, Dict, Iterable, List

import tensorflow as tf

# helper functions


def bytes_feature(value: str) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(entry: Dict[str, Any]) -> tf.train.Example:
    feature = {
        "label": int64_feature(entry["label"]),
        "text": bytes_feature(entry["text"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def make_formatter(label_to_index: Dict[str, int]) -> Callable:
    def _formatter(entry: Dict) -> Dict[str, Any]:
        formatted_entry: Dict[str, Any] = {}
        formatted_entry["label"] = label_to_index[entry["label"]]
        formatted_entry["text"] = entry["text"]
        return formatted_entry

    return _formatter


def do_pipeline(class_mapping: Dict, records: Iterable) -> List[str]:
    formatter = make_formatter(class_mapping)
    serializer: Callable[
        [tf.train.Example], str
    ] = lambda entry: entry.SerializeToString()

    # format and serialize
    formatted_entries = map(formatter, records)
    examples = map(create_example, formatted_entries)
    serialized_entries = map(serializer, examples)

    return list(serialized_entries)


def read_tfrecord(example: tf.train.Example) -> tuple[tf.Tensor, ...]:
    tfrecord_format = {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "text": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    text = example["text"]
    label = example["label"]
    return (text, label)
