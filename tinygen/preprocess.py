import argparse
import json
import logging
import os
from collections import Counter
from itertools import tee
from typing import Callable, Dict, Iterable, List

import pandas as pd
import tensorflow as tf

from tinygen.io.tfrecords import do_pipeline


# FIXME: this is the application of the filter
# maybe call filter the lambda, then call built-in filter function directly here
def make_class_filter(allowed_classes: List) -> Callable:
    """ """
    allowed_labels = set(allowed_classes)

    def _filter(records: Iterable) -> Iterable:
        """
        Filter out examples not pertaining to one of allowed classes
        """
        return filter(lambda entry: entry["label"] in allowed_labels, records)

    return _filter


class Parameters(object):
    input_file: str
    output_path: str
    class_mapping: Dict

    def __init__(self, params: Dict) -> None:
        # manage reformatting
        configs = {
            "input_file": params.pop("input_file"),
            "output_path": params.pop("output_path"),
            "class_mapping": json.loads(params.pop("class_mapping")),
        }

        # set attributes
        self.__dict__.update(configs)

    def __repr__(self) -> str:
        return str(self.__dict__)


def preprocess_parameters(args: Dict) -> Parameters:
    pars = Parameters(args)
    return pars


def run(pars: Parameters) -> None:
    """
    Run preprocess

    :param subparsers: subparsers
    :type subparsers: argparse._SubParsersAction
    """
    # read file
    df = pd.DataFrame(pd.read_csv(pars.input_file))
    logging.info(f"Read {len(df)} rows")

    # read
    records = df.to_dict(orient="records")

    # filter
    record_filter = make_class_filter(list(pars.class_mapping.keys()))
    filtered = record_filter(records)

    # create two output branches
    to_be_counted, to_be_written = tee(filtered, 2)

    # stats
    counter = Counter(map(lambda entry: entry["label"], to_be_counted))
    logging.info(f"Items per class: {dict(counter)}")

    # formatting, tf.examples, serialization
    serialized_entries = do_pipeline(pars.class_mapping, to_be_written)

    # write - actual write
    tfrecords_path = os.path.join(pars.output_path, "tfrecords")
    os.makedirs(tfrecords_path, exist_ok=True)
    with tf.io.TFRecordWriter(os.path.join(tfrecords_path, "data.tfrecords")) as writer:
        o = list(map(lambda entry: writer.write(entry), serialized_entries))
        logging.info(f"Wrote {len(o)} records")


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Build argument parser

    :return: parser
    :rtype: argparse.ArgumentParser
    """
    p = subparsers.add_parser("preprocess", help="Preprocess data")

    p.add_argument("--input_file", type=str, required=True, help="Input file")
    p.add_argument(
        "--class_mapping",
        type=str,
        required=True,
        help="""
    Class mapping as JSON string: example: '{\"class1\": 0, \"class2\": 1}\',
    """,
    )
    p.add_argument("--output_path", type=str, required=True, help="Output path")
