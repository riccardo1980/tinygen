import argparse
import json
import logging
import os
from typing import Dict

import pandas as pd
import tensorflow as tf

from tinygen.io.tfrecords import do_pipeline


class parameters(object):
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


def preprocess_parameters(args: Dict) -> parameters:
    pars = parameters(args)
    return pars


def run(pars: parameters) -> None:
    """
    Run preprocess

    :param subparsers: subparsers
    :type subparsers: argparse._SubParsersAction
    """

    # read file
    df = pd.DataFrame(pd.read_csv(pars.input_file))
    logging.info(f"Read {len(df)} rows")

    # read class mapping
    label_to_index = pars.class_mapping

    # formatting / serialization pipeline
    records = df.to_dict(orient="records")
    serialized_entries = do_pipeline(label_to_index, records)

    # write tfrecords
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
