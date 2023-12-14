import argparse
import json
import logging
import os
from typing import Dict

import pandas as pd
import tensorflow as tf

from tinygen.io.tfrecords import do_pipeline
from tinygen.io.utils import extract_labels

# Parameters


class Parameters(object):
    input_file: str
    output_path: str

    def __init__(self, configs: Dict) -> None:
        self.__dict__.update(configs)

    def __repr__(self) -> str:
        return str(self.__dict__)


def run(pars: Parameters) -> None:
    """
    Run preprocess

    :param pars: parameters
    :type pars: Parameters
    """
    # read file
    df = pd.DataFrame(pd.read_csv(pars.input_file))
    logging.info(f"Read {len(df)} rows")

    # get integer to class mapping
    label_to_index = extract_labels(df["label"])
    logging.info(f"Found {len(label_to_index)} labels")

    # write mapping file
    os.makedirs(pars.output_path, exist_ok=True)
    with open(os.path.join(pars.output_path, "label_to_index.json"), "w") as f:
        json.dump(label_to_index, f)

    # formatting / serialization pipeline
    records = df.to_dict(orient="records")
    serialized_entries = do_pipeline(label_to_index, records)

    # write tfrecords
    tfrecords_path = os.path.join(pars.output_path, "tfrecords")
    os.makedirs(tfrecords_path, exist_ok=True)
    with tf.io.TFRecordWriter(os.path.join(tfrecords_path, "data.tfrecords")) as writer:
        o = list(map(lambda entry: writer.write(entry), serialized_entries))
        logging.info(f"Wrote {len(o)} records")


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser

    :return: parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--input_file", type=str, required=True, help="Input file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path")

    return parser


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
    )

    parser = build_parser()
    args = parser.parse_args()

    pars = Parameters(vars(args))
    logging.info(pars)

    run(pars)
