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
            "train_dataset_path": params.pop("input_file"),
            "eval_dataset_path": params.pop("input_file", None),
            "output_path": params.pop("output_path"),
            "num_classes": params.pop("num_classes"),
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

    :param pars: parameters
    :type pars: Parameters
    """

    pass


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Build argument parser

    :param subparsers: subparsers
    :type subparsers: argparse._SubParsersAction
    """

    p = subparsers.add_parser("train", help="Train")

    p.add_argument("--train_dataset_path", type=str, required=True)
    p.add_argument("--eval_dataset_path", type=str, required=False)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
