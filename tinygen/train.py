import argparse
from typing import Dict


class parameters(object):
    train_dataset_path: str
    eval_dataset_path: str
    output_path: str
    num_classes: int
    shuffle_buffer_size: int
    batch_size: int
    epochs: int

    def __init__(self, params: Dict) -> None:
        # manage reformatting
        configs = {
            "train_dataset_path": params.pop("train_dataset_path"),
            "eval_dataset_path": params.pop("eval_dataset_path"),
            "output_path": params.pop("output_path"),
            "num_classes": params.pop("num_classes"),
            "shuffle_buffer_size": params.pop("shuffle_buffer_size"),
            "batch_size": params.pop("batch_size"),
            "epochs": params.pop("epochs"),
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

    # read train dataset
    # read eval dataset

    # build model

    # train

    # save

    pass


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Build argument parser

    :param subparsers: subparsers
    :type subparsers: argparse._SubParsersAction
    """

    p = subparsers.add_parser("train", help="Train")

    p.add_argument("--train_dataset_path", type=str, required=True)
    p.add_argument("--eval_dataset_path", type=str, required=False, default=None)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--shuffle_buffer_size", type=int, required=False, default=0)
    p.add_argument("--batch_size", type=int, required=False, default=32)
    p.add_argument("--epochs", type=int, required=False, default=10)
