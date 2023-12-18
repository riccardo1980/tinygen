import argparse
import os
from typing import Dict

from tinygen.io.utils import convert_to_fuse

# Parameters


class parameters(object):
    train_dataset_path: str
    eval_dataset_path: str
    num_classes: int
    shuffle_buffer_size: int
    batch_size: int
    epochs: int
    model_path: str
    logs_path: str
    checkpoints_path: str

    # constructor

    def __init__(self, params: Dict) -> None:
        # manage reformatting
        configs = {
            "train_dataset_path": convert_to_fuse(params.pop("train_dataset_path")),
            "eval_dataset_path": convert_to_fuse(params.pop("eval_dataset_path")),
            "num_classes": params.pop("num_classes"),
            "shuffle_buffer_size": params.pop("shuffle_buffer_size"),
            "batch_size": params.pop("batch_size"),
            "epochs": params.pop("epochs"),
        }

        output_subfolders: Dict[str, str] = {}
        if params["output_path"]:
            output_subfolders = {
                "model_path": os.path.join(
                    convert_to_fuse(params["output_path"]), "model"
                ),
                "logs_path": os.path.join(
                    convert_to_fuse(params["output_path"]), "logs"
                ),
                "checkpoints_path": os.path.join(
                    convert_to_fuse(params["output_path"]), "checkpoints"
                ),
            }
        else:
            # use environment variables
            def _assert_is_present(key: str) -> str:
                out: str = os.getenv(key, "")
                assert out != "", f"{key} is not set"
                return out

            output_subfolders = {
                "model_path": _assert_is_present("AIP_MODEL_DIR"),
                "logs_path": _assert_is_present("AIP_TENSORBOARD_LOG_DIR"),
                "checkpoints_path": _assert_is_present("AIP_CHECKPOINT_DIR"),
            }

        configs.update(output_subfolders)

        # set attributes
        self.__dict__.update(configs)

    def __repr__(self) -> str:
        return str(self.__dict__)


def get_parameters(args: Dict) -> parameters:
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
    p.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="""
    Output base path.
    Leave it blank when training on Vertex AI, environment variables will be used.
        """,
    )
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--shuffle_buffer_size", type=int, required=False, default=0)
    p.add_argument("--batch_size", type=int, required=False, default=32)
    p.add_argument("--epochs", type=int, required=False, default=10)
