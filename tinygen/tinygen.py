import argparse
import logging
from typing import Callable, Dict

from tinygen import evaluate, evaluate_pars, preprocess, train, train_pars

modules: Dict[str, Dict[str, Callable]] = {
    "preprocess": {
        "runner": preprocess.run,
        "parameter_processor": preprocess.preprocess_parameters,
        "build_parser": preprocess.build_parser,
    },
    "train": {
        "runner": train.run,
        "parameter_processor": train_pars.get_parameters,
        "build_parser": train.build_parser,
    },
    "evaluate": {
        "runner": evaluate.run,
        "parameter_processor": evaluate_pars.get_parameters,
        "build_parser": evaluate.build_parser,
    },
}


def build_subparsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tinygen")

    subparsers = parser.add_subparsers(required=True, dest="command")

    # register the subparser
    for module in modules.values():
        module["build_parser"](subparsers)

    return parser


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(funcName)s | %(message)s",  # noqa: E501
        level=logging.INFO,
    )

    parser = build_subparsers()
    args = parser.parse_args()

    logging.info(f"subcommand: {args.command}")

    module = modules[args.command]
    pars = module["parameter_processor"](vars(args))
    logging.info(pars)
    module["runner"](pars)
