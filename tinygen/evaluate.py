import argparse
import logging
from typing import Dict

import tensorflow as tf

from tinygen.evaluate_pars import Parameters
from tinygen.io.dataset import get_dataset
from tinygen.metrics import classification_metrics_build

def run(pars: Parameters) -> None:
    """
    Run evaluation

    :param pars: parameters
    :type pars: Parameters
    """
    logging.info("Running evaluation")
    dataset = get_dataset(
        pars.dataset_path,
        num_classes=pars.num_classes,
        shuffle_buffer_size=pars.shuffle_buffer_size,
        batch_size=pars.batch_size,
    )

    # load a saved model
    model = tf.keras.models.load_model(
        pars.model_path,
        compile=False,
    )
    model.compile(
        loss="categorical_crossentropy",
        metrics=classification_metrics_build(pars.num_classes),
    )

    # can't call evaluate!
    stats = model.evaluate(dataset, verbose=2)
    out_dict = {k: v for k, v in zip(model.metrics_names, stats)}
    logging.info(out_dict)


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Build argument parser

    :param subparsers: subparsers
    :type subparsers: argparse._SubParsersAction
    """
    p = subparsers.add_parser("evaluate", help="Evaluate")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=False, default=32)
    p.add_argument("--shuffle_buffer_size", type=int, required=False, default=1000)

    p.set_defaults(func=run)


def get_parameters(args: Dict) -> Parameters:
    pars = Parameters(args)
    return pars
