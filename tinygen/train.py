import argparse
import logging

import tensorflow as tf

from tinygen.io.dataset import get_dataset
from tinygen.models.base_model import Base
from tinygen.train_pars import parameters


def run(pars: parameters) -> None:
    """
    Run preprocess

    :param pars: parameters
    :type pars: Parameters
    """

    # read train dataset
    train_dataset = get_dataset(
        pars.train_dataset_path, pars=pars, one_hot_labels=True
    )  # noqa: F841

    # read eval dataset
    if pars.eval_dataset_path:
        eval_dataset = get_dataset(
            pars.eval_dataset_path, pars=pars, one_hot_labels=True
        )  # noqa: F841
    else:
        eval_dataset = None  # noqa: F841

    # build model
    model = Base(pars)
    model.adapt(train_dataset)
    logging.debug(model.vectorizer.get_vocabulary())
    logging.info(f"vocabulary size: {len(model.vectorizer.get_vocabulary())}")

    metrics = [tf.keras.metrics.CategoricalAccuracy()]

    for id in range(pars.num_classes):
        metrics.append(
            tf.keras.metrics.Precision(name=f"precision_{id}_p", class_id=id),
        )
        metrics.append(
            tf.keras.metrics.Recall(name=f"recall_{id}_p", class_id=id),
        )

    # compile
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=pars.learning_rate),
        metrics=metrics,
    )

    # callbacks
    # FIXME: add callbacks

    # fit
    model.fit(
        train_dataset, validation_data=eval_dataset, verbose=2, epochs=pars.epochs
    )

    model.summary()

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
    p.add_argument("--learning_rate", type=float, required=False, default=0.001)
    p.add_argument("--dropout", type=float, required=False, default=0.2)
    p.add_argument("--embedding_dim", type=int, required=False, default=128)
