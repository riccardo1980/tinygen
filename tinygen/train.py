import argparse
import logging
import os
from typing import Tuple

import tensorflow as tf

from tinygen.callbacks import callbacks_build
from tinygen.io.dataset import get_dataset
from tinygen.metrics import classification_metrics_build
from tinygen.models.base_model import base_classifier_build
from tinygen.train_pars import TrainParameters


def run(pars: TrainParameters) -> None:
    """
    Run preprocess

    :param pars: parameters
    :type pars: Parameters
    """
    # read train dataset
    train_dataset = get_dataset(
        pars.train_dataset_path,
        num_classes=pars.num_classes,
        shuffle_buffer_size=pars.shuffle_buffer_size,
        batch_size=pars.batch_size,
        one_hot_labels=True,
    )

    # read eval dataset
    if pars.eval_dataset_path:
        eval_dataset = get_dataset(
            pars.eval_dataset_path,
            num_classes=pars.num_classes,
            shuffle_buffer_size=pars.shuffle_buffer_size,
            batch_size=pars.batch_size,
            one_hot_labels=True,
        )
    else:
        eval_dataset = None

    # see: https://keras.io/examples/nlp/text_classification_from_scratch

    # vectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        output_mode="int", standardize=None
    )
    vectorize_layer.adapt(train_dataset.unbatch().map(lambda text, lbl: text))
    logging.info(f"vocabulary size: {len(vectorize_layer.get_vocabulary())}")

    # vectorize the datasets
    def vectorize_text(
        text: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_dataset = train_dataset.map(vectorize_text)
    train_dataset = train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(vectorize_text)
        eval_dataset = eval_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

    # # build LSTM classification model
    model = base_classifier_build(
        input_dim=len(vectorize_layer.get_vocabulary()),
        embedding_dim=pars.embedding_dim,
        num_classes=pars.num_classes,
        dropout=pars.dropout,
    )

    # # compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=pars.learning_rate),
        loss="categorical_crossentropy",
        metrics=classification_metrics_build(pars.num_classes),
    )

    # fit
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        verbose=2,
        epochs=pars.epochs,
        callbacks=callbacks_build(pars),
    )

    model.summary()

    # build end-to-end model
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vectorize_layer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)

    end_to_end_model = tf.keras.Model(inputs, outputs)

    # save model
    end_to_end_model.save(pars.model_path, save_format="tf")


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
