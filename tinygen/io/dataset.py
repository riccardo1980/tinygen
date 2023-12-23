import os

import tensorflow as tf

from tinygen.io.tfrecords import read_tfrecord
from tinygen.train_pars import parameters


def get_dataset(
    path: str, pars: parameters, one_hot_labels: bool = True
) -> tf.data.Dataset:
    """
    Read TFRecords and return a dataset.

    :param pars:
    :param one_hot_labels:
    :return: dataset
    """

    dataset = tf.data.TFRecordDataset(
        tf.data.Dataset.list_files(os.path.join(path, "*.tfrecords"))
    )
    dataset = dataset.map(
        read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if one_hot_labels:
        dataset = dataset.map(
            lambda text, label: (text, tf.one_hot(label, pars.num_classes)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    if pars.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(pars.shuffle_buffer_size)

    dataset = dataset.batch(pars.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
