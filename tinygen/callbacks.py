import os
from typing import List

import tensorflow as tf

from tinygen.train_pars import Parameters


def callbacks_build(configs: Parameters) -> List[tf.keras.callbacks.Callback]:
    """
    Builds basic callbacks

    Callbacks:
        - model checkpoint
    """
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(configs.checkpoints_path, "saved_model.pb"),
        save_weights_only=False,
        save_best_only=False,
        save_freq="epoch",
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(configs.logs_path),
        update_freq="epoch",
        histogram_freq=0,
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, verbose=1
    )

    reduce_lr_on_plateau = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
    )

    callbacks = [
        model_checkpoint_callback,
        tensorboard_callback,
        early_stopping_callback,
        reduce_lr_on_plateau,
    ]

    return callbacks
