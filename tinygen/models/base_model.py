from typing import Any, Callable, Dict, List, Optional, Union  # noqa: F401

import tensorflow as tf

from tinygen.metrics import PerClassPrecision, PerClassRecall


def base_classifier_build(
    input_dim: int,
    embedding_dim: int,
    num_classes: int,
    dropout: float,
    name: str = "base_model",
) -> tf.keras.Model:
    model = tf.keras.Sequential(
        name=name,
        layers=[
            tf.keras.layers.Input(shape=(None,)),
            tf.keras.layers.Embedding(
                input_dim=input_dim, output_dim=embedding_dim, mask_zero=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(rate=dropout),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
    )
    return model


class BaseClassifier(tf.keras.Model):
    embedding_dim: int
    num_classes: int
    dropout_rate: float

    vectorizer: tf.keras.layers.TextVectorization
    embedder: tf.keras.Model
    backbone: tf.keras.Model
    regularizer: tf.keras.Model
    head: tf.keras.Model

    # @staticmethod
    def __build_metrics(self, num_classes: int) -> List[tf.keras.metrics.Metric]:
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        for id in range(num_classes):
            metrics.append(PerClassPrecision(class_id=id, name=f"precision_{id}_p"))
            metrics.append(
                PerClassRecall(class_id=id, name=f"recall_{id}_p"),
            )
        return metrics

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_classes: int,
        dropout: float,
        name: str = "base_model",
        **kwargs: Any,
    ):
        super(BaseClassifier, self).__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.input_dim, output_dim=self.embedding_dim, mask_zero=True
        )
        self.backbone = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.head = tf.keras.layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embedding(inputs)
        x = self.backbone(x)
        x = self.dropout(x, training=training)
        x = self.head(x)

        return x

    def compile(
        self,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = "rmsprop",
        loss: Union[str, tf.keras.losses.Loss] = None,
        metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        loss_weights: Optional[Union[List, Dict]] = None,
        weighted_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        run_eagerly: Optional[bool] = None,
        steps_per_execution: Optional[int] = None,
        jit_compile: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        if loss is None:
            loss = "categorical_crossentropy"

        if metrics is None:
            metrics = self.__build_metrics(self.num_classes)

        super(BaseClassifier, self).compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs,
        )
