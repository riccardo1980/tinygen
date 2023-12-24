from typing import Any, Callable, Dict, List, Optional, Union  # noqa: F401

import tensorflow as tf

from tinygen.metrics import PerClassPrecision, PerClassRecall
from tinygen.train_pars import parameters


class BaseClassifier(tf.keras.models.Model):
    embedding_dim: int
    num_classes: int
    dropout_rate: float

    vectorizer: tf.keras.layers.TextVectorization
    embedder: tf.keras.Model
    backbone: tf.keras.Model
    regularizer: tf.keras.Model
    head: tf.keras.Model

    @staticmethod
    def __build_embedder(input_dim: int, output_dim: int) -> tf.keras.Model:
        embedder = tf.keras.Sequential(name=f"embedder_{input_dim}_{output_dim}")
        embedder.add(
            tf.keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                mask_zero=True,
            )
        )
        return embedder

    @staticmethod
    def __build_backbone() -> tf.keras.Model:
        backbone = tf.keras.Sequential(name="backbone")
        backbone.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
        # backbone.add(tf.keras.layers.Dense(5, activation="relu"))
        return backbone

    @staticmethod
    def __build_head(num_classes: int) -> tf.keras.Model:
        head = tf.keras.Sequential(name="head")
        head.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        return head

    @staticmethod
    def __build_regularizer(dropout_rate: float) -> tf.keras.Model:
        regularizer = tf.keras.Sequential(name="regularizer")
        regularizer.add(tf.keras.layers.Dropout(rate=dropout_rate))
        return regularizer

    @staticmethod
    def __build_metrics(num_classes: int) -> List[tf.keras.metrics.Metric]:
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        for id in range(num_classes):
            metrics.append(PerClassPrecision(class_id=id, name=f"precision_{id}_p"))
            metrics.append(
                PerClassRecall(class_id=id, name=f"recall_{id}_p"),
            )
        return metrics

    def __init__(self, pars: parameters):
        super(BaseClassifier, self).__init__()
        self.embedding_dim = pars.embedding_dim
        self.num_classes = pars.num_classes
        self.dropout_rate = pars.dropout

        self.vectorizer = tf.keras.layers.TextVectorization(
            output_mode="int", standardize=None
        )
        # remaining layers are built on adapt

    def adapt(
        self,
        data: tf.data.Dataset,
        extractor: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = lambda X, y: X,
        batch_size: int = 32,
    ) -> None:
        self.vectorizer.adapt(data.unbatch().map(extractor).batch(batch_size))
        self.embedder = self.__build_embedder(
            len(self.vectorizer.get_vocabulary()), self.embedding_dim
        )
        self.backbone = self.__build_backbone()
        self.regularizer = self.__build_regularizer(self.dropout_rate)
        self.head = self.__build_head(self.num_classes)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs = self.vectorizer(inputs)
        embeddings = self.embedder(inputs, training=training)
        features = self.backbone(embeddings, training=training)
        alive_features = self.regularizer(features, training=training)
        scores = self.head(alive_features, training=training)
        return scores

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
