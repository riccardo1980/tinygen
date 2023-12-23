from typing import Callable

import tensorflow as tf

from tinygen.train_pars import parameters


class Base(tf.keras.models.Model):
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
        backbone.add(tf.keras.layers.Dense(64, activation="relu"))
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

    def __init__(self, pars: parameters):
        super(Base, self).__init__()
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
        logits = self.head(alive_features, training=training)
        return logits
