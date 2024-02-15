from typing import Dict, List

import tensorflow as tf


def classification_metrics_build(num_classes: int) -> List[tf.keras.metrics.Metric]:
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    for id in range(num_classes):
        metrics.append(PerClassPrecision(class_id=id, name=f"precision_{id}_p"))
        metrics.append(
            PerClassRecall(class_id=id, name=f"recall_{id}_p"),
        )
    return metrics


class PerClassPrecision(tf.keras.metrics.Metric):
    def __init__(
        self, class_id: int, name: str = "per_class_precision", **kwargs: Dict
    ):
        super(PerClassPrecision, self).__init__(name=name, **kwargs)

        self.class_id = class_id

        self.true_positives = self.add_weight(
            name="tp", initializer="zeros", dtype=tf.int32
        )
        self.predicted = self.add_weight(
            name="predicted", initializer="zeros", dtype=tf.int32
        )

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None
    ) -> None:
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        predicted_values = tf.equal(y_pred, self.class_id)
        predicted = tf.cast(tf.math.count_nonzero(predicted_values), dtype=tf.int32)
        self.predicted.assign_add(predicted)

        true_positives_values = tf.logical_and(
            tf.equal(y_true, self.class_id), tf.equal(y_pred, self.class_id)
        )
        true_positives = tf.cast(
            tf.math.count_nonzero(true_positives_values), dtype=tf.int32
        )
        self.true_positives.assign_add(true_positives)

    def result(self) -> tf.Tensor:
        if self.predicted == tf.constant(0):
            return tf.constant(0.0, dtype=tf.float32)
        else:
            return tf.cast(self.true_positives / self.predicted, dtype=tf.float32)

    def reset_state(self) -> None:
        self.true_positives.assign(0)
        self.predicted.assign(0)


class PerClassRecall(tf.keras.metrics.Metric):
    def __init__(self, class_id: int, name: str = "per_class_recall", **kwargs: Dict):
        super(PerClassRecall, self).__init__(name=name, **kwargs)

        self.class_id = class_id

        self.true_positives = self.add_weight(
            name="tp", initializer="zeros", dtype=tf.int32
        )
        self.support = self.add_weight(
            name="support", initializer="zeros", dtype=tf.int32
        )

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None
    ) -> None:
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        support_values = tf.equal(y_true, self.class_id)
        support = tf.cast(tf.math.count_nonzero(support_values), dtype=tf.int32)
        self.support.assign_add(support)

        true_positives_values = tf.logical_and(
            tf.equal(y_true, self.class_id), tf.equal(y_pred, self.class_id)
        )
        true_positives = tf.cast(
            tf.math.count_nonzero(true_positives_values), dtype=tf.int32
        )
        self.true_positives.assign_add(true_positives)

    def result(self) -> tf.Tensor:
        if self.support == tf.constant(0, dtype=tf.int32):
            return tf.constant(0.0, dtype=tf.float32)
        else:
            return tf.cast(self.true_positives / self.support, dtype=tf.float32)

    def reset_state(self) -> None:
        self.true_positives.assign(0)
        self.support.assign(0)
