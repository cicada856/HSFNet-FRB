from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras


def effective_number_class_weights(labels, num_classes, beta=0.999):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / np.mean(weights)
    return weights.astype(np.float32), counts.astype(np.int64)


@tf.keras.utils.register_keras_serializable(package="frabjous")
class SparseCategoricalFocalLoss(keras.losses.Loss):
    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        reduction=keras.losses.Reduction.AUTO,
        name="sparse_categorical_focal_loss",
    ):
        super(SparseCategoricalFocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else [float(v) for v in alpha]

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())

        row_idx = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        indices = tf.stack([row_idx, y_true], axis=1)
        p_t = tf.gather_nd(y_pred, indices)
        loss = -tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)

        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=loss.dtype)
            loss = loss * tf.gather(alpha, y_true)
        return loss

    def get_config(self):
        config = super(SparseCategoricalFocalLoss, self).get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


def build_sparse_classification_loss(loss_name, y_for_balance, num_classes, gamma=2.0, beta=0.999):
    loss_name = (loss_name or "cross_entropy").lower()
    if loss_name in ("cross_entropy", "ce", "sparse_categorical_crossentropy"):
        return "sparse_categorical_crossentropy", None
    if loss_name == "focal":
        return SparseCategoricalFocalLoss(gamma=gamma, alpha=None, name="sparse_categorical_focal_loss"), None
    if loss_name in ("class_balanced_focal", "cb_focal"):
        alpha, counts = effective_number_class_weights(y_for_balance, num_classes=num_classes, beta=beta)
        metadata = {
            "class_balance_counts": [int(v) for v in counts],
            "class_balance_alpha": [float(v) for v in alpha],
            "class_balance_beta": float(beta),
        }
        return SparseCategoricalFocalLoss(gamma=gamma, alpha=alpha, name="sparse_categorical_class_balanced_focal_loss"), metadata
    raise ValueError("Unknown loss %s. Expected cross_entropy, focal, or class_balanced_focal." % loss_name)
