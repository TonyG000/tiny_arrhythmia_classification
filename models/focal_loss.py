import tensorflow as tf


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        num_classes = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(y_true, num_classes)

        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        ce = -tf.math.log(p_t)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        if self.class_weight is not None:
            alpha = tf.constant(self.class_weight, dtype=tf.float32)
            alpha_t = tf.gather(alpha, y_true)
            focal_weight = alpha_t * focal_weight

        return tf.reduce_mean(focal_weight * ce)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "class_weight": self.class_weight})
        return config
