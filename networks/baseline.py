import tensorflow as tf
from typing import Optional


class ZeroValueBaseline(tf.keras.Model):
    def __init__(self):
        """
        Returns fixed value of 0 as the next prediction.
        """
        super(ZeroValueBaseline, self).__init__()
        self.base_value = tf.Variable(0.0, dtype=tf.float32)

    def call(self, inputs):
        return self.base_value


class CurrentValueBaseline(tf.keras.Model):
    def __init__(self, label_index: Optional[int]=None):
        """
        Returns the current value as the next prediction.
        """
        super(CurrentValueBaseline, self).__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs

        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, num_out_steps: int):
        super(MultiStepLastBaseline, self).__init__()
        self.num_out_steps = num_out_steps

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.num_out_steps, 1])

