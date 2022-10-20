import tensorflow as tf
from typing import Dict


class LossObject:
    """
    Wraps a tensorflow loss function for hydra composability.
    """
    def __init__(self, loss_fn: str):
        self.loss_fn = loss_fn
        self.registry: Dict = {
            "huber": tf.keras.losses.Huber(),
            "mse": tf.keras.losses.MeanSquaredError(),
        }

    def build_loss(self) -> tf.keras.losses.Loss:
        return self.registry[self.loss_fn]
