import tensorflow as tf
from tensorflow import Tensor
from typing import List


class FlatLinearModel(tf.keras.Model):
    def __init__(
            self,
            num_outputs: int,
            name: str = "LinearModel",
            **kwargs,
    ):
        super(FlatLinearModel, self).__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(units=num_outputs,  name="dense_output", activation="linear")

    def call(self, inputs: Tensor) -> Tensor:
        x = self.flatten(inputs)
        x = self.dense_layer(x)
        return x


class LinearModel(tf.keras.Model):
    def __init__(
            self,
            num_outputs: int,
            name: str = "LinearModel",
            **kwargs,
    ):
        """
        Learns a linear transformation from inputs to outputs.
        Tranforms the final axis: [batch, time, inputs] -> [batch, time, num_outputs].
        Applied independently to each item across batch & time dimensions.
        """
        super(LinearModel, self).__init__(name=name, **kwargs)
        self.dense_layer = tf.keras.layers.Dense(units=num_outputs,  name="dense_output", activation="linear")

    def call(self, inputs: Tensor) -> Tensor:
        return self.dense_layer(inputs)


class MLP(tf.keras.Model):
    def __init__(
            self,
            dense_units: List[int],
            num_outputs: int,
            activation_fn_hidden: str,
            activation_fn_output: str,
            dropout_rate: float,
            use_batchnorm: bool,
            name: str = "MLP",
            **kwargs,
    ):
        super(MLP, self).__init__(name=name, **kwargs)
        self.dense_units = dense_units
        self.dropout_rates = [dropout_rate] * len(dense_units)

        dense_layers = []
        for k, (units, rate) in enumerate(zip(self.dense_units, self.dropout_rates)):
            dense_layers.append(
                tf.keras.layers.Dense(
                    units,
                    name=f"dense_{k}",
                    activation=activation_fn_hidden,
                )
            )
            dense_layers.append(tf.keras.layers.Dropout(rate))

            if use_batchnorm:
                dense_layers.append(tf.keras.layers.BatchNormalization())

        dense_layers.append(
            tf.keras.layers.Dense(num_outputs, name="dense_output", activation=activation_fn_output)
        )
        self.dense_layers = dense_layers

    def call(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x


class FlatMLP(MLP):
    def __init__(self, name: str = "FlatMLP", **kwargs,):
        self.flatten = tf.keras.layers.Flatten()
        super(FlatMLP, self).__init__(name=name, **kwargs)

    def call(self, inputs: Tensor) -> Tensor:
        x = self.flatten(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        return x

class ConcatMLP(MLP):
    def __init__(self, name: str = "ConcatMLP", **kwargs):
        self.concatenate = tf.keras.layers.Concatenate()
        super(ConcatMLP, self).__init__(name=name, **kwargs)

    def call(self, inputs: List[Tensor]):
        x = self.concatenate(**inputs)
        for layer in self.dense_layers:
            x = layer(x)
        return x
