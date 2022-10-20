import abc
import tensorflow as tf
from typing import List


class RecurrentBaseNetwork(abc.ABC, tf.keras.Model):
    def __init__(
            self,
            rnn_units: List[int],
            dense_units: List[int],
            num_outputs: int,
            dropout_rate_rnn: float,
            dropout_rate_dense: float,
            use_batchnorm: bool,
            name: str
    ):
        super(RecurrentBaseNetwork, self).__init__(name=name)
        self.rnn_units = rnn_units

        # Use rnn dropout sparingly.
        self.dropout_rate_rnn = [dropout_rate_rnn] * len(dense_units)
        self.dropout_rate_dense = [dropout_rate_dense] * len(dense_units)

        dense_layers = []
        for k, (units, rate_dense) in enumerate(zip(dense_units, self.dropout_rate_dense)):
            dense_layers.append(
                tf.keras.layers.Dense(
                    units,
                    name=f"dense_{k}",
                    activation="relu",
                )
            )
            if rate_dense > 0.0:
                dense_layers.append(tf.keras.layers.Dropout(rate_dense))
            if use_batchnorm:
                dense_layers.append(tf.keras.layers.BatchNormalization())

        # [batch, time, last_rnn_units or last_hidden] -> [batch, time, labels]
        dense_layers.append(tf.keras.layers.Dense(units=num_outputs))
        self.dense_layers = dense_layers

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return x


class LSTMNetwork(RecurrentBaseNetwork):
    def __init__(
            self,
            rnn_units: List[int],
            dense_units: List[int],
            num_outputs: int,
            dropout_rate_rnn: float,
            dropout_rate_dense: float,
            use_batchnorm: bool,
            name: str="LSTMNetwork"
    ):
        super(LSTMNetwork, self).__init__(
            rnn_units=rnn_units,
            dense_units=dense_units,
            num_outputs=num_outputs,
            dropout_rate_rnn=dropout_rate_rnn,
            dropout_rate_dense=dropout_rate_dense,
            use_batchnorm=use_batchnorm,
            name=name
        )

        # [batch, time, features] -> [batch, time, rnn_units]
        # self.rnn_layers = [
        #     tf.keras.layers.LSTM(units, return_sequences=True, name=f"lstm_{k}") for k, units in enumerate(rnn_units)
        # ]
        self.rnn_layers = []
        for k, (units, rate_rnn) in enumerate(zip(rnn_units, self.dropout_rate_rnn)):
            self.rnn_layers.append(tf.keras.layers.LSTM(units, return_sequences=True, name=f"lstm_{k}"))
            if rate_rnn > 0.0:
                self.rnn_layers.append(tf.keras.layers.Dropout(rate_rnn))


class GRUNetwork(RecurrentBaseNetwork):
    def __init__(
            self,
            rnn_units: List[int],
            dense_units: List[int],
            num_outputs: int,
            dropout_rate_rnn: float,
            dropout_rate_dense: float,
            use_batchnorm: bool,
            name: str="GRUNetwork"
    ):
        super(GRUNetwork, self).__init__(
            rnn_units=rnn_units,
            dense_units=dense_units,
            num_outputs=num_outputs,
            dropout_rate_rnn=dropout_rate_rnn,
            dropout_rate_dense=dropout_rate_dense,
            use_batchnorm=use_batchnorm,
            name=name
        )

        # [batch, time, features] -> [batch, time, rnn_units]
        # self.rnn_layers = [
        #     tf.keras.layers.GRU(units, return_sequences=True, name=f"gru_{k}") for k, units in enumerate(rnn_units)
        # ]
        self.rnn_layers = []
        for k, (units, rate_rnn) in enumerate(zip(rnn_units, self.dropout_rate_rnn)):
            self.rnn_layers.append(tf.keras.layers.GRU(units, return_sequences=True, name=f"lstm_{k}"))
            if rate_rnn > 0.0:
                self.rnn_layers.append(tf.keras.layers.Dropout(rate_rnn))
