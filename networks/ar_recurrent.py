import abc
import tensorflow as tf


class AutoRegressiveMultiStepRNN(abc.ABC, tf.keras.Model):
    def __init__(
            self,
            recurrent_units: int,
            num_features: int,
            num_out_steps: int,
            name: str,
    ):
        super(AutoRegressiveMultiStepRNN, self).__init__(name=name)
        self.recurrent_units = recurrent_units
        self.num_out_steps = num_out_steps
        self.dense = tf.keras.layers.Dense(num_features)

        # TODO(JP): rm?
        self.dense_out = tf.keras.layers.Dense(1)


    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []

        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)

        # print("========")
        # print("warmup p shape: ", prediction.shape)
        for n in range(1, self.num_out_steps):
            # Use the last prediction as input.
            x = prediction

            # Execute one rnn step.
            x, state = self.rnn_cell(x, states=state, training=training)
            # print("rnn x shape: ", x.shape)

            # Convert the rnn output to a prediction.
            prediction = self.dense(x)
            # print("dense x shape: ", prediction.shape)

            # Append prediction to the output.
            predictions.append(prediction)

        # predictions.shape -> (time, batch, features)
        predictions = tf.stack(predictions)

        # predictions.shape -> (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        # print("predictions feat shape: ", predictions.shape)

        # TODO(JP): rm?
        # predictions.shape -> (batch, time, labels)
        predictions = self.dense_out(predictions)
        # print("predictions out shape: ", predictions.shape)

        return predictions

    def warmup(self, inputs):
        # inputs.shape -> [batch, time, features]
        # x.shape -> [batch, recurrent_units]
        x, *state = self.rnn(inputs)

        # prediction.shape -> [batch, num_features]
        prediction = self.dense(x)
        return prediction, state


class AutoRegressiveMultiStepLSTM(AutoRegressiveMultiStepRNN):
    def __init__(self, recurrent_units, name="AutoRegressiveMultiStepLSTM", *args, **kwargs):
        super(AutoRegressiveMultiStepLSTM, self).__init__(recurrent_units=recurrent_units, name=name, *args, **kwargs)
        self.rnn_cell = tf.keras.layers.LSTMCell(recurrent_units)

        # Wrap `rnn_cell` for convenience during `warmup` method
        self.rnn = tf.keras.layers.RNN(cell=self.rnn_cell, return_state=True)


class AutoRegressiveMultiStepGRU(AutoRegressiveMultiStepRNN):
    def __init__(self, recurrent_units, name="AutoRegressiveMultiStepGRU", *args, **kwargs):
        super(AutoRegressiveMultiStepGRU, self).__init__(recurrent_units=recurrent_units, name=name, *args, **kwargs)
        self.rnn_cell = tf.keras.layers.GRUCell(recurrent_units)

        # Wrap `rnn_cell` for convenience during `warmup` method
        self.rnn = tf.keras.layers.RNN(cell=self.rnn_cell, return_state=True)
