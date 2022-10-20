import abc
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Tuple


class DatasetGenerator(abc.ABC):
    def __init__(
            self,
            train_df: pd.DataFrame,
            validation_df: pd.DataFrame,
            test_df: pd.DataFrame,
            full_df: pd.DataFrame,
            batch_size: int,
            *args,
            **kwargs
    ):
        self.batch_size = batch_size
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df
        self.full_df = full_df

    def make_dataset(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def train_dataset(self) -> tf.data.Dataset:
        return self.make_dataset(self.train_df)

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        return self.make_dataset(self.validation_df)

    @property
    def test_dataset(self) -> tf.data.Dataset:
        return self.make_dataset(self.test_df)

    @property
    def full_dataset(self) -> tf.data.Dataset:
        return self.make_dataset(self.full_df)


class LinearDatasetGenerator(DatasetGenerator):
    def __init__(
            self,
            train_df: pd.DataFrame,
            validation_df: pd.DataFrame,
            test_df: pd.DataFrame,
            full_df: pd.DataFrame,
            batch_size: int,
            shuffle_data: bool,
            shuffle_buffer_size: int,
            label_columns: List[str]
    ):
        super(LinearDatasetGenerator, self).__init__(
            train_df=train_df, validation_df=validation_df, test_df=test_df, full_df=full_df, batch_size=batch_size
        )
        self.shuffle_data = shuffle_data
        self.shuffle_buffer_size = shuffle_buffer_size

        # LDG currently only tested to work with a single label column
        assert len(label_columns) == 1
        self.label_columns = label_columns[0]

    def make_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        features = df.copy()
        labels = features.pop(self.label_columns)

        features = np.array(features)
        labels = np.array(labels)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        if self.shuffle_data:
            dataset = dataset.shuffle(self.shuffle_buffer_size)

        return dataset.batch(self.batch_size)

    @property
    def test_dataset(self) -> tf.data.Dataset:
        # features = pd.read_csv(self.test_data_filepath)
        # dataset = tf.data.Dataset.from_tensor_slices(np.array(features))
        # return dataset.batch(self.batch_size)
        dataset = tf.data.Dataset.from_tensor_slices(np.array(self.test_df))
        return dataset.batch(self.batch_size)


class WindowedDatasetGenerator(DatasetGenerator):
    def __init__(
            self,
            train_df: pd.DataFrame,
            validation_df: pd.DataFrame,
            test_df: pd.DataFrame,
            full_df: pd.DataFrame,
            batch_size: int,
            input_width: int,
            label_width: int,
            shift: int,
            label_columns: Optional[List[str]]=None,
    ):
        super(WindowedDatasetGenerator, self).__init__(
            train_df=train_df, validation_df=validation_df, test_df=test_df, full_df=full_df, batch_size=batch_size
        )
        """
        Example:
            `input_width` = 24 and `shift`=`label_width`=1 --> make a 1 hour prediction based on last 24.
        """
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(self.train_df.columns)}

        # Compute window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift

        # Time-based indexing.
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def make_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        """
        :param df:
        :return: (features, labels): ([batch_size, input_width (time), num_features],
                                        [batch_size, label_width (time), num_labels])
        """
        data = np.array(df, dtype=np.float32)
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )
        return dataset.map(self._split_window)

    def _split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

            # TODO(JP): rm?
            # inputs = tf.stack([inputs[:, :, 1:]], axis=-1)
            inputs = inputs[:, :, 1:]

        # Slicing does not preserve static shape information -> set the
        # shapes manually to make `tf.data.Dataset`s easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
