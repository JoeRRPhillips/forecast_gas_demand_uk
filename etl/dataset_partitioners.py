import abc
import math
import pandas as pd
from typing import Optional, Tuple


class DataSplitter(abc.ABC):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TimeSeriesDataSplitter(DataSplitter):
    def __init__(self, df: pd.DataFrame):
        """
        Splits a timeseries dataframe by reserving the final rows for validation.
        """
        super(TimeSeriesDataSplitter, self).__init__(df=df)

    def __call__(self, validation_ratio: float) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        :param validation_ratio: [0.0,1.0] amount of training data to reserve for validation
        :return: train df, validation df
        """
        assert validation_ratio >= 0.0
        if validation_ratio == 0.0:
            return self.df, None

        num_train = math.floor(len(self.df) * (1.0 - validation_ratio))
        train_df = self.df.iloc[:num_train, :]
        validation_df = self.df.iloc[num_train:, :]

        return train_df, validation_df
