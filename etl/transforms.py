import pandas as pd
from typing import List


def summarise(df: pd.DataFrame, colnames: List[str], name: str) -> pd.DataFrame:
    df[f"{name}_max"] = df[colnames].max(axis=1)
    df[f"{name}_mean"] = df[colnames].mean(axis=1)
    df[f"{name}_min"] = df[colnames].min(axis=1)
    df[f"{name}_std"] = df[colnames].std(axis=1)
    return df


def lag_features(df: pd.DataFrame, colnames: List[str], time_periods: List[int]) -> pd.DataFrame:
    for colname in colnames:
        for lag in time_periods:
            df[f"{colname}_lag{lag}"] = df[colname].shift(lag)

    # df.dropna(inplace=True)

    return df


def ema(df: pd.DataFrame, colnames: List[str], time_periods: List[int]) -> pd.DataFrame:
    for colname in colnames:
        for span in time_periods:
            df[f"{colname}_ema{span}"] = df[colname].ewm(span=span).mean()

    # df.dropna(inplace=True)

    return df

