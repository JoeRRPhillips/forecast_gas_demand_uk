import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from etl import transforms
from utils.load import load_model
from utils.prediction_writers import write_predictions


@hydra.main(config_path="configs", config_name="config_predict")
def predict(cfg: DictConfig):
    if cfg.predict.test:
        _predict(cfg, cfg.predict_test)

    if cfg.predict.full:
        _predict(cfg, cfg.predict_full)


def _predict(cfg: DictConfig, sub_cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model = load_model(cfg)

    if cfg.predict.method == "batch":
        predictions = _predict_batch(cfg, model)
    elif cfg.predict.method == "autoregressive":
        predictions = _predict_autoregressive(cfg, model)
    else:
        raise ValueError("Unknown prediction method")

    write_predictions(predictions, cfg, sub_cfg, model)


def _predict_batch(cfg: DictConfig, model: tf.keras.Model) -> np.ndarray:
    train_df = pd.read_csv(cfg.data.train_data_filepath)
    test_df = pd.read_csv(cfg.data.test_data_filepath)

    # Make sure column input order is consistent
    features_train_df = train_df.drop(cfg.data.labels_columns[0], axis=1)
    test_df = test_df.reindex(features_train_df.columns, axis=1)
    assert all([c1 == c2 for c1, c2 in zip(features_train_df.columns, test_df.columns)])

    if cfg.data.features.lags:
        # colnames = itertools.product(cfg.data.ar.colnames, cfg.data.ar.gridpoints)
        # colnames = list(map(lambda x: x[0] + str(x[1]), colnames))
        colnames = []
        for gp in cfg.data.lag_features.gridpoints:
            for col in cfg.data.lag_features.colnames:
                colnames.append(col + str(gp))

        lags = [0]
        if cfg.data.features.lags:
            lags.extend(cfg.data.lag_features.time_periods)

        if cfg.data.features.ema:
            lags.extend(cfg.data.ema_features.time_periods)

        # Number of rows to borrow from training set
        max_lag = max(lags)

        # end_train_df = train_df.iloc[:max_lag, :].drop("demand", axis=1)
        end_train_df = features_train_df.iloc[:max_lag, :]
        full_df = pd.concat([end_train_df, test_df])
        full_df = transforms.lag_features(df=full_df, colnames=colnames, time_periods=cfg.data.lag_features.time_periods)
        full_df = transforms.ema(df=full_df, colnames=colnames, time_periods=cfg.data.ema_features.time_periods)
        full_df.dropna(inplace=True)

        # Select raw input features to keep
        full_df = full_df[cfg.data.features_columns]

        # Trim off the borrowed rows
        test_df = full_df.tail(len(test_df))

    # Check pred and train order correct --> TODO: move data processing together
    train_final_df = pd.read_csv(cfg.data.full_postprocessed_data_filepath)
    train_final_df.pop("demand")
    assert all([c1 == c2 for c1, c2 in zip(train_final_df.columns, test_df.columns)])

    test_data = tf.convert_to_tensor(np.array(test_df))

    predictions = []
    batch_size = cfg.predict.batch_size
    num_data = test_data.shape[0]
    for i in range(0, num_data, batch_size):
        j = min(i+batch_size, num_data)
        test_input = test_data[i:j]
        predictions_batch = model.predict(test_input)
        predictions.append(predictions_batch)

    predictions = np.squeeze(np.vstack(predictions), axis=-1)

    return predictions


def _predict_autoregressive(cfg: DictConfig, model: tf.keras.Model) -> np.ndarray:
    test_df = pd.read_csv(cfg.data.test_data_filepath)

    for col in cfg.data.drop_columns:
        if col in test_df.columns:
            test_df.drop(col, axis=1, inplace=True)

    assert test_df.columns[0] == cfg.data.labels_columns[0]

    test_data = np.array(test_df)

    predictions = []
    # batch_size = 1  # Autoregressive: force 1 in case of lagged demands
    num_data = test_data.shape[0] - 1
    for i in range(0, num_data):
        # test_input = tf.reshape(tf.convert_to_tensor(test_data[i, :]), shape=[1, 1, -1])
        test_input = tf.reshape(tf.convert_to_tensor(test_data[i, :]), shape=[1 , -1])
        pred = model.predict(test_input)
        predictions.append(pred)

        # Autoregressive: bootstrap predictions. Expects 0th column to contain lagged demand placeholders.
        test_data[0][i+1] = pred

    predictions = np.squeeze(np.vstack(predictions), axis=-1)

    return predictions


if __name__ == "__main__":
    predict()
