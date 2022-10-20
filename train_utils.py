import hydra
import pandas as pd
import tensorflow as tf
import wandb
from omegaconf import DictConfig, OmegaConf

from etl import dataset_partitioners, transforms
from utils import helpers
from utils.callbacks import build_callbacks


def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.logging.log_wandb:
        run = wandb.init(project="ldz", config=OmegaConf.to_container(cfg), job_type="train")

    # Load data
    full_df = pd.read_csv(cfg.data.train_data_filepath)

    if cfg.data.features.summarise:
        colnames = helpers.column_names(cfg.data.summary_features.gridpoints, cfg.data.summary_features.colnames)
        colnames_day = list(filter(lambda col: "night" not in col, colnames))
        colnames_night = list(filter(lambda col: "night" in col, colnames))

        full_df = transforms.summarise(df=full_df, colnames=colnames_day, name=colnames_day[0][:-2])
        full_df = transforms.summarise(df=full_df, colnames=colnames_night, name=colnames_night[0][:-2])

        if cfg.data.summary_features.lag_summaries:
            ...
        if cfg.data.summary_features.ema_summaries:
            ...

    if cfg.data.features.lags:
        colnames = helpers.column_names(cfg.data.lag_features.gridpoints, cfg.data.lag_features.colnames)
        full_df = transforms.lag_features(df=full_df, colnames=colnames, time_periods=cfg.data.lag_features.time_periods)

    if cfg.data.features.ema:
        colnames = helpers.column_names(cfg.data.ema_features.gridpoints, cfg.data.ema_features.colnames)
        full_df = transforms.ema(df=full_df, colnames=colnames, time_periods=cfg.data.ema_features.time_periods)

    full_df.dropna(inplace=True)

    # Select raw input features to keep
    features_and_labels = cfg.data.labels_columns + cfg.data.features_columns
    full_df = full_df[features_and_labels]

    full_df.to_csv(cfg.data.full_postprocessed_data_filepath, index=False)

    # Split training data into train-validation sets
    data_splitter = dataset_partitioners.TimeSeriesDataSplitter(full_df)
    train_df, validation_df = data_splitter(cfg.train.validation_ratio)

    # Build TF datasets
    dataset_generator = hydra.utils.instantiate(
        cfg.dataset_generator,
        batch_size=cfg.train.batch_size,
        train_df=train_df,
        validation_df=validation_df,
        test_df=None,
        full_df=full_df,
    )
    train_dataset = dataset_generator.train_dataset
    validation_dataset = dataset_generator.validation_dataset
    full_dataset = dataset_generator.full_dataset

    loss_object = hydra.utils.instantiate(cfg.loss)
    model = hydra.utils.instantiate(cfg.model)
    model.compile(
        loss=loss_object.build_loss(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.optimisation.learning_rate),
    )

    callbacks = build_callbacks(
        config=cfg.logging,
        model_name=model.name,
        config_optim=cfg.optimisation,
    )

    if cfg.use_full_data:
        train_dataset = full_dataset

    history = model.fit(
        train_dataset,
        epochs=cfg.train.epochs,
        validation_data=validation_dataset,
        callbacks=callbacks
    )

    print("HISTORY: \n", history)

    if cfg.logging.log_wandb:
        wandb.finish()
