import tensorflow as tf
import wandb
from omegaconf import DictConfig
from pathlib import Path
from typing import List


def build_callbacks(
        config: DictConfig,
        model_name: str,
        config_optim: DictConfig,
) -> List[tf.keras.callbacks.Callback]:
    checkpoint_dir = Path(f"{config.checkpoint_dir}/{model_name}")
    if not checkpoint_dir.exists():
        Path.mkdir(checkpoint_dir, parents=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{model_name}", histogram_freq=1)
    callbacks = [tensorboard_callback]

    if config.save_model:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            save_weights_only=False,
            verbose=1,
            monitor=config.monitor,
            save_best_only=config.save_best_only,
            save_freq=config.save_freq,
        )
        callbacks.append(checkpoint_callback)

    if config.log_wandb:
        wandb_callback = wandb.keras.WandbCallback(
            monitor=config.monitor,
            log_weights=False,
            log_evaluation=True,
            save_model=False,
        )
        callbacks.append(wandb_callback)

    if config_optim.reduce_lr_on_plateau_factor > 0.0:
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config_optim.reduce_lr_on_plateau_monitor,
            verbose=1,
            factor=config_optim.reduce_lr_on_plateau_factor,
            patience=config_optim.reduce_lr_on_plateau_patience,
            min_lr=config_optim.reduce_lr_on_plateau_min_learning_rate,
        )
        callbacks.append(lr_callback)

    return callbacks
