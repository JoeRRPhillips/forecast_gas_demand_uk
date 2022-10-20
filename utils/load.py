import hydra
import tensorflow as tf
from omegaconf import DictConfig


def load_model(cfg: DictConfig) -> tf.keras.Model:
    loss_object = hydra.utils.instantiate(cfg.loss)
    model = hydra.utils.instantiate(cfg.model)
    model.compile(
        loss=loss_object.build_loss(),
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.optimisation.learning_rate),
    )
    model.load_weights(cfg.predict.saved_model_path)
    return model
