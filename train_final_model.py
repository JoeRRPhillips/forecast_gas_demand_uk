import hydra
from omegaconf import DictConfig

from train_utils import train_model


@hydra.main(config_path="configs", config_name="config_train_final")
def train(cfg: DictConfig):
    train_model(cfg)


if __name__ == "__main__":
    train()
