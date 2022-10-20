import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def write_predictions(
        predictions: np.ndarray,
        cfg: DictConfig,
        sub_cfg: DictConfig,
        model: tf.keras.Model,
):
    train_datetime = "/".join(cfg.predict.saved_model_path.split("/")[7:9])
    output_dir = Path(f"{cfg.predict.output_dir}/{train_datetime}/{model.name}")
    if not output_dir.exists():
        Path.mkdir(output_dir, parents=True)

    output_filename = sub_cfg.output_filename
    predict_csv_fp = Path(f"{output_dir}/{output_filename}")
    shutil.copyfile(src=Path(sub_cfg.template_output_file), dst=predict_csv_fp)

    predict_df = pd.read_csv(predict_csv_fp)
    label_colum = cfg.data.labels_columns[0]
    predict_df[label_colum] = predictions
    predict_df.to_csv(predict_csv_fp, index=False)

    with open(f"{output_dir}/config.yaml", "w") as fp:
        OmegaConf.save(config=cfg, f=fp.name)
