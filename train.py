import argparse
import os

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.datamodule import DFC2022DataModule
from src.trainer import DFC2022SemanticSegmentationTask


def main(conf):
    pl.seed_everything(0)
    task = DFC2022SemanticSegmentationTask(**conf.experiment.module)
    datamodule = DFC2022DataModule(**conf.datamodule)
    trainer = pl.Trainer(**conf.trainer)
    trainer.fit(model=task, datamodule=datamodule)

    with open(os.path.join(trainer.logger.log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)


if __name__ == "__main__":
    # Taken from https://github.com/pangeo-data/cog-best-practices
    _rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(_rasterio_best_practices)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to config.yaml file"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config_file)
    main(conf)
