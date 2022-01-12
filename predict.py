import argparse
import glob
import os

import kornia.augmentation as K
import rasterio
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datamodule import DFC2022DataModule
from src.trainer import DFC2022SemanticSegmentationTask


def write_mask(mask, path, output_dir):
    with rasterio.open(path) as src:
        profile = src.profile
    profile["count"] = 1
    profile["dtype"] = "uint8"
    region = os.path.dirname(path).split(os.sep)[-2]
    filename = os.path.basename(os.path.splitext(path)[0])
    output_path = os.path.join(output_dir, region, f"{filename}_prediction.tif")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)


@torch.no_grad()
def main(log_dir, output_directory, device):
    os.makedirs(output_directory, exist_ok=True)

    # Load checkpoint and config
    conf = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    ckpt = glob.glob(os.path.join(log_dir, "checkpoints", "*.ckpt"))[0]

    # Load model
    task = DFC2022SemanticSegmentationTask.load_from_checkpoint(ckpt)
    task = task.to(device)
    task.eval()

    # Load datamodule and dataloader
    datamodule = DFC2022DataModule(**conf.datamodule)
    datamodule.setup()
    dataloader = datamodule.predict_dataloader()

    pad = K.PadTo(size=(2048, 2048), pad_mode="constant", pad_value=0.0)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = batch["image"].to(device)
        h, w = x.shape[-2:]
        x = pad(x)
        mask = task(x)
        mask = mask[0, :, :h, :w]
        mask = mask.argmax(dim=0).cpu().numpy()
        filename = datamodule.predict_dataset.files[i]["image"]
        write_mask(mask, filename, output_directory)


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
        "--log_dir",
        type=str,
        required=True,
        help="Path to log directory containing config.yaml and checkpoint",
    )
    parser.add_argument(
        "--predict_on",
        type=str,
        default="val",
        choices=["val", "train-unlabeled"],
        help="Dataset to generate predictions of",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Path to output_directory to save predicted mask geotiffs",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args.log_dir, args.output_directory, args.device)
