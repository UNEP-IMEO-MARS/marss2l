import argparse
import logging
from typing import Optional

import fsspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from marss2l.dataframe_image_plumes import load_dataframe_split, read_csv_images
from marss2l.loaders import CSV_PATH_DEFAULT, DatasetPlumes
from marss2l.utils import fs_for_path, fs_from_path, pathjoin, setup_stream_logger
from marss2l.validation_utils import THRESHOLD_PIXELS, run_validation

config_default = {"batch_norm": True, "film_train_zero_id": True}


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.classification_head = None

    def forward(self, x):
        return -x["mbmp"]


def run_eval(
    output_dir: str,
    split: str = "test",
    csv_path: str = CSV_PATH_DEFAULT,
    device_name: str = "cuda",
    logger: Optional[logging.Logger] = None,
    all_locs=None,
    num_workers: int = 4,
    batch_size: int = 16,
    threshold_mbmp: float = -0.95,
    suffix_output: str = "",
    threshold_pixels: int = THRESHOLD_PIXELS,
    path_prepend_data: Optional[str] = None,
    fs: Optional[fsspec.AbstractFileSystem] = None,
):

    if logger is None:
        logger = logging.getLogger(__name__)
        setup_stream_logger(logger, logging.INFO)

    torch.backends.cudnn.benchmark = True
    device = torch.device(device_name)
    if fs is None:
        fs = fs_from_path(csv_path)
    # Filesystem for the output dir (preds), independent of the images fs.
    fsout = fs_for_path(output_dir, fs)

    # Load and split dataframe similar to eval_final
    dataframe_images = read_csv_images(csv_path, fs, path_prepend_data=path_prepend_data)
    dataframe_images_test, _, _ = load_dataframe_split(
        dataframe_or_csv_path=dataframe_images,
        split=split,
        fs=fs,
        load_plumes=False,
        logger=logger,
    )

    # Load options from config
    test_dataset = DatasetPlumes(
        mode="test",
        strprependlogs=split,
        device=device,
        multipass=True,
        cloud_mask=False,
        wind=False,
        do_simulation=False,
        image_dataframe=dataframe_images_test,
        norm_wind=True,
        bands_l8=True,
        logger=logger,
        film_dict_mapping=None,
        film_train_zero_id=False,
        cat_mbmp=True,
        analysis_mode=True,
        fs=fs,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    model = BaselineModel()

    fsout.makedirs(output_dir, exist_ok=True)
    output = run_validation(
            test_loader,
            model,
            mode="test",
            threshold_pixels=threshold_pixels,
            threshold=threshold_mbmp,
            apply_sigmoid=False,
            extra_keys_to_gpu=["mbmp"],
        )
    with fsout.open(pathjoin(output_dir, f"preds_{split}{suffix_output}.csv"), "w") as f:
        output.to_csv(f, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help="Directory to save the experiments results. e.g. train_logs/baseline_mbmp",
        default="train_logs/baseline_mbmp",
    )
    parser.add_argument(
        "--split",
        default="test_2023",
        help="Split to evaluate. e.g. test, post_2022_test, test_2023",
    )
    parser.add_argument(
        "--csv_path",
        default=CSV_PATH_DEFAULT,
        help="Path to the csv file with the data",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size to run the evaluation"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of workers to load the data"
    )
    parser.add_argument(
        "--threshold_pixels",
        default=THRESHOLD_PIXELS,
        type=int,
        help=f"Threshold to use in the connected components to scene-level prediction. Default {THRESHOLD_PIXELS}",
    )
    parser.add_argument(
        "--threshold_mbmp",
        default=-0.95,
        type=int,
        help=f"Threshold to use to set values as plume. Default {-0.95}",
    )

    parser.add_argument("--suffix_output", default="", help="Suffix to add to the output files")
    parser.add_argument(
        "--path_prepend_data",
        type=str,
        default=None,
        help="Path to prepend to data paths (s2path, plumepath, cloudmaskpath, ch4path). Required for dataset downloaded from Hugging Face.",
    )

    args_parsed = parser.parse_args()
    logger = logging.getLogger(__name__)
    torch.multiprocessing.set_start_method("spawn")

    csv_path = args_parsed.csv_path

    run_eval(
        output_dir=args_parsed.output_dir,
        split=args_parsed.split,
        csv_path=args_parsed.csv_path,
        suffix_output=args_parsed.suffix_output,
        device_name="cpu",
        logger=logger,
        num_workers=args_parsed.num_workers,
        batch_size=args_parsed.batch_size,
        threshold_pixels=args_parsed.threshold_pixels,
        threshold_mbmp=args_parsed.threshold_mbmp,
        path_prepend_data=args_parsed.path_prepend_data
    )
