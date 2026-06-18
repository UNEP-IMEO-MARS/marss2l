import json
import os
import uuid
from typing import List, Optional, Union

import loguru
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fsspec import AbstractFileSystem
from loguru._logger import Logger
from torch.utils.data import DataLoader

from marss2l.models import SegmentationModelMARSS2L
from marss2l.utils import pathjoin, fs_from_path
from marss2l.trainer import THRESHOLD_PIXELS, Trainer

bce_loss = nn.BCEWithLogitsLoss(reduction="none")



def run_validation(
    test_loader: DataLoader,
    model: SegmentationModelMARSS2L,
    mode: str = "test",
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    threshold: float = 0.5,
    apply_sigmoid: bool = True,
    extra_keys_to_gpu: Optional[List[str]] = None,
    threshold_pixels: int = THRESHOLD_PIXELS,
):
    """
    Run validation on a DataLoader and return predictions and metrics.
    
    This is a convenience wrapper that creates a Trainer object and calls its run_validation method.
    
    Args:
        test_loader (DataLoader): DataLoader for test/validation data
        model (SegmentationModelMARSS2L): Model to evaluate
        mode (str): "test" or "val" mode
        device (torch.device): Device to use for computation
        threshold (float): Threshold for binary predictions
        apply_sigmoid (bool): Whether to apply sigmoid to model output
        extra_keys_to_gpu (Optional[List[str]]): Extra keys to move to GPU
        threshold_pixels (int): Minimum number of pixels for scene prediction
    
    Returns:
        pd.DataFrame or tuple: DataFrame with predictions, optionally with images dict
    """
    # Create a minimal Trainer object to use its run_validation method
    trainer = Trainer(
        model=model,
        save_path=None,  # No need to save anything for validation only
        device=device,
    )
    
    output, loss = trainer.run_validation(
        test_loader=test_loader,
        mode=mode,
        threshold=threshold,
        apply_sigmoid=apply_sigmoid,
        extra_keys_to_gpu=extra_keys_to_gpu,
        threshold_pixels=threshold_pixels,
    )
    return output


def load_stats_and_config(
    train_folder: str,
    model_name: str,
    basefolder_experiments: str,
    fs: Optional[AbstractFileSystem] = None,
    logger: Optional[Logger] = None,
    csv_file: str = "preds_test_2023",
) -> tuple[pd.DataFrame, Optional[dict]]:
    """
    Load the evaluation results and configuration from a CSV file.
    If also loads the configuration file if it exists.

    It assumes that the CSV has been generated before using `marss2l.eval_final` script.

    It searches for the CSV file in:
    - `{basefolder_experiments}/{train_folder}/{csv_file}.csv`

    It adds a column `model_name` to the DataFrame with the value of `model_name`.

        Args:
            train_folder (str): Name of the training folder,
                which is used to find the model folder.
            model_name (str): Name of the model, which is used to identify the results.
            basefolder_experiments (str): _basefolder_experiment_ is the base folder where the experiments are stored.
            fs (AbstractFileSystem): File system to use for reading the files.
            logger (Optional[Logger], optional): Logger to use for logging.
                Defaults to None, in which case a logger is created.
            csv_file (str, optional): Name of the CSV file to load.
                Defaults to "preds_test_2023".

        Returns:
            tuple[pd.DataFrame, Optional[dict]]:
    """

    if fs is None:
        fs = fs_from_path(basefolder_experiments)

    if logger is None:
        logger = loguru.logger

    csv_file = os.path.splitext(csv_file)[0]
    model_folder = pathjoin(basefolder_experiments, train_folder)
    path = pathjoin(model_folder, f"{csv_file}.csv")
    path_config = pathjoin(model_folder, "config_experiment.json")
    logger.info(f"Loading eval results from {path}")
    with fs.open(path, "r") as fh:
        output = pd.read_csv(fh)
    if fs.exists(path_config):
        logger.info(f"Loading config from {path_config}")
        with fs.open(path_config, "r") as fh:
            config = json.load(fh)
    else:
        config = None

    if output["scene_pred"].isna().any():
        print("Dropping:", output.loc[output["scene_pred"].isna()])
        output = output.loc[~output["scene_pred"].isna()].copy()

    output["model_name"] = model_name
    output["id_loc_image"] = output["id_loc_image"].apply(uuid.UUID)

    return output, config
