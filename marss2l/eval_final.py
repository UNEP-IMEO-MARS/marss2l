import json
import logging
import os
from typing import Annotated, Optional

import cyclopts
import fsspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from marss2l.dataframe_image_plumes import load_dataframe_split, read_csv_images
from marss2l.loaders import CSV_PATH_DEFAULT, DatasetPlumes
from marss2l.metrics import get_pixellevel_metrics, get_scenelevel_metrics
from marss2l.models import load_model, load_weights
from marss2l.utils import fs_from_path, setup_file_logger, setup_stream_logger
from marss2l.validation_utils import THRESHOLD_PIXELS, run_validation

# def debug_collate(batch):
#     elem = batch[0]

#     # Check each key's type across all batch items
#     for key in elem.keys():
#         types = [type(d[key]) for d in batch]
#         tensor_types = [d[key].dtype if torch.is_tensor(d[key]) else None for d in batch]
#         shapes = [d[key].shape if torch.is_tensor(d[key]) else None for d in batch]

#         # Look for inconsistencies
#         if len(set(types)) > 1 or len(set(str(t) for t in tensor_types)) > 1:
#             print(f"Key '{key}' has inconsistent types: {list(zip(types, tensor_types))}")
#             # Print specific values to help diagnose
#             for i, d in enumerate(batch):
#                 print(f"  Item {i}, {key}: {d[key]} (type={type(d[key])}, "
#                       f"dtype={d[key].dtype if torch.is_tensor(d[key]) else None})")
#             return None  # Stop and don't try to collate

#     # Use default collation if we didn't find issues
#     return default_collate(batch)

config_default = {"batch_norm": True, "film_train_zero_id": True}
DEFAULT_WEIGHTS_FILE_NAME = "best_epoch"


app = cyclopts.App(
    name="eval_final",
    help="""Evaluate trained MARS-S2L models on test/validation splits.

Loads a trained model checkpoint and configuration, runs inference on specified data split, 
and computes scene-level and pixel-level metrics (precision, recall, F1, IoU, etc.).

Outputs predictions CSV with per-image results and summary metrics.

Smoke test mode (fast validation):
    python -m marss2l.eval_final --smoke-test --output-dir train_logs/MARSS2L_20250326 --device-name cpu --batch-size 4 --num-workers 2

Example usage:
    python -m marss2l.eval_final --output-dir train_logs/MARSS2L_20250326 --split test_2023
"""
)


@app.default
def run_eval(
    output_dir: Annotated[str, cyclopts.Parameter(help="Directory containing model checkpoint and config")],
    split: Annotated[str, cyclopts.Parameter(help="Data split to evaluate (e.g., 'test', 'test_2023', 'post_2022_test')")] = "test_2023",
    csv_path: Annotated[str, cyclopts.Parameter(help="Path to CSV file with image metadata")] = CSV_PATH_DEFAULT,
    device_name: Annotated[str, cyclopts.Parameter(help="Device for inference (cuda or cpu)")] = "cuda",
    logger: Annotated[Optional[logging.Logger], cyclopts.Parameter(help="Logger instance (auto-created if None)")] = None,
    num_workers: Annotated[int, cyclopts.Parameter(help="Number of dataloader workers")] = 4,
    batch_size: Annotated[int, cyclopts.Parameter(help="Batch size for inference")] = 16,
    suffix_output: Annotated[str, cyclopts.Parameter(help="Suffix to add to output CSV files")] = "",
    threshold_pixels: Annotated[int, cyclopts.Parameter(help="Min connected pixels for scene-level detection")] = THRESHOLD_PIXELS,
    weights_file_name: Annotated[str, cyclopts.Parameter(help="Checkpoint filename to load")] = DEFAULT_WEIGHTS_FILE_NAME,
    path_prepend_data: Annotated[Optional[str], cyclopts.Parameter(help="Prepend path to data files (for HuggingFace datasets)")] = None,
    smoke_test: Annotated[bool, cyclopts.Parameter(help="Run evaluation on subset of data without saving results")] = False,
    fs: Annotated[Optional[fsspec.AbstractFileSystem], cyclopts.Parameter(help="Filesystem for reading data")] = None,
):
    """
    Run model evaluation on a specified data split.
    
    Loads trained model weights and configuration from output_dir, creates a test dataset,
    runs inference, and computes comprehensive evaluation metrics including:
    - Scene-level: precision, recall, F1, average precision
    - Pixel-level: IoU, precision, recall, F1
    
    Outputs:
    - preds_{split}{suffix}.csv: Per-image predictions and metrics (unless smoke_test)
    - Console: Summary statistics table
    
    Args:
        output_dir: Directory containing best_epoch checkpoint and config_experiment.json
        split: Name of the data split to evaluate
        csv_path: Path to master CSV file with all image metadata
        device_name: PyTorch device (cuda/cpu)
        logger: Optional logger instance
        num_workers: DataLoader worker processes
        batch_size: Inference batch size
        suffix_output: Optional suffix for output files
        threshold_pixels: Minimum connected component size for scene detection
        weights_file_name: Name of checkpoint file (default: best_epoch)
        path_prepend_data: Path prefix for data files
        smoke_test: If True, evaluates subset without saving files
        fs: Optional filesystem object for remote data
    """

    if logger is None:
        if smoke_test:
            logger = setup_stream_logger(level=logging.INFO)
        else:
            logger = setup_file_logger("log", "eval_final")
    
    # Auto-append weights file name to suffix if non-default and no suffix provided
    if len(suffix_output) == 0 and weights_file_name != DEFAULT_WEIGHTS_FILE_NAME:
        suffix_output = f"_{weights_file_name}"

    torch.backends.cudnn.benchmark = True
    device = torch.device(device_name)
    weights_file = os.path.join(output_dir, weights_file_name)
    if not os.path.exists(weights_file):
        logger.error(f"Model weights not found in {output_dir}. It will not run the eval")
        return
    if fs is None:
        fs = fs_from_path(csv_path)

    # Load options from config
    config_file = os.path.join(output_dir, "config_experiment.json")
    assert os.path.exists(
        config_file
    ), f"Path {config_file} does not exist. Should contain the json with the configuration of the experiment."
    config = config_default.copy()
    with open(config_file, "r") as f:
        config.update(json.load(f))

    model_name = config["model"]
    multipass = config["multipass"]
    cloud_mask = config["cloud_mask"]
    wind = config["wind"]
    classification_head = config["classification_head"]
    norm_wind = config["norm_wind"]
    do_simulation = config["do_simulation"]
    bands_l8 = config["bands_l8"]
    cat_mbmp = config["cat_mbmp"]
    batch_norm = config["batch_norm"]

    if model_name == "film":
        film_dict_mapping = config["film_dict_mapping"]
        film_train_zero_id = (
            config["film_train_zero_id"] if "film_train_zero_id" in config else False
        )
        one_param_per_channel = config.get("one_param_per_channel", True)
        max_index_film = config.get("max_index_film", None)
        if max_index_film is None:
            logger.warning(
                "max_index_film not found in the config file. It will be calculated from the data"
            )
            max_index_film = max(film_dict_mapping.values()) + 1
    else:
        film_dict_mapping = None
        film_train_zero_id = False
        one_param_per_channel = True
        max_index_film = None

    dataframe_images = read_csv_images(csv_path, fs, path_prepend_data=path_prepend_data)
    dataframe_images_test, _, _ = load_dataframe_split(
        dataframe_or_csv_path=dataframe_images,
        split=split,
        fs=fs,
        logger=logger,
        load_plumes=False,
        smoke_test=smoke_test,
    )

    test_dataset = DatasetPlumes(
        mode="test",
        strprependlogs=split,
        device=device,
        multipass=multipass,
        cloud_mask=cloud_mask,
        wind=wind,
        do_simulation=False,
        image_dataframe=dataframe_images_test,
        norm_wind=norm_wind,
        bands_l8=bands_l8,
        logger=logger,
        film_dict_mapping=film_dict_mapping,
        film_train_zero_id=film_train_zero_id,
        analysis_mode=True,
        cat_mbmp=cat_mbmp,
        fs=fs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=default_collate,
        pin_memory=False, # Otherwise we should set device in dataset to cpu
    )

    model = load_model(
        model_name=model_name,
        in_channels=len(test_dataset.bands_out),
        classification_head=classification_head,
        batch_norm=batch_norm,
        max_index_film=max_index_film,
        one_param_per_channel=one_param_per_channel,
        finetune_film=False,
        finetune_class_head=False,
        logger=logger,
    )

    model = model.to(device)
    load_weights(model, weights_file, device=device)

    logger.info(f"Running evaluation on {split} split, file {csv_path} model weights from {weights_file}")

    output = run_validation(
            test_loader,
            model,
            threshold_pixels=threshold_pixels,
            device=device,
            mode="test",
        )
    
    if not smoke_test:
        output.to_csv(os.path.join(output_dir, f"preds_{split}{suffix_output}.csv"), index=False)

    # Log eval metrics
    outs_merge = output.drop(["location_name", "tile"], axis=1)
    outs_same_period_with_fluxrate = pd.merge(outs_merge, dataframe_images_test, 
                                              on ="id_loc_image")
    outs_same_period_with_fluxrate["isplumenum"] = outs_same_period_with_fluxrate["isplume"].astype(int)
    mets_iter = get_scenelevel_metrics(outs_same_period_with_fluxrate.scene_pred, 
                                       outs_same_period_with_fluxrate.isplumenum, 
                                       threshold=0.5,
                                       as_percentage=True)
    mets_seg = get_pixellevel_metrics(TP=outs_same_period_with_fluxrate.TP, 
                                      TN=outs_same_period_with_fluxrate.TN, 
                                      FP=outs_same_period_with_fluxrate.FP, 
                                      FN=outs_same_period_with_fluxrate.FN,
                                        as_percentage=True)
    mets_iter.update(mets_seg)
    mets_iter.update({"nsamples": outs_same_period_with_fluxrate.shape[0],
                     "nlocs": outs_same_period_with_fluxrate.location_name.nunique(),
                     "nplumes": outs_same_period_with_fluxrate.isplumenum.sum(),
                     "nnoplume": (1-outs_same_period_with_fluxrate.isplumenum).sum()})
    mets = pd.DataFrame([mets_iter])
    logger.info(f"Eval metrics:\n{mets.to_string(index=False)}")

    # if model is FiLM evaluate the site_id zero
    if film_train_zero_id and (model_name == "film") and not smoke_test:
        test_dataset.film_dict_mapping = None
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False, 
            pin_memory=False
        )
        output = run_validation(test_loader, model, threshold_pixels=threshold_pixels, mode="test")
        output.to_csv(
            os.path.join(output_dir, f"preds_{split}{suffix_output}_site_id_zero.csv"),
            index=False,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app()
