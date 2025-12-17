import argparse
import json
import logging
import os
from typing import Optional

import fsspec
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from marss2l.dataframe_image_plumes import load_dataframe_split, read_csv_images
from marss2l.loaders import CSV_PATH_DEFAULT, DatasetPlumes
from marss2l.models import load_model, load_weights
from marss2l.utils import fs_from_path, setup_stream_logger, setup_file_logger
from marss2l.validation_utils import THRESHOLD_PIXELS, run_validation
from marss2l.metrics import get_scenelevel_metrics, get_pixellevel_metrics

import pandas as pd

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


def run_eval(
    output_dir: str,
    split: str = "test",
    csv_path: str = CSV_PATH_DEFAULT,
    device_name: str = "cuda",
    logger: Optional[logging.Logger] = None,
    log_images: bool = False,
    all_locs=None,
    num_workers: int = 4,
    batch_size: int = 16,
    suffix_output: str = "",
    threshold_pixels: int = THRESHOLD_PIXELS,
    weights_file_name: str = DEFAULT_WEIGHTS_FILE_NAME,
    path_prepend_data: Optional[str] = None,
    fs: Optional[fsspec.AbstractFileSystem] = None,
):

    if logger is None:
        logger = logging.getLogger(__name__)
        setup_stream_logger(logger, logging.INFO)
    

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
        load_plumes=False
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

    if log_images:
        output, images = run_validation(
            test_loader,
            model,
            mode="test",
            threshold_pixels=threshold_pixels,
            device=device,
            log_images=log_images,
        )
        for im in images.keys():
            np.save(
                os.path.join(output_dir, f"plot_{split}{suffix_output}_{im}.npy"),
                images[im],
            )
    else:
        output = run_validation(
            test_loader,
            model,
            threshold_pixels=threshold_pixels,
            device=device,
            mode="test",
        )
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
    if film_train_zero_id and (model_name == "film"):
        test_dataset.film_dict_mapping = None
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        output = run_validation(test_loader, model, threshold_pixels=threshold_pixels, mode="test")
        output.to_csv(
            os.path.join(output_dir, f"preds_{split}{suffix_output}_site_id_zero.csv"),
            index=False,
        )



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the experiments results. e.g. train_logs/multipass_wind_sim/",
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
    parser.add_argument("--suffix_output", default="", help="Suffix to add to the output files")
    parser.add_argument(
        "--device", default="cuda", help="Device to run the model. e.g. cuda or cpu"
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
    parser.add_argument("--log_images", action="store_true", default=False)
    parser.add_argument(
        "--weights_file_name",
        default=DEFAULT_WEIGHTS_FILE_NAME,
        help="Name of the weights file to load default: %(default)s",
    )
    parser.add_argument(
        "--path_prepend_data",
        type=str,
        default=None,
        help="Path to prepend to data paths (s2path, plumepath, cloudmaskpath, ch4path). Required for dataset downloaded from Hugging Face.",
    )

    args_parsed = parser.parse_args()
    logger = setup_file_logger("log","eval_final")

    suffix_output = args_parsed.suffix_output
    # Append the weights file name to the suffix if it is different from the default
    if len(suffix_output) == 0 and args_parsed.weights_file_name != DEFAULT_WEIGHTS_FILE_NAME:
        suffix_output = f"_{args_parsed.weights_file_name}"

    run_eval(
        output_dir=args_parsed.output_dir,
        split=args_parsed.split,
        csv_path=args_parsed.csv_path,
        suffix_output=suffix_output,
        device_name=args_parsed.device,
        logger=logger,
        num_workers=args_parsed.num_workers,
        batch_size=args_parsed.batch_size,
        threshold_pixels=args_parsed.threshold_pixels,
        weights_file_name=args_parsed.weights_file_name,
        log_images=args_parsed.log_images,
        path_prepend_data=args_parsed.path_prepend_data,
    )
