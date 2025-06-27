from marss2l.loaders import DatasetPlumes, CSV_PATH_DEFAULT
from marss2l.models import load_model, load_weights
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional
import logging
import argparse 
import os
from marss2l.utils import setup_stream_logger, get_remote_filesystem
import json
from marss2l.validation_utils import run_validation, THRESHOLD_PIXELS
import fsspec
from torch.utils.data.dataloader import default_collate


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

config_default = {
    "batch_norm": True,
    "film_train_zero_id": True
}
DEFAULT_WEIGHTS_FILE_NAME = "best_epoch"

def run_eval(output_dir:str, 
             split:str="test",
             csv_path:str=CSV_PATH_DEFAULT,
             device_name:str="cuda",
             logger:Optional[logging.Logger]=None,
             log_images:bool=False,
             all_locs=None,
             num_workers:int=4,
             batch_size:int=16,
             suffix_output:str="",
             threshold_pixels:int=THRESHOLD_PIXELS,
             weights_file_name:str=DEFAULT_WEIGHTS_FILE_NAME,
             fs:Optional[fsspec.AbstractFileSystem]=None):
    
    if logger is None:
        logger = logging.getLogger(__name__)
        setup_stream_logger(logger, logging.INFO)

    torch.backends.cudnn.benchmark = True
    device = torch.device(device_name)
    weights_file = os.path.join(output_dir, weights_file_name)
    if not os.path.exists(weights_file):
        logger.error(f"Model weights not found in {output_dir}. It will not run the eval")
        return

    if csv_path.startswith("az://"):
        assert fs is not None, "If using az:// paths, fs should be provided"
    else:
        fs = fsspec.filesystem("file")
    
    # Load options from config
    config_file = os.path.join(output_dir, "config_experiment.json")
    assert os.path.exists(config_file), f"Path {config_file} does not exist. Should contain the json with the configuration of the experiment."
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
        film_train_zero_id = config["film_train_zero_id"] if "film_train_zero_id" in config else False
        one_param_per_channel = config.get("one_param_per_channel", True)
        max_index_film = config.get("max_index_film", None)
        if max_index_film is None:
            logger.warning("max_index_film not found in the config file. It will be calculated from the data")
            max_index_film = max(film_dict_mapping.values()) + 1
    else:
        film_dict_mapping = None
        film_train_zero_id = False
        one_param_per_channel = True
        max_index_film = None

    test_dataset = DatasetPlumes(mode="test",
                                split=split,
                                device=device,
                                multipass = multipass,
                                cloud_mask = cloud_mask,
                                wind = wind,
                                do_simulation=False,
                                dataframe_or_csv_path=csv_path,
                                norm_wind=norm_wind,
                                bands_l8=bands_l8,
                                logger=logger,
                                film_dict_mapping=film_dict_mapping,
                                film_train_zero_id=film_train_zero_id,
                                cat_mbmp=cat_mbmp,
                                all_locs=all_locs,
                                load_ch4=log_images,
                                fs=fs)

    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            shuffle=False,
                            collate_fn=default_collate)
    
    model = load_model(model_name=model_name, in_channels=len(test_dataset.bands_out),
                       classification_head=classification_head,
                       batch_norm=batch_norm,
                       max_index_film=max_index_film,
                       one_param_per_channel=one_param_per_channel,
                       finetune_film=False,
                       finetune_class_head=False,
                       logger=logger)

    model = model.to(device)
    load_weights(model, weights_file, device=device)
    
    if log_images:
        output, images = run_validation(test_loader, 
                                        model,  
                                        mode="test", 
                                        threshold_pixels=threshold_pixels,
                                        device=device,
                                        log_images=log_images)
        for im in images.keys():
            np.save(os.path.join(output_dir, f"plot_{split}{suffix_output}_{im}.npy"), images[im])
    else:
        output = run_validation(test_loader, model, threshold_pixels=threshold_pixels, 
                                device=device,
                                mode="test")
    output.to_csv(os.path.join(output_dir, f"preds_{split}{suffix_output}.csv"), index=False)
    

    # if model is FiLM evaluate the site_id zero
    if film_train_zero_id and (model_name == "film"):
        test_dataset.film_dict_mapping = None
        test_loader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 num_workers=num_workers,
                                 shuffle=False)
        output = run_validation(test_loader, model, 
                                threshold_pixels=threshold_pixels,
                                mode="test")
        output.to_csv(os.path.join(output_dir, f"preds_{split}{suffix_output}_site_id_zero.csv"), index=False)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Directory to save the experiments results. e.g. train_logs/multipass_wind_sim/")
    parser.add_argument("--split", default="test_2023", help="Split to evaluate. e.g. test, post_2022_test, test_2023")
    parser.add_argument("--csv_path", default=CSV_PATH_DEFAULT, help="Path to the csv file with the data")
    parser.add_argument('--suffix_output', default="", help="Suffix to add to the output files")
    parser.add_argument("--device", default="cuda", help="Device to run the model. e.g. cuda or cpu")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size to run the evaluation")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers to load the data")
    parser.add_argument("--threshold_pixels", default=THRESHOLD_PIXELS, type=int, 
                        help=f"Threshold to use in the connected components to scene-level prediction. Default {THRESHOLD_PIXELS}")
    parser.add_argument("--log_images", action='store_true', default=False)
    parser.add_argument('--weights_file_name', default=DEFAULT_WEIGHTS_FILE_NAME, 
                        help="Name of the weights file to load default: %(default)s")

    args_parsed = parser.parse_args()
    logger = logging.getLogger(__name__)

    csv_path = args_parsed.csv_path
    if csv_path.startswith("az://"):
        fsread = get_remote_filesystem()
        assert fsread.exists(csv_path), f"Path {csv_path} does not exist. Should contain the csv with the data."
    else:
        assert os.path.exists(csv_path), f"Path {csv_path} does not exist. Should contain the csv with the data."
        fsread = fsspec.filesystem("file")

    suffix_output=args_parsed.suffix_output
    # Append the weights file name to the suffix if it is different from the default
    if len(suffix_output) == 0  and args_parsed.weights_file_name != DEFAULT_WEIGHTS_FILE_NAME:
        suffix_output = f"_{args_parsed.weights_file_name}"

    run_eval(output_dir=args_parsed.output_dir, split=args_parsed.split,
             csv_path=args_parsed.csv_path, suffix_output=suffix_output,
             device_name=args_parsed.device, logger=logger, 
             num_workers=args_parsed.num_workers, batch_size=args_parsed.batch_size,
             threshold_pixels=args_parsed.threshold_pixels,
             fs=fsread,
             weights_file_name=args_parsed.weights_file_name,
             log_images=args_parsed.log_images)

