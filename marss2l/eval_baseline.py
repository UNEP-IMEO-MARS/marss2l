from marss2l.loaders import DatasetPlumes, CSV_PATH_DEFAULT
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional
import logging
import argparse 
import os
from marss2l.utils import setup_stream_logger, get_remote_filesystem
from marss2l.validation_utils import run_validation, THRESHOLD_PIXELS
import fsspec


config_default = {
    "batch_norm": True,
    "film_train_zero_id": True
}

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.classification_head = None
    
    def forward(self, x):
        return -x["mbmp"]


def run_eval(output_dir:str, 
             split:str="test",
             csv_path:str=CSV_PATH_DEFAULT,
             device_name:str="cuda",
             logger:Optional[logging.Logger]=None,
             log_images:bool=False,
             all_locs=None,
             num_workers:int=4,
             batch_size:int=16,
             threshold_mbmp:float=-0.95,
             suffix_output:str="",
             threshold_pixels:int=THRESHOLD_PIXELS,
             fs:Optional[fsspec.AbstractFileSystem]=None):
    
    if logger is None:
        logger = logging.getLogger(__name__)
        setup_stream_logger(logger, logging.INFO)

    torch.backends.cudnn.benchmark = True
    device = torch.device(device_name)

    if csv_path.startswith("az://"):
        assert fs is not None, "If using az:// paths, fs should be provided"
    else:
        fs = fsspec.filesystem("file")
    
    # Load options from config
    test_dataset = DatasetPlumes(mode="test",
                                split=split,
                                device=device,
                                multipass = True,
                                cloud_mask = False,
                                wind = False,
                                do_simulation=False,
                                dataframe_or_csv_path=csv_path,
                                norm_wind=True,
                                bands_l8=True,
                                logger=logger,
                                film_dict_mapping=None,
                                film_train_zero_id=False,
                                cat_mbmp=True,
                                all_locs=all_locs,
                                load_ch4=True,
                                fs=fs)

    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            shuffle=False)
    
    model = BaselineModel()
    
    os.makedirs(output_dir, exist_ok=True)
    if log_images:
        output, images = run_validation(test_loader, 
                                        model, 
                                        mode="test", 
                                        threshold_pixels=threshold_pixels,
                                        threshold=threshold_mbmp,
                                        apply_sigmoid=False,
                                        extra_keys_to_gpu=["mbmp"],
                                        log_images=log_images)
        for im in images.keys():
            np.save(os.path.join(output_dir, f"plot_{split}{suffix_output}_{im}.npy"), images[im])
    else:
        output = run_validation(test_loader, 
                                model, 
                                mode="test",
                                threshold_pixels=threshold_pixels, 
                                threshold=threshold_mbmp,
                                apply_sigmoid=False,
                                extra_keys_to_gpu=["mbmp"])
    output.to_csv(os.path.join(output_dir, f"preds_{split}{suffix_output}.csv"), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="Directory to save the experiments results. e.g. train_logs/baseline_mbmp",
                        default="train_logs/baseline_mbmp")
    parser.add_argument("--split", default="test_2023", help="Split to evaluate. e.g. test, post_2022_test, test_2023")
    parser.add_argument("--csv_path", default=CSV_PATH_DEFAULT, help="Path to the csv file with the data")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size to run the evaluation")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers to load the data")
    parser.add_argument("--threshold_pixels", default=THRESHOLD_PIXELS, type=int, 
                        help=f"Threshold to use in the connected components to scene-level prediction. Default {THRESHOLD_PIXELS}")
    parser.add_argument("--threshold_mbmp", default=-0.95, type=int, 
                        help=f"Threshold to use to set values as plume. Default {-0.95}")
    parser.add_argument("--log_images", action='store_true', default=False)
    parser.add_argument('--suffix_output', default="", help="Suffix to add to the output files")

    args_parsed = parser.parse_args()
    logger = logging.getLogger(__name__)
    torch.multiprocessing.set_start_method('spawn')

    csv_path = args_parsed.csv_path
    if csv_path.startswith("az://"):
        fsread = get_remote_filesystem()
        assert fsread.exists(csv_path), f"Path {csv_path} does not exist. Should contain the csv with the data."
    else:
        assert os.path.exists(csv_path), f"Path {csv_path} does not exist. Should contain the csv with the data."
        fsread = fsspec.filesystem("file")


    run_eval(output_dir=args_parsed.output_dir, split=args_parsed.split,
             csv_path=args_parsed.csv_path, suffix_output=args_parsed.suffix_output,
             device_name="cpu", logger=logger, 
             num_workers=args_parsed.num_workers, batch_size=args_parsed.batch_size,
             threshold_pixels=args_parsed.threshold_pixels,
             threshold_mbmp=args_parsed.threshold_mbmp,
             fs=fsread,
             log_images=args_parsed.log_images)

