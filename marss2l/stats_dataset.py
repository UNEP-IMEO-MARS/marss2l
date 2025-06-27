description ="""
This script loads the dataset with all images and compute the stats per band.    
"""

import os
import logging
from marss2l.utils import setup_file_logger, get_remote_filesystem, pathjoin
import argparse
from marss2l import loaders
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from marss2l.mars_sentinel2 import quantification
import numpy as np
import pandas as pd
from typing import Optional


def run(csv_path:str, batch_size:int=128, num_workers:int=4, 
        output_file:Optional[str]=None, max_iter:Optional[int]=None):
    logger = logging.getLogger(__name__)
    setup_file_logger("logs", "stats_dataset", logger)
    fs = get_remote_filesystem()
    if output_file is None:
        output_file = pathjoin(os.path.dirname(args_parsed.csv_path), "stats_dataset.csv")
    
    dataframe_data_traintest = loaders.read_csv(csv_path, 
                                                add_columns_for_analysis=False, 
                                                fs=fs)
    dataset = loaders.DatasetPlumes(mode="test",
                        split="no split",
                        device="cpu",
                        dataframe_or_csv_path=dataframe_data_traintest,
                        multipass = True,
                        cloud_mask = True,
                        wind = False,
                        do_simulation=False,
                        norm_wind=False,
                        bands_l8=True,
                        logger=logger,
                        film_dict_mapping=None,
                        film_train_zero_id=None,
                        cat_mbmp=True,
                        all_locs=None,
                        load_ch4=True,
                        fs=fs)
    
    test_loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        shuffle=False)
    
    stats = []
    with torch.no_grad():
        for task in tqdm(test_loader, desc="Eval model"):
            xbatch = task["y_context_ls0_0"]
            targetbatch = task["y_target"].squeeze(1)
            ch4batch = task["ch4"].squeeze(1)
            isplumebatch = task["isplume"].cpu().numpy()
            for batchidx in range(len(xbatch)):
                x = xbatch[batchidx]
                target = targetbatch[batchidx]
                ch4 = ch4batch[batchidx]
                location_name = task["location_name"][batchidx]
                tile = task["tile"][batchidx]
                id_loc_image = str(task["id_loc_image"][batchidx])
                wind_vector = task["wind"][batchidx].cpu().numpy()
                input_data = {"location_name": location_name,
                              "tile": tile,
                              "id_loc_image": id_loc_image,
                              "isplume": isplumebatch[batchidx],
                              "wind_u": float(wind_vector[0]),
                              "wind_v": float(wind_vector[1])
                }
                # bands out e.g. ['MBMP', 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B02_bg', 'B03_bg', 'B04_bg', 'B08_bg', 'B11_bg', 'B12_bg', 'U', 'V', 'cloudmask']
                stats_out = compute_stats(dataset.bands_out, 
                                          isplume=input_data["isplume"], 
                                          ch4=ch4,
                                          target=target,
                                          x=x,
                                          wind_vector=wind_vector)
                input_data.update(stats_out)
                stats.append(input_data)                
            
            if max_iter is not None and len(stats) >= max_iter:
                break
    
    stats_df = pd.DataFrame(stats)
    with fs.open(output_file, "w") as f:
        stats_df.to_csv(f, index=False)

def compute_stats(bands_out: list, isplume: int, ch4: torch.Tensor, target: torch.Tensor, x: torch.Tensor, wind_vector: np.ndarray) -> dict:
    """
    Compute various statistics for given input data.
    
    Args:
        bands_out : list
            List of band names to compute statistics for.
        isplume : int
            Indicator whether the data contains a plume (1 if true, 0 otherwise).
        ch4 : torch.Tensor
            Tensor containing CH4 (methane) concentration data.
        target : torch.Tensor
            Tensor containing target labels indicating plume presence (1 for plume, 0 for no plume).
        x : torch.Tensor
            Tensor containing the data. Bands in this tensor are assumed to be in the same order as in `bands_out`.
        wind_vector : np.ndarray
            Numpy array representing the wind vector.
    
            
    Returns:
        dict
            Dictionary containing computed statistics. The keys include:
            - "ch4_mean": Mean of CH4 concentrations.
            - "ch4_std": Standard deviation of CH4 concentrations.
            - "ch4_min": Minimum of CH4 concentrations.
            - "ch4_max": Maximum of CH4 concentrations.
            - "ch4_mean_plume": Mean of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_std_plume": Standard deviation of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_min_plume": Minimum of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_max_plume": Maximum of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_mean_noplume": Mean of CH4 concentrations outside the plume (if isplume is 1).
            - "ch4_std_noplume": Standard deviation of CH4 concentrations outside the plume (if isplume is 1).
            - "ch4_min_noplume": Minimum of CH4 concentrations outside the plume (if isplume is 1).
            - "ch4_max_noplume": Maximum of CH4 concentrations outside the plume (if isplume is 1).
            - Additional keys for flux rate quantification if isplume is 1.
            - For each band in bands_out (excluding "_bg" band, "U", "V"):
                - "{band}_mean": Mean of the band data.
                - "{band}_std": Standard deviation of the band data.
                - "{band}_min": Minimum of the band data.
                - "{band}_max": Maximum of the band data.
                - "{band}_mean_plume": Mean of the band data within the plume (if isplume is 1).
                - "{band}_std_plume": Standard deviation of the band data within the plume (if isplume is 1).
                - "{band}_min_plume": Minimum of the band data within the plume (if isplume is 1).
                - "{band}_max_plume": Maximum of the band data within the plume (if isplume is 1).
                - "{band}_mean_noplume": Mean of the band data outside the plume (if isplume is 1).
                - "{band}_std_noplume": Standard deviation of the band data outside the plume (if isplume is 1).
                - "{band}_min_noplume": Minimum of the band data outside the plume (if isplume is 1).
                - "{band}_max_noplume": Maximum of the band data outside the plume (if isplume is 1).
            - For "cloudmask" band:
                - "{cloudmask_value}": Count of each unique value in the cloudmask band.
    
    Notes:
    ------
    - The function assumes that the input tensors are properly aligned and have compatible shapes.
    - The function uses the `quantification` module to obtain flux rate statistics if `isplume` is 1.
    """
    stats_item = {}

    # Stats target (number of pixels 1)
    stats_item["npixelsplume"] = target.sum().item()
    stats_item["npixels"] = target.numel()

    # mean, std, min, and max for CH4
    stats_item["ch4_mean"] = ch4.mean().item()
    stats_item["ch4_std"] = ch4.std().item()
    stats_item["ch4_min"] = ch4.min().item()
    stats_item["ch4_max"] = ch4.max().item()
    if isplume == 1:
        stats_item["ch4_mean_plume"] = ch4[target == 1].mean().item()
        stats_item["ch4_std_plume"] = ch4[target == 1].std().item()
        stats_item["ch4_min_plume"] = ch4[target == 1].min().item()
        stats_item["ch4_max_plume"] = ch4[target == 1].max().item()
        stats_item["ch4_mean_noplume"] = ch4[target == 0].mean().item()
        stats_item["ch4_std_noplume"] = ch4[target == 0].std().item()
        stats_item["ch4_min_noplume"] = ch4[target == 0].min().item()
        stats_item["ch4_max_noplume"] = ch4[target == 0].max().item()
        wind_speed = np.linalg.norm(wind_vector)
        stats_item.update(quantification.obtain_flux_rate(ch4.numpy(), target.numpy(), wind_speed=wind_speed, 
                                                a_u_eff=quantification.A_UEFF_S2,
                                                b_u_eff=quantification.B_UEFF_S2, 
                                                sig_xch4=quantification.SIGMA_CH4_S2_PPB,
                                                resolution=(10,10),
                                                return_std=True))
    
    # TODO average difference RGBNIR bands between image and background

    # TODO re-calculate MBMP masking plume?
                
    for bidx, b in enumerate(bands_out):
        if "_bg" in b:
            continue
        if b in {"U", "V"}:
            continue
        if b == "cloudmask":
            # Count unique values
            unique, counts = torch.unique(x[bidx], return_counts=True)
            stats_item.update({f"{b}_{u.item()}": c.item() for u, c in zip(unique, counts)})
        else:
            # Compute mean, std, min, and max
            xband = x[bidx] / 2 # Divide by to to obtain the value in ToA units
            stats_item[f"{b}_mean"] = xband.mean().item()
            stats_item[f"{b}_std"] = xband.std().item()
            stats_item[f"{b}_min"] = xband.min().item()
            stats_item[f"{b}_max"] = xband.max().item()

            if isplume == 1:
                # Compute mean, std, min, and max inside and outside of the plume
                stats_item[f"{b}_mean_plume"] = xband[target == 1].mean().item()
                stats_item[f"{b}_std_plume"] = xband[target == 1].std().item()
                stats_item[f"{b}_min_plume"] = xband[target == 1].min().item()
                stats_item[f"{b}_max_plume"] = xband[target == 1].max().item()
                stats_item[f"{b}_mean_noplume"] = xband[target == 0].mean().item()
                stats_item[f"{b}_std_noplume"] = xband[target == 0].std().item()
                stats_item[f"{b}_min_noplume"] = xband[target == 0].min().item()
                stats_item[f"{b}_max_noplume"] = xband[target == 0].max().item()

    return stats_item

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--csv_path", type=str, help="Path to the csv with the data",
                        default= loaders.CSV_PATH_DEFAULT)
    parser.add_argument("--batch_size", type=int, help="Batch size to train the model. Default: %(default)s",
                        default=128)
    parser.add_argument("--num_workers", type=int, help="Number of workers to use in the dataloader. Default: %(default)s",
                        default=4)
    parser.add_argument("--output_file", type=str, help="Path to the output file",
                        required=False)
    parser.add_argument("--max_iter", type=int, help="Maximum number of iterations to run",
                        required=False)
    args_parsed = parser.parse_args()
    run(args_parsed.csv_path, batch_size=args_parsed.batch_size, 
        num_workers=args_parsed.num_workers, 
        output_file=args_parsed.output_file,
        max_iter=args_parsed.max_iter)
