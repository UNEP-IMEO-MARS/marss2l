
import torch 
import numpy as np 

from tqdm import tqdm
import torch.nn as nn
from typing import Optional, List, Union
from torch.utils.data import DataLoader
from marss2l.mars_sentinel2.plume_detection import threshold_cutoff_connected_components
import pandas as pd
import os
# from torchmetrics.functional import confusion_matrix
from torchmetrics.functional.classification import binary_confusion_matrix
from marss2l.mars_sentinel2 import quantification
from marss2l.models import SegmentationModelMARSS2L

from marss2l.utils import pathjoin
from fsspec import AbstractFileSystem
import json
import logging
import uuid



bce_loss = nn.BCEWithLogitsLoss(reduction="none")
THRESHOLD_PIXELS = 100

def threshold_cutoff(pred_continuous:torch.Tensor, 
                     threshold_pixels:float=THRESHOLD_PIXELS,
                     tol:float=1e-3) -> float:
    """
    Implements binary search to find the continuous value that produces more than `threshold_pixels` pixels connected 
    in the scene.

    Args:
        pred_continuous (torch.Tensor): (H, W) or Tensor with float values (not necessarily between 0 and 1)
        threshold_pixels (float, optional): Minimum number of pixels in the scene. Defaults to THRESHOLD_PIXELS.
        tol (float, optional): Tolerance for the binary search. Defaults to 1e-3.

    Returns:
        scene_prob (float): minimum value such that sum(connected_components(pred_continuous >= scene_prob)) >= threshold_pixels
    """
    pred_continuous_values = pred_continuous
    
    min_value = torch.min(pred_continuous_values).item()
    max_value = torch.max(pred_continuous_values).item()
    
    # binary search
    threshold = (min_value + max_value) / 2
    while (max_value - min_value) > tol:
        npixels = torch.sum(pred_continuous_values >= threshold).item()
        if npixels >= threshold_pixels:
            min_value = threshold
        else:
            max_value = threshold
        threshold = (min_value + max_value) / 2
    
    return float(threshold)


def run_validation(test_loader:DataLoader, model:SegmentationModelMARSS2L, 
                   loss_function:Optional[nn.Module]=None,
                   mode:str="test", 
                   device:Optional[torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   log_images:bool=False,
                   threshold:float=0.5,
                   apply_sigmoid:bool=True,
                   weight_by_ch4:bool=False,
                   extra_keys_to_gpu:Optional[List[str]]=None,
                   threshold_pixels:int=THRESHOLD_PIXELS):
    
    # Generate predictions
    model.eval()
    items = []
    sig = nn.Sigmoid()
    lf = []

    if log_images:
        ch4 = []
        mbmp = []
        target = []
        preds = []

    if hasattr(model, "module"):
        model_instance = model.module
    else:
        model_instance = model
    classification_head = model_instance.classification_head is not None
    
    keys_to_gpu = ["site_ids", "y_context_ls0_0"]
    if weight_by_ch4 and "ch4" not in keys_to_gpu:
        keys_to_gpu.append("ch4")
    if extra_keys_to_gpu is not None and len(extra_keys_to_gpu) > 0:
        keys_to_gpu += extra_keys_to_gpu

    # Log images for plotting
    with torch.no_grad():
        for task in tqdm(test_loader, desc="Eval model"):
            batch = {key: task[key].to(device) for key in keys_to_gpu}

            y_target = task["y_target"].to(device)

            out = model(batch)
            if loss_function is not None:
                kwargs_loss = {"target": y_target, "pred": out}
                if weight_by_ch4:
                    kwargs_loss["ch4"] = batch["ch4"]
                l = loss_function(**kwargs_loss).item()
                if l < 0:
                    print(f"Negative loss {task['id_loc_image']}, {l} {task['location_name']}")
                lf.append(l)

            if classification_head:
                out, scene_predbatch = out
                if apply_sigmoid:
                    scene_predbatch = sig(scene_predbatch)
                scene_predbatch = scene_predbatch.squeeze(1)
                
            if apply_sigmoid:
                out = sig(out) 
            
            out = out.squeeze(1) # (B, 1, H, W) -> (B, H, W)
            
            target_binary = task["isplume"].cpu().numpy()

            if log_images:
                ch4.append(task["ch4"].detach().cpu().numpy())
                mbmp.append(task["mbmp"].detach().cpu().numpy())
                target.append(task["y_target"].detach().cpu().numpy())
                preds.append(out.detach().cpu().numpy())
            
            for batchidx in range(len(out)):
                # Compute scene_pred from the segmentation mask
                if mode == "test":
                    scene_pred_segmentation_mask = threshold_cutoff_connected_components(out[batchidx].cpu().numpy(), 
                                                                                         threshold_pixels=threshold_pixels, 
                                                                                         tol=1e-4)
                else:
                    scene_pred_segmentation_mask = threshold_cutoff(out[batchidx], threshold_pixels=threshold_pixels, 
                                                                    tol=1e-4)
                
                if classification_head:
                    scene_pred = scene_predbatch[batchidx].cpu().numpy()
                else:
                    scene_pred = scene_pred_segmentation_mask 

                # Compute extra metrics
                if mode == "test":
                    # Segmentation metrics
                    pred_binary = (out[batchidx] >= threshold).float()
                    cmat = binary_confusion_matrix(preds=pred_binary, target=y_target[batchidx]).cpu().numpy()
                    item_extra = {"TP": float(cmat[1, 1]),
                                    "FP": float(cmat[0, 1]),
                                    "TN": float(cmat[0, 0]),
                                    "FN": float(cmat[1, 0])}
                    if "ch4" in task:
                        # Quantify prediction
                        pred_binary = pred_binary.cpu().numpy()
                        if (scene_pred > threshold) and np.sum(pred_binary) > 0:
                            ch4_iter = task["ch4"][batchidx,0].cpu().numpy()
                            wind_vector = task["wind"][batchidx].cpu().numpy()
                            wind_speed = np.linalg.norm(wind_vector)
                            item_extra.update(quantification.obtain_flux_rate(ch4_iter, pred_binary, wind_speed=wind_speed, 
                                                                            a_u_eff=quantification.A_UEFF_S2,
                                                                            b_u_eff=quantification.B_UEFF_S2, 
                                                                            sig_xch4=quantification.SIGMA_CH4_S2_PPB,
                                                                            resolution=(10,10),
                                                                            return_std=True))     
                else:
                    item_extra = {}
                

                item_metrics = {"scene_pred": scene_pred, 
                                "scene_pred_segmentation_mask": scene_pred_segmentation_mask,
                                "target": target_binary[batchidx],
                                "location_name": task["location_name"][batchidx],
                                "tile": task["tile"][batchidx],
                                "id_loc_image": str(task["id_loc_image"][batchidx])}
                if classification_head:
                    item_metrics["scene_pred_classification_head"] = scene_pred
                item_metrics.update(item_extra)
                
                items.append(item_metrics)

    output = pd.DataFrame(items)

    if log_images:
        images = {
            "ch4": np.concatenate(ch4),
            "mbmp": np.concatenate(mbmp),
            "target": np.concatenate(target),
            "preds": np.concatenate(preds),
        }

        return output, images

    if loss_function is not None:
        log_loss = np.nanmean(np.array(lf))
        return output, log_loss
    
    return output



def load_stats_and_config(train_folder:str, 
                          model_name:str, 
                          basefolder_experiments:str,
                          fs:AbstractFileSystem,
                          logger:Optional[logging.Logger]=None,
                          csv_file:str="preds_test_2023",
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
        logger (Optional[logging.Logger], optional): Logger to use for logging.
            Defaults to None, in which case a logger is created.
        csv_file (str, optional): Name of the CSV file to load.
            Defaults to "preds_test_2023".

    Returns:
        tuple[pd.DataFrame, Optional[dict]]: 
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)

    csv_file = os.path.splitext(csv_file)[0]
    model_folder = pathjoin(basefolder_experiments, train_folder)
    path = pathjoin(model_folder, f"{csv_file}.csv")
    path_config = pathjoin(model_folder, "config_experiment.json")
    logger.info(f"Loading eval results from {path}")
    if path.startswith("az://"):
        with fs.open(path, "r") as fh:
            output = pd.read_csv(fh)
        if fs.exists(path_config):
            logger.info(f"Loading config from {path_config}")
            with fs.open(path_config, "r") as fh:
                config = json.load(fh)
        else:
            if "id_zero" not in csv_file:
                logger.warning(f"config file {path_config} not found")
            config = None
    else:
        output = pd.read_csv(path)
        if "id_zero" not in csv_file and os.path.exists(path_config):
            logger.info(f"Loading config from {path_config}")
            with open(path_config, "r") as fh:
                config = json.load(fh)
        else:
            if "id_zero" not in csv_file:
                logger.warning(f"config file {path_config} not found")
            config = None
    
    if output["scene_pred"].isna().any():
        print("Dropping:", output.loc[output["scene_pred"].isna()])
        output = output.loc[~output["scene_pred"].isna()].copy()
        
    output["model_name"] = model_name
    output["id_loc_image"] = output["id_loc_image"].apply(uuid.UUID)
    
    return output, config