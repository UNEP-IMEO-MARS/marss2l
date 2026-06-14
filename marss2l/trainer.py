import logging
from datetime import datetime
from typing import Callable, Optional, List

import fsspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional.classification import binary_confusion_matrix
from marss2l.models import SegmentationModelMARSS2L
from marss2l.utils import fs_for_output, pathjoin

from marss2l.loss import (
    DEFAULT_POS_WEIGHT,
    DEFAULT_WEIGHT_BY_CH4,
    CH4_MIN_FOR_WEIGHTING,
    CH4_MAX_FOR_WEIGHTING,
    SCALE_CH4_LOSS,
    DeltaXCH4Loss,
)
from marss2l.metrics import get_scenelevel_metrics
from marss2l.mars_sentinel2 import quantification
from marss2l.mars_sentinel2.plume_detection import threshold_cutoff_connected_components

THRESHOLD_PIXELS = 100
DEFAULT_LEARNING_RATE = 5e-4


def threshold_cutoff(
    pred_continuous: torch.Tensor, threshold_pixels: float = THRESHOLD_PIXELS, tol: float = 1e-3
) -> float:
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


class Trainer:
    """
    Training class for the neural process models
    """

    def __init__(
        self,
        model: SegmentationModelMARSS2L,
        save_path: Optional[str] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        early_stopping=True,
        weight_by_ch4: bool = DEFAULT_WEIGHT_BY_CH4,
        pos_weight: float = DEFAULT_POS_WEIGHT,
        class_head: bool = False,
        only_classification: bool = False,
        weight_by_ime: bool = False,
        ch4_min_for_weighting: float = CH4_MIN_FOR_WEIGHTING,
        ch4_max_for_weighting: float = CH4_MAX_FOR_WEIGHTING,
        scale_ch4_loss: float = SCALE_CH4_LOSS,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        patience_early_stopping: int = 30,
        weight_decay: float = 1e-5,
        best_epoch_name: str = "best_epoch",
        last_epoch_name: str = "last_epoch",
        logger: Optional[logging.Logger] = None,
        weight_by_noise: bool = False,
        noise_warmup_epochs: int = 5,
        noise_transition_epochs: int = 20,
    ):

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        # Model and data
        self.model = model
        self.save_path = save_path
        self.early_stopping = early_stopping
        self.prediction_threshold = 0.5  # For validation metrics only
        self.device = device
        self.weight_by_ch4 = weight_by_ch4
        self.patience_early_stopping = patience_early_stopping
        self.weight_by_noise = weight_by_noise
        self.noise_warmup_epochs = noise_warmup_epochs
        self.noise_transition_epochs = noise_transition_epochs

        # Output filesystem: chosen by save_path, reusing fs only if it is an Azure FS.
        if save_path is not None:
            self.fswritter = fs_for_output(save_path, fs)
        else:
            self.fswritter = fsspec.filesystem("file")

        if self.save_path is not None:
            self.path_best_epoch = pathjoin(self.save_path, best_epoch_name)
            self.path_last_epoch = pathjoin(self.save_path, last_epoch_name)
        else:
            self.path_best_epoch = None
            self.path_last_epoch = None

        parameters_iterator = filter(lambda p: p.requires_grad, self.model.parameters())

        if weight_decay > 0:
            self.opt = torch.optim.AdamW(
                parameters_iterator, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            # Use Adam
            self.opt = torch.optim.Adam(parameters_iterator, lr=learning_rate)

        # Create loss function
        self.loss_function = DeltaXCH4Loss(
            weight_by_ch4=weight_by_ch4,
            pos_weight=pos_weight,
            class_head=class_head,
            only_classification=only_classification,
            weight_by_ime=weight_by_ime,
            device=device,
            ch4_min_for_weighting=ch4_min_for_weighting,
            ch4_max_for_weighting=ch4_max_for_weighting,
            scale_ch4_loss=scale_ch4_loss,
            weight_by_noise=weight_by_noise,
            noise_warmup_epochs=noise_warmup_epochs,
            noise_transition_epochs=noise_transition_epochs,
        )
        if self.weight_by_noise:
            self.loss_function_val = DeltaXCH4Loss(
                weight_by_ch4=weight_by_ch4,
                pos_weight=pos_weight,
                class_head=class_head,
                only_classification=only_classification,
                weight_by_ime=weight_by_ime,
                device=device,
                ch4_min_for_weighting=ch4_min_for_weighting,
                ch4_max_for_weighting=ch4_max_for_weighting,
                scale_ch4_loss=scale_ch4_loss,
                weight_by_noise=False,  # No noise weighting for validation
            )
        else:
            self.loss_function_val = self.loss_function


        nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.save_path is not None and self.fswritter.exists(self.path_best_epoch):
            path_best_epoch_old = self.path_best_epoch + f"_{nowstr}"
            # Copy old best epoch
            self.fswritter.copy(self.path_best_epoch, path_best_epoch_old)
            self.logger.warning(
                f"Best epoch file found at {self.path_best_epoch}. Copied to {path_best_epoch_old} to avoid overwriting."
            )

        if self.save_path is not None and self.fswritter.exists(self.path_last_epoch):
            path_last_epoch_old = self.path_last_epoch + f"_{nowstr}"
            # Copy old last epoch
            self.fswritter.copy(self.path_last_epoch, path_last_epoch_old)
            self.logger.warning(
                f"Last epoch file found at {self.path_last_epoch}. Copied to {path_last_epoch_old} to avoid overwriting."
            )

        # Losses
        self.metrics_early_stopping = []
        self.maes = []

    def _unravel_to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.view(-1).detach().cpu().numpy()

    def drop_to_default(self):
        for m in self.model.module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()

    def run_validation(
        self,
        test_loader: DataLoader,
        mode: str = "test",
        threshold: float = 0.5,
        apply_sigmoid: bool = True,
        extra_keys_to_gpu: Optional[List[str]] = None,
        threshold_pixels: int = THRESHOLD_PIXELS,
    ):
        """
        Run validation on a DataLoader and return predictions and metrics.

        Args:
            test_loader (DataLoader): DataLoader for test/validation data
            mode (str): "test" or "val" mode
            threshold (float): Threshold for binary predictions
            apply_sigmoid (bool): Whether to apply sigmoid to model output
            extra_keys_to_gpu (Optional[List[str]]): Extra keys to move to GPU
            threshold_pixels (int): Minimum number of pixels for scene prediction

        Returns:
            pd.DataFrame or tuple: DataFrame with predictions, optionally with images dict
        """
        # Generate predictions
        self.model.eval()
        items = []
        losses = []
        sig = nn.Sigmoid()


        if hasattr(self.model, "module"):
            model_instance = self.model.module
        else:
            model_instance = self.model
        classification_head = model_instance.classification_head is not None

        keys_to_gpu = ["site_ids", "y_context_ls0_0"]
        if self.weight_by_ch4:
            keys_to_gpu.append("ch4forweighting")
        if extra_keys_to_gpu is not None and len(extra_keys_to_gpu) > 0:
            keys_to_gpu += extra_keys_to_gpu

        # Log images for plotting
        with torch.no_grad():
            for task in tqdm(test_loader, desc="Eval model"):
                batch = {key: task[key].to(self.device) for key in keys_to_gpu}

                y_target = task["y_target"].to(self.device)

                out = self.model(batch)
                
                # Compute loss (matching train method pattern)
                kwargs_loss = {"target": y_target, "pred": out, "epoch": 0}
                if self.weight_by_ch4:
                    kwargs_loss["ch4"] = batch["ch4forweighting"]
                loss = self.loss_function_val(**kwargs_loss)
                losses.append(loss.item())

                if classification_head:
                    out, scene_predbatch = out
                    if apply_sigmoid:
                        scene_predbatch = sig(scene_predbatch)
                    scene_predbatch = scene_predbatch.squeeze(1)

                if apply_sigmoid:
                    out = sig(out)

                out = out.squeeze(1)  # (B, 1, H, W) -> (B, H, W)

                target_binary = task["isplume"].cpu().numpy()

                for batchidx in range(len(out)):
                    # Compute scene_pred from the segmentation mask
                    if mode == "test":
                        scene_pred_segmentation_mask = threshold_cutoff_connected_components(
                            out[batchidx].cpu().numpy(), threshold_pixels=threshold_pixels, tol=1e-4
                        )
                    else:
                        scene_pred_segmentation_mask = threshold_cutoff(
                            out[batchidx], threshold_pixels=threshold_pixels, tol=1e-4
                        )

                    if classification_head:
                        scene_pred = scene_predbatch[batchidx].cpu().numpy()
                    else:
                        scene_pred = scene_pred_segmentation_mask

                    # Compute extra metrics
                    if mode == "test":
                        # Segmentation metrics
                        pred_binary = (out[batchidx] >= threshold).float()
                        cmat = (
                            binary_confusion_matrix(preds=pred_binary, target=y_target[batchidx])
                            .cpu()
                            .numpy()
                        )
                        item_extra = {
                            "TP": float(cmat[1, 1]),
                            "FP": float(cmat[0, 1]),
                            "TN": float(cmat[0, 0]),
                            "FN": float(cmat[1, 0]),
                        }
                        if "ch4" in task:
                            # Quantify prediction
                            pred_binary = pred_binary.cpu().numpy()
                            if (scene_pred > threshold) and np.sum(pred_binary) > 0:
                                ch4_iter = task["ch4"][batchidx, 0].cpu().numpy()
                                wind_vector = task["wind"][batchidx].cpu().numpy()
                                wind_speed = np.linalg.norm(wind_vector)
                                item_extra.update(
                                    quantification.obtain_flux_rate(
                                        ch4_iter,
                                        pred_binary,
                                        wind_speed=wind_speed,
                                        a_u_eff=quantification.A_UEFF_S2,
                                        b_u_eff=quantification.B_UEFF_S2,
                                        sig_xch4=quantification.SIGMA_CH4_S2_PPB,
                                        resolution=(10, 10),
                                        return_std=True,
                                    )
                                )
                    else:
                        item_extra = {}

                    item_metrics = {
                        "scene_pred": scene_pred,
                        "scene_pred_segmentation_mask": scene_pred_segmentation_mask,
                        "target": target_binary[batchidx],
                        "location_name": task["location_name"][batchidx],
                        "tile": task["tile"][batchidx],
                        "id_loc_image": str(task["id_loc_image"][batchidx]),
                    }
                    if classification_head:
                        item_metrics["scene_pred_classification_head"] = scene_pred
                    item_metrics.update(item_extra)

                    items.append(item_metrics)

        output = pd.DataFrame(items)
        
        # Add average loss to the output
        avg_loss = float(np.mean(losses))

        return output, avg_loss

    def eval_epoch(self, epoch: int, val_loader: DataLoader, smoke_test: bool = False):

        preds, val_loss = self.run_validation(
            val_loader,
            mode="val"
        )

        # Compute metrics
        metrics = get_scenelevel_metrics(
            preds["scene_pred"], preds["target"], threshold=self.prediction_threshold
        )
        
        # Add validation loss to metrics
        metrics["val_loss"] = val_loss

        # Append segmentation mask metrics
        if "scene_pred_classification_head" in preds:
            metrics2 = get_scenelevel_metrics(
                preds["scene_pred_segmentation_mask"],
                preds["target"],
                threshold=self.prediction_threshold,
            )
            # rename keys
            metrics2 = {f"{key}_segmentation_mask": value for key, value in metrics2.items()}
            metrics.update(metrics2)

        if not smoke_test:
            wandb.log(metrics, step=epoch + 1)

        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs:int=100, smoke_test: bool = False):

        # wandb.log(metrics)

        best_metric = None
        best_epoch = None

        keys_to_gpu = ["site_ids", "y_context_ls0_0"]
        if self.weight_by_ch4:
            keys_to_gpu.append("ch4forweighting")
        if self.weight_by_noise:
            keys_to_gpu.append("ch4noise")
        
        if smoke_test:
            n_epochs = 2

        for epoch in range(n_epochs):
            self.epoch = epoch

            self.logger.info("Starting training epoch {}".format(epoch))
            self.model.train()
            # autograd.set_detect_anomaly(True)
            train_losses = []

            with tqdm(train_loader, unit="batch", desc="Training: ") as tepoch:
                for task in tepoch:

                    batch = {key: task[key].to(self.device) for key in keys_to_gpu}

                    y_target = task["y_target"].to(self.device)

                    out = self.model(batch)
                    kwargs_loss = {"target": y_target, "pred": out, "epoch": epoch}
                    if self.weight_by_ch4:
                        kwargs_loss["ch4"] = batch["ch4forweighting"]
                    if self.weight_by_noise:
                        kwargs_loss["ch4noise"] = batch["ch4noise"]
                    loss = self.loss_function(**kwargs_loss)
                    
                    # Autoregressive prev_step context
                    loss.backward()
                    tepoch.set_postfix(loss=loss.item())

                    self.opt.step()
                    self.opt.zero_grad()
                    train_losses.append(loss.item())

            metrics = self.eval_epoch(epoch, val_loader, smoke_test=smoke_test)
            metric_name_early_stopping = "average_precision"
            metric_early_stopping = metrics[metric_name_early_stopping]

            improved = False
            if (best_metric is None) or (metric_early_stopping >= best_metric):
                dict_save = {}
                dict_save.update(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.opt.state_dict(),
                    }
                )
                if not smoke_test and self.path_best_epoch is not None:
                    with self.fswritter.open(self.path_best_epoch, "wb") as f:
                        torch.save(dict_save, f)
                best_metric = metric_early_stopping
                best_epoch = epoch
                improved = True

            self.metrics_early_stopping.append(metric_early_stopping)
            average_train_loss = np.mean(np.array(train_losses))
            val_loss = metrics.get("val_loss", float('nan'))

            if not smoke_test:
                wandb.log({"epoch": epoch, "loss": average_train_loss}, step=epoch + 1)
            self.logger.info(
                f"Epoch {epoch} - Train loss: {average_train_loss:.4f} - Val loss: {val_loss:.4f} - Val {metric_name_early_stopping}: {metric_early_stopping:.4f} - Best Val {metric_name_early_stopping} ({best_epoch}): {best_metric:.4f}"
            )

            if (
                self.early_stopping
                and (not improved)
                and (epoch > self.patience_early_stopping)
                and (
                    np.max(np.array(self.metrics_early_stopping[-self.patience_early_stopping :]))
                    < best_metric
                )
            ):
                self.logger.info(
                    f"Stopping early: best {metric_name_early_stopping} {best_metric:.4f} not improved in last {self.patience_early_stopping} epochs"
                )
                break

        # Save last_epoch
        dict_save = {}
        dict_save.update(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            }
        )
        if not smoke_test and self.path_last_epoch is not None:
            with self.fswritter.open(self.path_last_epoch, "wb") as f:
                torch.save(dict_save, f)

        self.logger.info("Training complete!")
