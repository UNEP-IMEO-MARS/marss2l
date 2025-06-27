
import torch 
import numpy as np 

from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import os
import logging
from typing import Optional

import os
from marss2l.validation_utils import run_validation
from marss2l.metrics import get_scenelevel_metrics
from datetime import datetime
import shutil


class Trainer():
    """
    Training class for the neural process models
    """
    def __init__(self,
                 model:nn.Module,
                 train_loader:DataLoader,
                 val_loader:DataLoader,
                 loss_function:nn.Module,
                 save_path:str,
                 learning_rate:float,
                 early_stopping=True,
                 weight_by_ch4:bool=False,
                 device:Optional[torch.device]=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 patience_early_stopping:int=30,
                 weight_decay:float=1e-5,
                 best_epoch_name:str="best_epoch",
                 last_epoch_name:str="last_epoch",
                 logger:Optional[logging.Logger]=None):
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
      
        # Model and data
        self.model = model
        self.save_path = save_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.early_stopping = early_stopping
        self.prediction_threshold = 0.5 # For validation metrics only
        self.device = device
        self.weight_by_ch4 = weight_by_ch4
        self.patience_early_stopping = patience_early_stopping

        self.path_best_epoch = os.path.join(self.save_path, best_epoch_name)
        self.path_last_epoch = os.path.join(self.save_path, last_epoch_name)

        parameters_iterator = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if weight_decay > 0:
            self.opt = torch.optim.AdamW(parameters_iterator, lr=learning_rate, 
                                        weight_decay=weight_decay)
        else:
            # Use Adam
            self.opt = torch.optim.Adam(parameters_iterator, lr=learning_rate)
        
        self.loss_function = loss_function
        nowstr = datetime.now().strftime('%Y%m%d_%H%M%S')
        if os.path.exists(self.path_best_epoch):
            path_best_epoch_old = self.path_best_epoch + f"_{nowstr}"
            # Copy old best epoch
            shutil.copy(self.path_best_epoch, path_best_epoch_old)
            self.logger.warning(f"Best epoch file found at {self.path_best_epoch}. Copied to {path_best_epoch_old} to avoid overwriting.")
        
        if os.path.exists(self.path_last_epoch):
            path_last_epoch_old = self.path_last_epoch + f"_{nowstr}"
            # Copy old last epoch
            shutil.copy(self.path_last_epoch, path_last_epoch_old)
            self.logger.warning(f"Last epoch file found at {self.path_last_epoch}. Copied to {path_last_epoch_old} to avoid overwriting.")

        # Losses
        self.metrics_early_stopping = []
        self.maes = []

    def _unravel_to_numpy(self, x):
        return x.view(-1).detach().cpu().numpy()
    
    def drop_to_default(self):
        for m in self.model.module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()

    def eval_epoch(self, epoch:int):

        preds, log_loss = run_validation(self.val_loader, self.model,
                                         loss_function=self.loss_function,
                                         device=self.device,
                                         weight_by_ch4=self.weight_by_ch4,
                                         mode="val",log_images=False)

        
        # Compute metrics
        metrics = get_scenelevel_metrics(preds['scene_pred'], preds['target'], 
                              threshold=self.prediction_threshold)
        
        # Append segmentation mask metrics
        if "scene_pred_classification_head" in preds:
            metrics2 = get_scenelevel_metrics(preds['scene_pred_segmentation_mask'], preds['target'],
                                   threshold=self.prediction_threshold)
            # rename keys
            metrics2 = {f"{key}_segmentation_mask": value for key, value in metrics2.items()}
            metrics.update(metrics2)

        metrics["val_loss"] = float(log_loss)
        
        wandb.log(metrics, step=epoch+1)

        return metrics

    def train(self, n_epochs = 100):

        #wandb.log(metrics)

        best_metric = None
        best_epoch = None

        keys_to_gpu = ["site_ids", "y_context_ls0_0"]
        if self.weight_by_ch4:
            keys_to_gpu.append("ch4")
        
        for epoch in range(n_epochs):
            self.epoch = epoch

            self.logger.info("Starting training epoch {}".format(epoch))
            self.model.train()
            #autograd.set_detect_anomaly(True)
            train_losses = []

            with tqdm(self.train_loader, unit="batch", desc="Training: ") as tepoch:
                for task in tepoch:
                    
                    batch = {key: task[key].to(self.device) for key in keys_to_gpu}

                    y_target = task["y_target"].to(self.device)

                    out = self.model(batch)
                    kwargs_loss = {"target": y_target, "pred": out}
                    if self.weight_by_ch4:
                        kwargs_loss["ch4"] = batch["ch4"]
                    loss = self.loss_function(**kwargs_loss)
                    # Autoregressive prev_step context
                    loss.backward()
                    tepoch.set_postfix(loss=loss.item())
                        
                    self.opt.step()
                    self.opt.zero_grad()
                    train_losses.append(loss.item())
            
            metrics = self.eval_epoch(epoch)
            metric_name_early_stopping = "average_precision"
            metric_early_stopping = metrics[metric_name_early_stopping]

            improved = False
            if (best_metric is None) or (metric_early_stopping >= best_metric):
                dict_save = {}
                dict_save.update({"epoch": epoch,
                                  "model_state_dict": self.model.state_dict(),
                                  "optimizer_state_dict": self.opt.state_dict(),
                })
                torch.save(dict_save, self.path_best_epoch)
                best_metric = metric_early_stopping
                best_epoch = epoch
                improved = True

            self.metrics_early_stopping.append(metric_early_stopping)
            average_train_loss = np.mean(np.array(train_losses))

            wandb.log({"epoch": epoch, "loss": average_train_loss}, step=epoch+1)
            val_loss = metrics["val_loss"]
            self.logger.info(f"Epoch {epoch} - Train loss: {average_train_loss:.4f} - Val loss: {val_loss:.4f} -Val {metric_name_early_stopping}: {metric_early_stopping:.4f} -Best Val {metric_name_early_stopping} ({best_epoch}): {best_metric:.4f}")
            
            if self.early_stopping and (not improved) and (epoch > self.patience_early_stopping) and (np.max(np.array(self.metrics_early_stopping[-self.patience_early_stopping:])) < best_metric):
              self.logger.info(f"Stopping early: best {metric_name_early_stopping} {best_metric:.4f} not improved in last {self.patience_early_stopping} epochs")
              break
        
        # Save last_epoch
        dict_save = {}
        dict_save.update({"epoch": epoch,
                           "model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": self.opt.state_dict(),
        })
        torch.save(dict_save, self.path_last_epoch)
        
        self.logger.info("Training complete!")
