from marss2l.trainer import Trainer
from marss2l.models import load_model
from marss2l import models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info
from marss2l.loaders import DatasetPlumes, SPLITS, CSV_PATH_DEFAULT, NSAMPLES_PER_EPOCH_DEFAULT,\
     DEFAULT_SPLIT, WINDOW_SIZE_TRAINING,DEFAULT_MULTIPASS,\
     DEFAULT_DO_SIMULATION, DEFAULT_FILM_TRAIN_ZERO_ID, DEFAULT_CAT_MBMP,\
     DEFAULT_BANDS_L8, DEFAULT_WIND, DEFAULT_NORM_WIND, DEFAULT_CLOUD_MASK,\
     DEFAULT_ONLY_ONSHORE, DEFAULT_ONLY_OFFSHORE
import argparse 
import os
import wandb
from typing import Tuple, Callable, Optional, List, Union
from marss2l.utils import CustomJSONEncoder, setup_file_logger, pathjoin, get_remote_filesystem
import json
import logging
import fsspec
from datetime import datetime
import shutil
import numpy as np

# Define constants for defaults
DEFAULT_MODEL_NAME = "film"
DEFAULT_CLASSIFICATION_HEAD = False
DEFAULT_BATCH_NORM = True
DEFAULT_DATA_PARALLEL = True
DEFAULT_NEPOCHS = 170
DEFAULT_BATCH_SIZE = 96
DEFAULT_BATCH_SIZE_VAL = 32
DEFAULT_DEVICE_NAME = "cuda"
DEFAULT_FINETUNE_FILM = False
DEFAULT_FINETUNE_CLASS_HEAD = False
DEFAULT_ONE_PARAM_PER_CHANNEL = True
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_POS_WEIGHT = 10
DEFAULT_WEIGHT_BY_CH4 = True
DEFAULT_PATIENCE_EARLY_STOPPING = 30
DEFAULT_NUM_WORKERS = 12
DEFAULT_NUM_WORKERS_VAL = 4
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_FINETUNE = False


PredType = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]

# How much to weight the classification loss vs the segmentation loss
WEIGHT_CLASSIFICATION_OUTPUT = WINDOW_SIZE_TRAINING * WINDOW_SIZE_TRAINING / 64

# See quantification module
# 8 is 8_000 / 1000 (conversion from ppb to ppmxm)
FACTOR_IME = (8 * 10*10 * 1_000 * 0.01604) / (1e6 * 22.4)
FACTOR_POS = 5

# Worker initialization function
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % (2**32)  # Ensure valid NumPy seed
        np.random.seed(seed)


def load_loss(weight_by_ch4: bool, pos_weight: float, 
              class_head:bool=False, 
              only_classification:bool=False,
              weight_by_ime:bool=False,
              device: torch.device=torch.device("cpu")) -> Callable:
    """
    Load the loss function based on the given parameters.

    Args:
        weight_by_ch4 (bool): Whether to weight the loss by CH4 concentration.
        pos_weight (float): Positive weight for the loss function.
        class_head (bool): Whether a classification head is used. If True, 
            there will be two outputs, one for the segmentation and one for the classification.
            Default is False.
        only_classification (bool): Whether to use only the classification head. Default is False.
        weight_by_ime (bool): Whether to weight the classification loss by the size of the plume.
            Default is False.
        device (torch.device): Device to run the model (e.g., cuda or cpu).

    Returns:
        Callable: The loss function to be used during training.
    """
    pos_weight_tensor = torch.Tensor([pos_weight])[:, None, None].to(device)
    bce_segmentation = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)

    # if class_head:
    #     bce_classification = nn.BCEWithLogitsLoss(reduction="none")
        # ime = (np.sum(methane_enhancement_image_values[binary_mask]) * 8 * 10*10 * 1_000 * 0.01604) / (1e6 * 22.4)

    if weight_by_ch4:
        def loss_fn(target: torch.Tensor, pred: PredType, ch4: torch.Tensor) -> torch.Tensor:
            target = target.squeeze(1)
            ch4 = ch4.squeeze(1)

            if class_head:
                # The proposed network does not use the classification head
                target_classification = (target.sum(dim=(-2, -1)) > 0).float()
                pred, pred_classification = pred

                # Weight by size of the plume
                if weight_by_ime:
                    ime = (target * ch4).sum(dim=(-2, -1)) * FACTOR_IME * FACTOR_POS
                    L = torch.sqrt(target.sum(dim=(-2, -1)) * 10 * 10)
                    good_inds = L > 0
                    ime_L = torch.where(good_inds, ime, 1) / torch.where(good_inds, L, 1)
                    # L = np.sqrt(npix_plume * np.prod(resolution))
                    pos_weight_class = torch.clamp(ime_L, 0.1, 10)
                else:
                    pos_weight_class = None

                ll_classification = F.binary_cross_entropy_with_logits(pred_classification.squeeze(1), 
                                                                       target_classification, 
                                                                       pos_weight=pos_weight_class,
                                                                       reduction="mean")

                ll_classification = WEIGHT_CLASSIFICATION_OUTPUT * ll_classification
                if only_classification:
                    return ll_classification
            else:
                ll_classification = 0

            
            ll = bce_segmentation(pred.squeeze(1), target)
            ch4_weight = torch.clamp(ch4, 100, 2000) / 1_000
            weight = ch4_weight * target + (1 - target)
            ll = ll * weight
            ll = ll.sum(dim=(-2, -1))
            return ll.mean() + ll_classification
    else:
        def loss_fn(target: torch.Tensor, pred: PredType) -> torch.Tensor:
            target = target.squeeze(1)

            if class_head:
                target_classification = (target.sum(dim=(-2, -1)) > 0).float()
                pred, pred_classification = pred
                ll_classification = F.binary_cross_entropy_with_logits(pred_classification.squeeze(1), 
                                                                       target_classification,
                                                                       reduction="mean")
                ll_classification = WEIGHT_CLASSIFICATION_OUTPUT * ll_classification
                if only_classification:
                    return ll_classification
            else:
                ll_classification = 0
            
            ll = bce_segmentation(pred.squeeze(1), target)
            ll = ll.sum(dim=(-2, -1))
            return ll.mean() + ll_classification

    return loss_fn

def run_training(output_dir:str, model_name:str=DEFAULT_MODEL_NAME, 
                 multipass:bool=DEFAULT_MULTIPASS, do_simulation:bool=DEFAULT_DO_SIMULATION, 
                 wind:bool=DEFAULT_WIND, cloud_mask:bool=DEFAULT_CLOUD_MASK, 
                 classification_head:bool=DEFAULT_CLASSIFICATION_HEAD,
                 norm_wind:bool=DEFAULT_NORM_WIND,
                 cat_mbmp:bool=DEFAULT_CAT_MBMP, bands_l8:bool=DEFAULT_BANDS_L8,
                 batch_norm:bool=DEFAULT_BATCH_NORM,
                 csv_path:str=CSV_PATH_DEFAULT,
                 split:str=DEFAULT_SPLIT,
                 film_train_zero_id:bool=DEFAULT_FILM_TRAIN_ZERO_ID,
                 logger:Optional[logging.Logger]=None,
                 num_workers:int=DEFAULT_NUM_WORKERS,
                 num_workers_val:int=DEFAULT_NUM_WORKERS_VAL,
                 cache_all:bool=True,
                 data_parallel:bool=DEFAULT_DATA_PARALLEL,
                 nepochs:int=DEFAULT_NEPOCHS,
                 batch_size:int = DEFAULT_BATCH_SIZE,
                 batch_size_val:int = DEFAULT_BATCH_SIZE_VAL,
                 window_size_training:int=WINDOW_SIZE_TRAINING,
                 n_samples_per_epoch_train:int=NSAMPLES_PER_EPOCH_DEFAULT,
                 device_name:str=DEFAULT_DEVICE_NAME,
                 all_locs:Optional[List[str]]=None,
                 finetune_film:bool = DEFAULT_FINETUNE_FILM,
                 finetune_classification_head:bool = DEFAULT_FINETUNE_CLASS_HEAD,
                 finetune:bool = DEFAULT_FINETUNE,
                 path_weights_forfinetuning:Optional[str]=None,
                 filename_weights_forfinetuning:str="best_epoch",
                 one_param_per_channel:bool=DEFAULT_ONE_PARAM_PER_CHANNEL,
                 learning_rate:float=DEFAULT_LEARNING_RATE,
                 pos_weight:float=DEFAULT_POS_WEIGHT,
                 weight_decay:float=DEFAULT_WEIGHT_DECAY,
                 weight_by_ch4:bool=DEFAULT_WEIGHT_BY_CH4,
                 patience_early_stopping:int=DEFAULT_PATIENCE_EARLY_STOPPING,
                 only_onshore:bool=DEFAULT_ONLY_ONSHORE,
                 only_offshore:bool=DEFAULT_ONLY_OFFSHORE,
                 fs:Optional[fsspec.AbstractFileSystem]=None):
    
    
    os.makedirs(output_dir, exist_ok=True)
    if not multipass:
        if cat_mbmp:
            logger.warning("cat_mbmp is only available for multipass, we will set it to False")
        cat_mbmp = False
    
    if csv_path.startswith("az://"):
        fs = get_remote_filesystem()
    else:
        fs = fsspec.filesystem("file")
    
    assert fs.exists(csv_path), f"Path {csv_path} does not exist. Should contain the csv with the data."

    assert bands_l8, "Only Landsat 8 bands are supported now"

    load_weights = finetune or finetune_film or finetune_classification_head

    film_dict_mapping = None
    if load_weights:
        if path_weights_forfinetuning is None:
            raise ValueError("Path to the weights for finetuning is required when finetuning")
        
        config_file_for_finetuning = pathjoin(path_weights_forfinetuning, "config_experiment.json")
        best_epoch_file_for_finetuning = pathjoin(path_weights_forfinetuning, 
                                                  filename_weights_forfinetuning)
        if not os.path.exists(config_file_for_finetuning) or not os.path.exists(best_epoch_file_for_finetuning):
            raise ValueError(f"Config file or best epoch file not found at {config_file_for_finetuning} or {best_epoch_file_for_finetuning}")
        with open(config_file_for_finetuning, "r") as f:
            config_base = json.load(f)
        
        model_name = config_base["model"]
        multipass = config_base["multipass"]
        cloud_mask = config_base["cloud_mask"]
        wind = config_base["wind"]
        norm_wind = config_base["norm_wind"]
        cat_mbmp = config_base["cat_mbmp"]
        bands_l8 = config_base["bands_l8"]
        film_train_zero_id = config_base["film_train_zero_id"]

        if model_name == "film":
            film_dict_mapping = config_base["film_dict_mapping"]
            assert film_dict_mapping is not None, "Film dict mapping is None but model is FiLM!"
    
    config_file = pathjoin(output_dir, "config_experiment.json")
    if os.path.exists(config_file):
        # copy config file to config_experiment_{now}.json
        nowstr = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_file_old = pathjoin(output_dir, f"config_experiment_{nowstr}.json")
        shutil.copy(config_file, config_file_old)

        logger.warning(f"Config file found at {config_file}. Copied to {config_file_old}. New config file will be created.")

    if model_name == "film" and film_dict_mapping is None:
        # This already added one to the index
        path_film_dict_file = pathjoin(os.path.dirname(csv_path), "location_name_mapping.json")
        logger.info(f"Loading film_dict_mapping from {path_film_dict_file}")
        with fs.open(path_film_dict_file, "r") as f:
            film_dict_mapping = json.load(f)

    split_train, split_val, _ = SPLITS[split]
    
    cpu_device = torch.device("cpu")
        # Common arguments for DatasetPlumes
    kwargs_dataset = {
        "device": cpu_device,
        "multipass": multipass,
        "cloud_mask": cloud_mask,
        "wind": wind,
        "norm_wind": norm_wind,
        "dataframe_or_csv_path": csv_path,
        "bands_l8": bands_l8,
        "logger": logger,
        "film_dict_mapping": film_dict_mapping,
        "film_train_zero_id": film_train_zero_id,
        "cat_mbmp": cat_mbmp,
        "all_locs": all_locs,
        "load_ch4": weight_by_ch4,
        "only_onshore": only_onshore,
        "only_offshore": only_offshore,
        "fs": fs
    }

    train_dataset = DatasetPlumes(
        mode="train",
        split=split_train,
        do_simulation=do_simulation,
        only_film_locs=False, # the zero id will be trained also if only_film_locs is True
        window_size_training=window_size_training,
        cache=cache_all,
        n_samples_per_epoch_train=n_samples_per_epoch_train,
        **kwargs_dataset
    )

    val_dataset = DatasetPlumes(
        mode="test",
        split=split_val,
        cache=True,
        do_simulation=False,
        **kwargs_dataset
    )

    
    # Load validation images to memory
    val_dataset.cache_all(nworkers=num_workers_val+num_workers)
    val_dataset.fs = None

    if cache_all:
        train_dataset.cache_all(nworkers=num_workers_val+num_workers)
        # Set to None to avoid fork issues
        train_dataset.fs = None
       

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True,
                              prefetch_factor=4,
                              persistent_workers=True)

    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size_val, 
                            shuffle=False,
                            num_workers=num_workers_val,
                            worker_init_fn=worker_init_fn,
                            pin_memory=True,
                            prefetch_factor=4,
                            persistent_workers=True)

    if train_dataset.film_dict_mapping is not None:
        max_index_film = max(train_dataset.film_dict_mapping.values()) + 1
    else:
        max_index_film = None

    torch.backends.cudnn.benchmark = True
    device = torch.device(device_name)

    model = load_model(model_name=model_name, 
                       in_channels=len(train_dataset.bands_out), 
                       classification_head=classification_head, 
                       max_index_film=max_index_film,
                       batch_norm=batch_norm,
                       one_param_per_channel=one_param_per_channel,
                       finetune_film=finetune_film,
                       finetune_class_head=finetune_classification_head,
                       logger=logger)
    model = model.to(device)
    if load_weights:
        logger.info(f"Loading weights from {best_epoch_file_for_finetuning}")
        models.load_weights(model, best_epoch_file_for_finetuning, device=None)

    loss_function = load_loss(weight_by_ch4=weight_by_ch4, 
                              class_head=classification_head,
                              pos_weight=pos_weight, 
                              only_classification=finetune_classification_head,
                              device=device)

    if data_parallel:
        logger.info("Using DataParallel")
        model = nn.DataParallel(model)
    if torch.__version__ >= "2.0":
        logger.info("Compiling model")
        model = torch.compile(model)

    ###### TRAINER
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      loss_function=loss_function,
                      save_path=output_dir,
                      learning_rate=learning_rate,
                      logger=logger,
                      weight_by_ch4=weight_by_ch4,
                      weight_decay=weight_decay,
                      device=device,
                      patience_early_stopping=patience_early_stopping)
    
    # TODO if load_weights load optimizer state dict? trainer.opt
    
    config_experiment = {"model":model_name, 
                         "multipass":multipass, 
                         "do_simulation":do_simulation, 
                         "wind":wind, 
                         "cloud_mask":cloud_mask, 
                         "classification_head":classification_head,
                         "norm_wind": norm_wind, 
                         "cat_mbmp":cat_mbmp, 
                         "batch_norm":batch_norm,
                         "csv_path":csv_path,
                         "all_data":split =="all",
                         "split": split,
                         "nepochs": nepochs,
                         "batch_size": batch_size,
                         "pos_weight": pos_weight,
                         "n_samples_per_epoch_train": n_samples_per_epoch_train,
                         "bands_l8":bands_l8, 
                         "n_samples_train": train_dataset.dataframe.shape[0],
                         "n_pos_train": train_dataset.total_pos,
                         "n_neg_train": train_dataset.total_neg,
                         "all_locs_train": train_dataset.all_locs,
                         "min_tile_date_train": min(train_dataset.dataframe['tile_date']),
                         "max_tile_date_train": max(train_dataset.dataframe['tile_date']),
                         "max_index_film": max_index_film,
                         "locs_few_samples_train": list(train_dataset.locs_few_samples),
                         "locs_few_neg_train": list(train_dataset.locs_few_neg),
                         "locs_few_pos_train": list(train_dataset.locs_few_pos),
                         "n_locs_train": len(train_dataset.all_locs),
                         "n_samples_val": val_dataset.dataframe.shape[0],
                         "n_pos_val": val_dataset.total_pos,
                         "n_neg_val": val_dataset.total_neg,
                         "all_locs_val": val_dataset.all_locs,
                         "min_tile_date_val": min(val_dataset.dataframe['tile_date']),
                         "max_tile_date_val": max(val_dataset.dataframe['tile_date']),
                         "n_locs_val": len(val_dataset.all_locs),
                         "film_dict_mapping": train_dataset.film_dict_mapping,
                         "film_dict_mapping_val": val_dataset.film_dict_mapping, # Should be the same but for sanity
                         "one_param_per_channel":one_param_per_channel,
                         "learning_rate": float(learning_rate),
                         "window_size_training":window_size_training,
                         "film_train_zero_id":film_train_zero_id,
                         "output_dir":output_dir,
                         "weight_by_ch4":weight_by_ch4,
                         "weight_decay":weight_decay,
                         "patience_early_stopping":patience_early_stopping,
                         "num_workers":num_workers,
                         "num_workers_val":num_workers_val,
                         "data_parallel":data_parallel,
                         "batch_size_val":batch_size_val,
                         "finetune_film":finetune_film,
                         "finetune_class_head":finetune_classification_head,
    }
    
    inprogress_config_file = os.path.join(output_dir, "config_experiment_inprogress.json")
    with open(inprogress_config_file, "w") as f:
        json.dump(config_experiment, f, cls=CustomJSONEncoder)
    
    with wandb.init(project="s2l89-model", reinit=True,
                     config=config_experiment) as run:
        ###### TRAINING
        logger.info(f"Training with config {config_experiment}")
        trainer.train(n_epochs=nepochs)

        # Save the config
        config_experiment["wandb_run_url"] = run.get_url()
        config_experiment["wandb_run_id"] = run.id
    
        with open(config_file, "w") as f:
            json.dump(config_experiment, f, cls=CustomJSONEncoder)
        
        os.remove(inprogress_config_file)

        logger.info(f"----- Training finished -----")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for plume detection")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="Model name (e.g., film, UnetOriginal, UnetPlusPlus)")
    parser.add_argument("--multipass", action=argparse.BooleanOptionalAction, default=DEFAULT_MULTIPASS, 
                        help="Use multipass input")
    parser.add_argument("--do_simulation", action=argparse.BooleanOptionalAction, default=DEFAULT_DO_SIMULATION, 
                        help="Use simulation data")
    parser.add_argument("--wind", action=argparse.BooleanOptionalAction, default=DEFAULT_WIND, 
                        help="Include wind data")
    parser.add_argument("--cloud_mask", action=argparse.BooleanOptionalAction, default=DEFAULT_CLOUD_MASK, 
                        help="Include cloud mask")
    parser.add_argument("--classification_head", action=argparse.BooleanOptionalAction, 
                        default=DEFAULT_CLASSIFICATION_HEAD, help="Use classification head")
    parser.add_argument("--norm_wind", action=argparse.BooleanOptionalAction, default=DEFAULT_NORM_WIND, 
                        help="Normalize wind data")
    parser.add_argument("--cat_mbmp", action=argparse.BooleanOptionalAction, default=DEFAULT_CAT_MBMP, 
                        help="Concatenate MBMP data")
    parser.add_argument("--bands_l8", action=argparse.BooleanOptionalAction, default=DEFAULT_BANDS_L8, 
                        help="Use Landsat 8 bands")
    parser.add_argument("--batch_norm", action=argparse.BooleanOptionalAction, default=DEFAULT_BATCH_NORM, 
                        help="Use batch normalization")
    parser.add_argument("--csv_path", type=str, default=CSV_PATH_DEFAULT, 
                        help="Path to the CSV file with data")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, 
                        choices=SPLITS.keys(),
                        help=f"Data split to use. one of {SPLITS.keys()}")
    parser.add_argument("--film_train_zero_id", action=argparse.BooleanOptionalAction, 
                        default=DEFAULT_FILM_TRAIN_ZERO_ID, help="Train with zero ID for FiLM")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, 
                        help="Number of workers for data loading")
    parser.add_argument("--num_workers_val", type=int, default=DEFAULT_NUM_WORKERS_VAL, 
                        help="Number of workers for validation data loading")
    parser.add_argument("--data_parallel", action=argparse.BooleanOptionalAction, 
                        default=DEFAULT_DATA_PARALLEL, help="Use data parallelism")
    parser.add_argument("--nepochs", type=int, default=DEFAULT_NEPOCHS, 
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=DEFAULT_BATCH_SIZE_VAL, 
                        help="Batch size for validation")
    parser.add_argument("--window_size_training", type=int, default=WINDOW_SIZE_TRAINING, 
                        help="Window size for training")
    parser.add_argument("--n_samples_per_epoch_train", type=int, 
                        default=NSAMPLES_PER_EPOCH_DEFAULT, 
                        help="Number of samples per epoch for training")
    parser.add_argument("--device_name", type=str, default=DEFAULT_DEVICE_NAME, 
                        help="Device name (e.g., cuda, cpu)")
    parser.add_argument("--finetune_film", action=argparse.BooleanOptionalAction, 
                        default=DEFAULT_FINETUNE_FILM, 
                        help="Finetune the FiLM parameters of the model")
    parser.add_argument("--finetune_class_head", action=argparse.BooleanOptionalAction, 
                        default=DEFAULT_FINETUNE_CLASS_HEAD,
                        help="Finetune the classification head")
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, 
                        default=DEFAULT_FINETUNE,
                        help="Finetune all the model")
    parser.add_argument("--only_onshore", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_ONLY_ONSHORE,
                        help="Use only onshore locations")
    parser.add_argument("--only_offshore", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_ONLY_OFFSHORE,
                        help="Use only offshore locations")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False,
                        help="Cache all images in memory")
    parser.add_argument("--path_weights_forfinetuning", type=str, default=None,
                        help="Path to the weights to be used for finetuning")
    parser.add_argument("--file_weights_forfinetuning", type=str, 
                        default="best_epoch",
                        help="File name of the weights to be used for finetuning")
    parser.add_argument("--one_param_per_channel", action=argparse.BooleanOptionalAction, 
                        default=DEFAULT_ONE_PARAM_PER_CHANNEL, 
                        help="Use one parameter per channel")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY,
                        help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, 
                        help="Learning rate")
    parser.add_argument("--pos_weight", type=float, default=DEFAULT_POS_WEIGHT, 
                        help="Positive weight for loss function")
    parser.add_argument("--weight_by_ch4", action=argparse.BooleanOptionalAction, default=DEFAULT_WEIGHT_BY_CH4, 
                        help="Weight loss by CH4 concentration")
    parser.add_argument("--patience_early_stopping", type=int, default=DEFAULT_PATIENCE_EARLY_STOPPING,
                        help="Patience for early stopping")
    args = parser.parse_args()

    
    logger = logging.getLogger(__name__)
    setup_file_logger("logs", "train_final", logger)

    if args.cache:
        logger.info("Caching all images in memory. Setting start method to fork")
        torch.multiprocessing.set_start_method('fork')
    else:
        logger.info("Not caching all images in memory. Setting start method to spawn")
        torch.multiprocessing.set_start_method('spawn')

    run_training(output_dir=args.output_dir, model_name=args.model_name, 
                 multipass=args.multipass, do_simulation=args.do_simulation, 
                 wind=args.wind, cloud_mask=args.cloud_mask, classification_head=args.classification_head,
                 norm_wind=args.norm_wind, cat_mbmp=args.cat_mbmp, bands_l8=args.bands_l8,
                 batch_norm=args.batch_norm, csv_path=args.csv_path, split=args.split,
                 film_train_zero_id=args.film_train_zero_id, num_workers=args.num_workers,
                 num_workers_val=args.num_workers_val, data_parallel=args.data_parallel,
                 nepochs=args.nepochs, batch_size=args.batch_size, batch_size_val=args.batch_size_val,
                 window_size_training=args.window_size_training, n_samples_per_epoch_train=args.n_samples_per_epoch_train,
                 device_name=args.device_name, all_locs=None, 
                 finetune_film=args.finetune_film,
                 finetune_classification_head=args.finetune_class_head,
                 finetune=args.finetune,
                 one_param_per_channel=args.one_param_per_channel,
                 weight_decay=args.weight_decay,
                 learning_rate=args.learning_rate, pos_weight=args.pos_weight,
                 path_weights_forfinetuning=args.path_weights_forfinetuning, 
                 filename_weights_forfinetuning=args.file_weights_forfinetuning,
                 weight_by_ch4=args.weight_by_ch4,
                 only_onshore=args.only_onshore,
                 only_offshore=args.only_offshore,
                 cache_all=args.cache,
                 patience_early_stopping=args.patience_early_stopping,
                 logger=logger)
