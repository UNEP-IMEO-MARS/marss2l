import json
import logging
import os
from datetime import datetime
from typing import Annotated, List, Optional

import cyclopts
import fsspec
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, get_worker_info

from marss2l import models
from marss2l.seed import seed_all
from marss2l.dataframe_image_plumes import (
    CSV_LOCSOURCES_PATH_DEFAULT,
    CSV_PLUME_PATH_DEFAULT,
    load_dataframe_split,
    read_csv_images,
    read_csv_locs_sources,
    read_csv_plumes,
)
from marss2l.loaders import (
    CSV_PATH_DEFAULT,
    DEFAULT_BANDS_L8,
    DEFAULT_CAT_MBMP,
    DEFAULT_CLOUD_MASK,
    DEFAULT_DO_SIMULATION,
    DEFAULT_FILM_TRAIN_ZERO_ID,
    DEFAULT_MULTIPASS,
    DEFAULT_NORM_WIND,
    DEFAULT_ONLY_OFFSHORE,
    DEFAULT_ONLY_ONSHORE,
    DEFAULT_SIMULATE_ON_SOURCE_FRACTION,
    DEFAULT_SPLIT,
    DEFAULT_WIND,
    DIV_FACTOR_SIMULATE_SOURCES,
    NSAMPLES_PER_EPOCH_DEFAULT,
    SPLITS,
    WINDOW_SIZE_TRAINING,
    DatasetPlumes,
)
from marss2l.loss import (
    DEFAULT_POS_WEIGHT,
    DEFAULT_WEIGHT_BY_CH4,
    CH4_MIN_FOR_WEIGHTING,
    CH4_MAX_FOR_WEIGHTING,
    SCALE_CH4_LOSS,
    DEFAULT_NOISE_WARMUP_EPOCHS,
    DEFAULT_NOISE_TRANSITION_EPOCHS
)
from marss2l.models import load_model
from marss2l.trainer import Trainer, DEFAULT_LEARNING_RATE
from marss2l.utils import (
    CustomJSONEncoder,
    fs_from_path,
    pathjoin,
    setup_file_logger,
    setup_stream_logger
)

# Define constants for defaults
DEFAULT_MODEL_NAME = "UnetOriginal"
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
DEFAULT_PATIENCE_EARLY_STOPPING = 30
DEFAULT_NUM_WORKERS = 12
DEFAULT_NUM_WORKERS_VAL = 4
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_FINETUNE = False
DEFAULT_WEIGHT_BY_IME = False
DEFAULT_ONLY_CLASSIFICATION = False
DEFAULT_SCALE_CH4_LOSS = SCALE_CH4_LOSS
DEFAULT_WEIGHT_BY_NOISE = False
DEFAULT_WANDB_PROJECT = "s2l89-model"


# Worker initialization function
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % (2**32)  # Ensure valid NumPy seed
        np.random.seed(seed)


app = cyclopts.App(
    name="train_final",
    help="""Train MARS-S2L models for methane plume detection, segmentation, and quantification in Sentinel-2 and Landsat imagery.

Supports plume simulation. Models are trained with BCE loss and CH4 concentration weighting.

Smoke test mode (fast validation):
    python -m marss2l.train_final --smoke-test --output-dir dummy \
        --device-name cpu --cache-all --num-workers 2 --num-workers-val 2\
        --n-samples-per-epoch-train 80 --batch-size 16 --batch-size-val 8\
"""
)


@app.default
def run(
    output_dir: Annotated[str, cyclopts.Parameter(help="Output directory for model checkpoints and config")] = "train_output",
    model_name: Annotated[str, cyclopts.Parameter(help="Model architecture (UnetOriginal, CH4Net, etc.)")] = DEFAULT_MODEL_NAME,
    multipass: Annotated[bool, cyclopts.Parameter(help="Enable multi-pass detection mode")] = DEFAULT_MULTIPASS,
    do_simulation: Annotated[bool, cyclopts.Parameter(help="Enable plume simulation during training")] = DEFAULT_DO_SIMULATION,
    wind: Annotated[bool, cyclopts.Parameter(help="Include wind information as input")] = DEFAULT_WIND,
    cloud_mask: Annotated[bool, cyclopts.Parameter(help="Include cloud mask as input")] = DEFAULT_CLOUD_MASK,
    classification_head: Annotated[bool, cyclopts.Parameter(help="Add classification head for image-level detection")] = DEFAULT_CLASSIFICATION_HEAD,
    norm_wind: Annotated[bool, cyclopts.Parameter(help="Normalize wind vectors")] = DEFAULT_NORM_WIND,
    cat_mbmp: Annotated[bool, cyclopts.Parameter(help="Concatenate MBMP (Matched Band Methane Plume) features")] = DEFAULT_CAT_MBMP,
    bands_l8: Annotated[bool, cyclopts.Parameter(help="Use Landsat-8 band configuration")] = DEFAULT_BANDS_L8,
    batch_norm: Annotated[bool, cyclopts.Parameter(help="Use batch normalization in model")] = DEFAULT_BATCH_NORM,
    csv_path: Annotated[str, cyclopts.Parameter(help="Path to CSV file with image metadata")] = CSV_PATH_DEFAULT,
    csv_plume_path: Annotated[Optional[str], cyclopts.Parameter(help="Path to CSV file with plume metadata for simulation")] = CSV_PLUME_PATH_DEFAULT,
    csv_sources_path: Annotated[Optional[str], cyclopts.Parameter(help="Path to CSV file with source locations for simulation")] = CSV_LOCSOURCES_PATH_DEFAULT,
    split: Annotated[str, cyclopts.Parameter(help="Data split strategy (e.g., 'all', 'spatial', 'temporal')")] = DEFAULT_SPLIT,
    film_train_zero_id: Annotated[bool, cyclopts.Parameter(help="Train FiLM zero ID for unknown locations")] = DEFAULT_FILM_TRAIN_ZERO_ID,
    logger: Optional[logging.Logger] = None,
    num_workers: Annotated[int, cyclopts.Parameter(help="Number of dataloader workers for training")] = DEFAULT_NUM_WORKERS,
    num_workers_val: Annotated[int, cyclopts.Parameter(help="Number of dataloader workers for validation")] = DEFAULT_NUM_WORKERS_VAL,
    cache_all: Annotated[bool, cyclopts.Parameter(help="Cache all training images in memory")] = True,
    data_parallel: Annotated[bool, cyclopts.Parameter(help="Use DataParallel for multi-GPU training")] = DEFAULT_DATA_PARALLEL,
    nepochs: Annotated[int, cyclopts.Parameter(help="Number of training epochs")] = DEFAULT_NEPOCHS,
    batch_size: Annotated[int, cyclopts.Parameter(help="Training batch size")] = DEFAULT_BATCH_SIZE,
    batch_size_val: Annotated[int, cyclopts.Parameter(help="Validation batch size")] = DEFAULT_BATCH_SIZE_VAL,
    window_size_training: Annotated[int, cyclopts.Parameter(help="Window size for random crops during training")] = WINDOW_SIZE_TRAINING,
    n_samples_per_epoch_train: Annotated[int, cyclopts.Parameter(help="Number of training samples per epoch")] = NSAMPLES_PER_EPOCH_DEFAULT,
    device_name: Annotated[str, cyclopts.Parameter(help="Device for training (cuda, cpu)")] = DEFAULT_DEVICE_NAME,
    all_locs: Annotated[Optional[List[str]], cyclopts.Parameter(help="Filter training to specific locations")] = None,
    finetune_film: Annotated[bool, cyclopts.Parameter(help="Finetune FiLM layers only")] = DEFAULT_FINETUNE_FILM,
    finetune_classification_head: Annotated[bool, cyclopts.Parameter(help="Finetune classification head only")] = DEFAULT_FINETUNE_CLASS_HEAD,
    finetune: Annotated[bool, cyclopts.Parameter(help="Finetune entire model from checkpoint")] = DEFAULT_FINETUNE,
    path_weights_forfinetuning: Annotated[Optional[str], cyclopts.Parameter(help="Path to checkpoint directory for finetuning")] = None,
    filename_weights_forfinetuning: Annotated[str, cyclopts.Parameter(help="Checkpoint filename for finetuning")] = "best_epoch",
    one_param_per_channel: Annotated[bool, cyclopts.Parameter(help="Use one FiLM parameter per channel")] = DEFAULT_ONE_PARAM_PER_CHANNEL,
    learning_rate: Annotated[float, cyclopts.Parameter(help="AdamW learning rate")] = DEFAULT_LEARNING_RATE,
    pos_weight: Annotated[float, cyclopts.Parameter(help="Positive class weight for BCE loss")] = DEFAULT_POS_WEIGHT,
    weight_decay: Annotated[float, cyclopts.Parameter(help="AdamW weight decay")] = DEFAULT_WEIGHT_DECAY,
    weight_by_ch4: Annotated[bool, cyclopts.Parameter(help="Weight loss by CH4 concentration")] = DEFAULT_WEIGHT_BY_CH4,
    weight_by_ime: Annotated[bool, cyclopts.Parameter(help="Weight loss by IME (Integrated Mass Enhancement)")] = DEFAULT_WEIGHT_BY_IME,
    only_classification: Annotated[bool, cyclopts.Parameter(help="Train only classification head (no segmentation)")] = DEFAULT_ONLY_CLASSIFICATION,
    ch4_min_for_weighting: Annotated[float, cyclopts.Parameter(help="Min CH4 concentration for loss weighting")] = CH4_MIN_FOR_WEIGHTING,
    ch4_max_for_weighting: Annotated[float, cyclopts.Parameter(help="Max CH4 concentration for loss weighting")] = CH4_MAX_FOR_WEIGHTING,
    scale_ch4_loss: Annotated[float, cyclopts.Parameter(help="Scaling factor for CH4-weighted loss")] = DEFAULT_SCALE_CH4_LOSS,
    patience_early_stopping: Annotated[int, cyclopts.Parameter(help="Early stopping patience (epochs without improvement)")] = DEFAULT_PATIENCE_EARLY_STOPPING,
    weight_by_noise: Annotated[bool, cyclopts.Parameter(help="Weight loss by noise level (for simulated plumes)")] = DEFAULT_WEIGHT_BY_NOISE,
    noise_warmup_epochs: Annotated[int, cyclopts.Parameter(help="Epochs before noise-based weighting starts")] = DEFAULT_NOISE_WARMUP_EPOCHS,
    noise_transition_epochs: Annotated[int, cyclopts.Parameter(help="Epochs to transition noise-based weighting")] = DEFAULT_NOISE_TRANSITION_EPOCHS,
    simulate_on_source_fraction: Annotated[float, cyclopts.Parameter(help="Fraction of samples to simulate on known sources")] = DEFAULT_SIMULATE_ON_SOURCE_FRACTION,
    div_factor_simulate_sources: Annotated[float, cyclopts.Parameter(help="Divisor for source-based simulation intensity")] = DIV_FACTOR_SIMULATE_SOURCES,
    only_onshore: Annotated[bool, cyclopts.Parameter(help="Train only on onshore locations")] = DEFAULT_ONLY_ONSHORE,
    only_offshore: Annotated[bool, cyclopts.Parameter(help="Train only on offshore locations")] = DEFAULT_ONLY_OFFSHORE,
    path_prepend_data: Annotated[Optional[str], cyclopts.Parameter(help="Prepend path to data files")] = None,
    smoke_test: Annotated[bool, cyclopts.Parameter(help="Run 2 epochs of training with a subset of train and validation data")] = False,
    wandb_project: Annotated[str, cyclopts.Parameter(help="Wandb project name for logging")] = os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
    fsread: Optional[fsspec.AbstractFileSystem] = None,
    seed: Annotated[Optional[int], cyclopts.Parameter(help="Random seed for reproducibility (sets all random number generators)")] = None,
):
    # Set random seed if provided
    if seed is not None:
        seed_all(seed)
    
    # Setup logger
    if logger is None:
        if smoke_test:
            logger = setup_stream_logger(level=logging.INFO)
        else:
            logger = setup_file_logger("logs", "train_final")

    if not multipass:
        if cat_mbmp:
            logger.warning("cat_mbmp is only available for multipass, we will set it to False")
        cat_mbmp = False

    if fsread is None:
        fsread = fs_from_path(csv_path)

    # Output filesystem: reuse the (credentialed) input fs for blob output, else local.
    fswritter = fsread if output_dir.startswith("az://") else fsspec.filesystem("file")

    if not smoke_test:
        fswritter.makedirs(output_dir, exist_ok=True)

    assert bands_l8, "Only Landsat 8 bands are supported now"

    load_weights = finetune or finetune_film or finetune_classification_head

    film_dict_mapping = None
    if load_weights:
        if path_weights_forfinetuning is None:
            raise ValueError("Path to the weights for finetuning is required when finetuning")

        # Read finetuning inputs from blob when the path is on Azure, else local.
        fsfinetune = (
            fsread
            if path_weights_forfinetuning.startswith("az://")
            else fsspec.filesystem("file")
        )

        config_file_for_finetuning = pathjoin(path_weights_forfinetuning, "config_experiment.json")
        best_epoch_file_for_finetuning = pathjoin(
            path_weights_forfinetuning, filename_weights_forfinetuning
        )
        if not fsfinetune.exists(config_file_for_finetuning) or not fsfinetune.exists(
            best_epoch_file_for_finetuning
        ):
            raise ValueError(
                f"Config file or best epoch file not found at {config_file_for_finetuning} or {best_epoch_file_for_finetuning}"
            )
        with fsfinetune.open(config_file_for_finetuning, "r") as f:
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
    if fswritter.exists(config_file):
        # copy config file to config_experiment_{now}.json
        nowstr = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file_old = pathjoin(output_dir, f"config_experiment_{nowstr}.json")
        fswritter.copy(config_file, config_file_old)

        logger.warning(
            f"Config file found at {config_file}. Copied to {config_file_old}. New config file will be created."
        )

    if model_name == "film" and film_dict_mapping is None:
        # This already added one to the index
        path_film_dict_file = pathjoin(os.path.dirname(csv_path), "location_name_mapping.json")
        logger.info(f"Loading film_dict_mapping from {path_film_dict_file}")
        with fsread.open(path_film_dict_file, "r") as f:
            film_dict_mapping = json.load(f)

    split_train, split_val, _ = SPLITS[split]

    cpu_device = torch.device("cpu")

    dataframe_images = read_csv_images(csv_path, fs=fsread, path_prepend_data=path_prepend_data)
    if do_simulation:
        dataframe_plumes = read_csv_plumes(csv_plume_path, fs=fsread)
        if simulate_on_source_fraction > 0:
            dataframe_sources = read_csv_locs_sources(csv_sources_path, fs=fsread)
        else:
            dataframe_sources = None
    else:
        dataframe_plumes = None
        dataframe_sources = None

    # Split dataframes in train and val
    dataframe_images_train, dataframe_plumes_train, dataframe_sources_train = load_dataframe_split(
        dataframe_or_csv_path=dataframe_images,
        dataframe_or_csv_path_plumes=dataframe_plumes,
        dataframe_or_csv_path_sources=dataframe_sources,
        split=split_train,
        fs=fsread,
        logger=logger,
        all_locs=all_locs,
        load_plumes=do_simulation,
        only_onshore=only_onshore,
        only_offshore=only_offshore,
        smoke_test=smoke_test,
    )

    # TODO validation with simulation?
    dataframe_images_val, _, _ = load_dataframe_split(
        dataframe_or_csv_path=dataframe_images,
        dataframe_or_csv_path_plumes=None,
        dataframe_or_csv_path_sources=None,
        split=split_val,
        fs=fsread,
        logger=logger,
        all_locs=all_locs,
        load_plumes=False,
        only_onshore=only_onshore,
        only_offshore=only_offshore,
        smoke_test=smoke_test,
    )

    # Common arguments for DatasetPlumes
    kwargs_dataset = {
        "device": cpu_device,
        "multipass": multipass,
        "cloud_mask": cloud_mask,
        "wind": wind,
        "norm_wind": norm_wind,
        "bands_l8": bands_l8,
        "logger": logger,
        "film_dict_mapping": film_dict_mapping,
        "film_train_zero_id": film_train_zero_id,
        "cat_mbmp": cat_mbmp,
        "fs": fsread,
    }

    train_dataset = DatasetPlumes(
        mode="train",
        strprependlogs=split_train,
        do_simulation=do_simulation,
        only_film_locs=False,  # the zero id will be trained also if only_film_locs is True
        window_size_training=window_size_training,
        image_dataframe=dataframe_images_train,
        plume_dataframe=dataframe_plumes_train,
        sources_dataframe=dataframe_sources_train,
        cache=cache_all,
        n_samples_per_epoch_train=n_samples_per_epoch_train,
        simulate_on_source_fraction=simulate_on_source_fraction,
        div_factor_simulate_sources=div_factor_simulate_sources,
        **kwargs_dataset,
    )

    val_dataset = DatasetPlumes(
        mode="test",
        strprependlogs=split_val,
        image_dataframe=dataframe_images_val,
        cache=True,
        do_simulation=False,
        **kwargs_dataset,
    )

    # Load validation images to memory
    logger.info("Caching validation dataset in memory")
    val_dataset.cache_all(nworkers=num_workers_val + num_workers)
    val_dataset.fs = None

    if cache_all:
        logger.info("Caching training dataset in memory")
        train_dataset.cache_all(nworkers=num_workers_val + num_workers)
        # Set to None to avoid fork issues
        train_dataset.fs = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=not smoke_test,
        prefetch_factor=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers_val,
        worker_init_fn=worker_init_fn,
        pin_memory=not smoke_test,
        prefetch_factor=4,
        persistent_workers=True,
    )

    if train_dataset.film_dict_mapping is not None:
        max_index_film = max(train_dataset.film_dict_mapping.values()) + 1
    else:
        max_index_film = None

    torch.backends.cudnn.benchmark = True
    device = torch.device(device_name)

    model = load_model(
        model_name=model_name,
        in_channels=len(train_dataset.bands_out),
        classification_head=classification_head,
        max_index_film=max_index_film,
        batch_norm=batch_norm,
        one_param_per_channel=one_param_per_channel,
        finetune_film=finetune_film,
        finetune_class_head=finetune_classification_head,
        logger=logger,
    )
    model = model.to(device)
    if load_weights:
        logger.info(f"Loading weights from {best_epoch_file_for_finetuning}")
        models.load_weights(model, best_epoch_file_for_finetuning, device=None, fs=fsfinetune)

    if data_parallel:
        logger.info("Using DataParallel")
        model = nn.DataParallel(model)
    if torch.__version__ >= "2.0":
        logger.info("Compiling model")
        model = torch.compile(model)

    ###### TRAINER
    trainer = Trainer(
        model=model,
        save_path=output_dir,
        learning_rate=learning_rate,
        logger=logger,
        weight_by_ch4=weight_by_ch4,
        pos_weight=pos_weight,
        class_head=classification_head,
        only_classification=only_classification,
        weight_by_ime=weight_by_ime,
        ch4_min_for_weighting=ch4_min_for_weighting,
        ch4_max_for_weighting=ch4_max_for_weighting,
        scale_ch4_loss=scale_ch4_loss,
        weight_decay=weight_decay,
        device=device,
        patience_early_stopping=patience_early_stopping,
        weight_by_noise=weight_by_noise,
        noise_warmup_epochs=noise_warmup_epochs,
        noise_transition_epochs=noise_transition_epochs,
        fs=fswritter,
    )

    # TODO if load_weights load optimizer state dict? trainer.opt

    config_experiment = {
        "model": model_name,
        "multipass": multipass,
        "do_simulation": do_simulation,
        "wind": wind,
        "cloud_mask": cloud_mask,
        "classification_head": classification_head,
        "norm_wind": norm_wind,
        "cat_mbmp": cat_mbmp,
        "batch_norm": batch_norm,
        "csv_path": csv_path,
        "all_data": split == "all",
        "split": split,
        "nepochs": nepochs,
        "batch_size": batch_size,
        "pos_weight": pos_weight,
        "n_samples_per_epoch_train": n_samples_per_epoch_train,
        "bands_l8": bands_l8,
        "n_samples_train": train_dataset.image_dataframe.shape[0],
        "n_pos_train": train_dataset.total_pos,
        "n_neg_train": train_dataset.total_neg,
        "all_locs_train": train_dataset.all_locs,
        "min_tile_date_train": min(train_dataset.image_dataframe["tile_date"]),
        "max_tile_date_train": max(train_dataset.image_dataframe["tile_date"]),
        "max_index_film": max_index_film,
        "locs_few_samples_train": list(train_dataset.locs_few_samples),
        "locs_few_neg_train": list(train_dataset.locs_few_neg),
        "locs_few_pos_train": list(train_dataset.locs_few_pos),
        "n_locs_train": len(train_dataset.all_locs),
        "n_samples_val": val_dataset.image_dataframe.shape[0],
        "n_pos_val": val_dataset.total_pos,
        "n_neg_val": val_dataset.total_neg,
        "all_locs_val": val_dataset.all_locs,
        "min_tile_date_val": min(val_dataset.image_dataframe["tile_date"]),
        "max_tile_date_val": max(val_dataset.image_dataframe["tile_date"]),
        "n_locs_val": len(val_dataset.all_locs),
        "film_dict_mapping": train_dataset.film_dict_mapping,
        "film_dict_mapping_val": val_dataset.film_dict_mapping,  # Should be the same but for sanity
        "one_param_per_channel": one_param_per_channel,
        "learning_rate": float(learning_rate),
        "window_size_training": window_size_training,
        "film_train_zero_id": film_train_zero_id,
        "output_dir": output_dir,
        "weight_by_ch4": weight_by_ch4,
        "pos_weight": pos_weight,
        "weight_by_ime": weight_by_ime,
        "only_classification": only_classification,
        "ch4_min_for_weighting": ch4_min_for_weighting,
        "ch4_max_for_weighting": ch4_max_for_weighting,
        "scale_ch4_loss": scale_ch4_loss,
        "weight_by_noise": weight_by_noise,
        "noise_warmup_epochs": noise_warmup_epochs,
        "noise_transition_epochs": noise_transition_epochs,
        "simulate_on_source_fraction": simulate_on_source_fraction,
        "div_factor_simulate_sources": div_factor_simulate_sources,
        "weight_decay": weight_decay,
        "patience_early_stopping": patience_early_stopping,
        "num_workers": num_workers,
        "num_workers_val": num_workers_val,
        "data_parallel": data_parallel,
        "batch_size_val": batch_size_val,
        "finetune_film": finetune_film,
        "finetune_class_head": finetune_classification_head,
    }

    inprogress_config_file = pathjoin(output_dir, "config_experiment_inprogress.json")
    if not smoke_test:
        with fswritter.open(inprogress_config_file, "w") as f:
            json.dump(config_experiment, f, cls=CustomJSONEncoder)
    # s2l89-model
    with wandb.init(
        project=wandb_project,
        reinit=True,
        config=config_experiment,
        mode="disabled" if smoke_test else "online"
    ) as run:
        ###### TRAINING
        logger.info(f"Training with config {config_experiment}")
        trainer.train(train_loader, val_loader, n_epochs=nepochs, smoke_test=smoke_test)

        # Save the config
        if not smoke_test:
            config_experiment["wandb_run_url"] = run.get_url()
            config_experiment["wandb_run_id"] = run.id
            with fswritter.open(config_file, "w") as f:
                json.dump(config_experiment, f, cls=CustomJSONEncoder)

            fswritter.rm(inprogress_config_file)

        logger.info(f"----- Training finished -----")

    return True


if __name__ == "__main__":
    import sys
    # python -m marss2l.train_final --smoke-test --output-dir dummy --device-name cpu --cache-all --num_workers 2 --num_workers_val 2 --n_samples_per_epoch_train 1024
    
    # Check if --cache-all or --cache is in arguments to set multiprocessing start method
    cache_enabled = any(arg in ["--cache-all", "--cache"] for arg in sys.argv) and \
                    "--no-cache-all" not in sys.argv and "--no-cache" not in sys.argv
    
    if cache_enabled:
        print("Caching all images in memory. Setting start method to fork")
        torch.multiprocessing.set_start_method("fork")
    else:
        print("Not caching all images in memory. Setting start method to spawn")
        torch.multiprocessing.set_start_method("spawn")
    
    app()
