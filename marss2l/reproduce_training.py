from marss2l import train_final, loaders
from marss2l import eval_final
import os
import argparse
from marss2l.utils import setup_file_logger, get_remote_filesystem
import fsspec
import logging
from collections import OrderedDict
from marss2l.loaders import SPLITS
import gc
import torch

EXPERIMENTS_RUN = OrderedDict(
    {
        "marss2l": {
            "model_name": "UnetOriginal",
            "multipass": True,
            "cloud_mask": True,
            "wind": True,
            "split": "all",
            "batch_norm": True,
            "pos_weight": 10.0,
            "weight_by_ch4": True,
            "do_simulation": True,
            "bands_l8": True,
            "cat_mbmp": True,
        },
        "ch4net": {
            "model_name": "UnetOriginal",
            "multipass": False,
            "cloud_mask": False,
            "wind": False,
            "split": "all",
            "batch_norm": True,
            "pos_weight": 10.0,
            "do_simulation": False,
            "bands_l8": True,
            "cat_mbmp": False,
        },
        "ch4net_sim": {
            "model_name": "UnetOriginal",
            "multipass": False,
            "cloud_mask": False,
            "wind": False,
            "split": "all",
            "batch_norm": True,
            "pos_weight": 10.0,
            "do_simulation": True,
            "bands_l8": True,
            "cat_mbmp": False,
        },
        "marss2l_nosim": {
            "model_name": "UnetOriginal",
            "multipass": True,
            "cloud_mask": True,
            "wind": True,
            "split": "all",
            "batch_norm": True,
            "pos_weight": 10.0,
            "weight_by_ch4": True,
            "do_simulation": False,
            "bands_l8": True,
            "cat_mbmp": True,
        },
    }
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basename_dir",
        required=True,
        help="Directory to save the experiments results. e.g. train_logs_20240308",
    )
    parser.add_argument(
        "--csv_path",
        default=loaders.CSV_PATH_DEFAULT,
        help="Path to the csv with the data. Default: %(default)s",
    )
    parser.add_argument(
        "--num_workers",
        default=train_final.DEFAULT_NUM_WORKERS,
        type=int,
        help="Number of workers to use in the dataloader. Default: %(default)s",
    )
    parser.add_argument(
        "--num_workers_val",
        default=train_final.DEFAULT_NUM_WORKERS_VAL,
        type=int,
        help="Number of workers to use in the dataloader for validation. Default: %(default)s",
    )
    parser.add_argument(
        "--device_name",
        default=train_final.DEFAULT_DEVICE_NAME,
        help="Device to run the model. e.g. cuda or cpu. Default: %(default)s",
    )
    parser.add_argument(
        "--nepochs",
        type=int,
        help="Number of epochs to train the model. Default: %(default)s",
        default=train_final.DEFAULT_NEPOCHS,
    )
    parser.add_argument(
        "--n_samples_per_epoch_train",
        type=int,
        help="Number of samples per epoch to train the model default: %(default)s",
        default=loaders.NSAMPLES_PER_EPOCH_DEFAULT,
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache all images in memory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to train the model. Default: %(default)s",
        default=train_final.DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--batch_size_val",
        type=int,
        help="Batch size to validate the model. Default: %(default)s",
        default=train_final.DEFAULT_BATCH_SIZE_VAL,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate to train the model. Default: %(default)s",
        default=train_final.DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--window_size_training",
        type=int,
        help="Window size to train the model. Default: %(default)s",
        default=loaders.WINDOW_SIZE_TRAINING,
    )
    parser.add_argument(
        "--data_parallel",
        action=argparse.BooleanOptionalAction,
        default=train_final.DEFAULT_DATA_PARALLEL,
        help="Use data parallelism",
    )

    args_parsed = parser.parse_args()
    logger = logging.getLogger(__name__)

    csv_path = args_parsed.csv_path
    num_workers = args_parsed.num_workers
    num_workers_val = args_parsed.num_workers_val
    device_name = args_parsed.device_name
    nepochs = args_parsed.nepochs
    batch_size = args_parsed.batch_size
    batch_size_val = args_parsed.batch_size_val
    window_size_training = args_parsed.window_size_training
    learning_rate = args_parsed.learning_rate
    data_parallel = args_parsed.data_parallel
    cache = args_parsed.cache
    if cache:
        logger.info("Caching all images in memory. Setting start method to fork")
        torch.multiprocessing.set_start_method("fork")
    else:
        logger.info("Not caching all images in memory. Setting start method to spawn")
        torch.multiprocessing.set_start_method("spawn")

    n_samples_per_epoch_train = args_parsed.n_samples_per_epoch_train

    setup_file_logger("logs", "reproduce_training", logger)

    if csv_path.startswith("az://"):
        fsread = get_remote_filesystem()
        assert fsread.exists(
            csv_path
        ), f"Path {csv_path} does not exist. Should contain the csv with the data."
    else:
        assert os.path.exists(
            csv_path
        ), f"Path {csv_path} does not exist. Should contain the csv with the data."
        fsread = fsspec.filesystem("file")

    basename_dir = args_parsed.basename_dir.replace("\\", "/").rstrip("/")

    for experiment_name, experiment_config in EXPERIMENTS_RUN.items():
        try:
            output_dir = os.path.join(basename_dir, experiment_name)
            best_epoch_file = os.path.join(output_dir, "best_epoch")
            config_file = os.path.join(output_dir, "config_experiment.json")

            if not os.path.exists(best_epoch_file) or not os.path.exists(config_file):
                logger.info(f"Training model for experiment: {experiment_name}")

                train_final.run_training(
                    output_dir,
                    **experiment_config,
                    csv_path=csv_path,
                    classification_head=False,
                    num_workers=num_workers,
                    num_workers_val=num_workers_val,
                    nepochs=nepochs,
                    device_name=device_name,
                    batch_size=batch_size,
                    batch_size_val=batch_size_val,
                    window_size_training=window_size_training,
                    learning_rate=learning_rate,
                    data_parallel=data_parallel,
                    cache_all=cache,
                    n_samples_per_epoch_train=n_samples_per_epoch_train,
                    logger=logger,
                    fs=fsread,
                )
                if not os.path.exists(best_epoch_file) or not os.path.exists(
                    config_file
                ):
                    logger.error(f"Model not trained for experiment: {experiment_name}")
                    continue

            else:
                logger.info(
                    f"Trained model found for experiment: {experiment_name}. Skipping."
                )

        except Exception as e:
            logger.error(
                f"Error training model for experiment: {experiment_name}", exc_info=True
            )
            continue

        # Force garbage collection and empty cache of GPU
        torch.cuda.empty_cache()
        gc.collect()

        split = experiment_config["split"]

        splits_test = [SPLITS[split][-1]]
        if split == "ablation":
            splits_test.append(SPLITS["pre2023"][-1])

        # Evaluate model in splits
        for split_test in splits_test:
            preds_csv = os.path.join(output_dir, f"preds_{split_test}.csv")
            try:
                if not os.path.exists(preds_csv):
                    logger.info(
                        f"Evaluating model in split {split_test} for experiment: {experiment_name}"
                    )
                    eval_final.run_eval(
                        output_dir,
                        split=split_test,
                        csv_path=csv_path,
                        batch_size=batch_size_val,
                        num_workers=num_workers_val,
                        log_images=False,
                        logger=logger,
                        fs=fsread,
                    )
                    if not os.path.exists(preds_csv):
                        logger.error(
                            f"Model not evaluated in split {split_test} for experiment: {experiment_name}"
                        )
                        continue
                else:
                    logger.info(f"Preds found for experiment: {experiment_name}:")

            except Exception as e:
                logger.error(
                    f"Error evaluating model in split {split_test} for experiment: {experiment_name}",
                    exc_info=True,
                )
                continue

            # Force garbage collection and empty cache of GPU
            torch.cuda.empty_cache()
            gc.collect()
