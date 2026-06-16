import json
import logging
import math
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import fsspec
import numpy as np
import pandas as pd
from adlfs import AzureBlobFileSystem
from huggingface_hub.file_download import build_hf_headers
from shapely.geometry import base, mapping

from marss2l.config import AzureConfig

def get_remote_filesystem():
    cfg = AzureConfig.from_env()
    if not cfg.is_configured:
        raise ValueError(
            "Azure credentials are not configured. Set "
            f"{', '.join(AzureConfig.required_env_vars)} (see .env.sample). "
            "Public data is read from Hugging Face; Azure access requires credentials."
        )
    return AzureBlobFileSystem(
        account_name=cfg.account_name,
        sas_token=cfg.sas_token,
        assume_container_exists=True,
        default_fill_cache=False,
        default_cache_type=None,
    )


def isremotepath(path: str) -> bool:
    return path.startswith("az://") or path.startswith("https://")


# def fs_from_path(fs: Optional[fsspec.AbstractFileSystem], path: str) -> fsspec.AbstractFileSystem:
#     if fs is None:
#         if path.startswith("az://"):
#             return get_remote_filesystem()

#         if path.startswith("https://"):
#             headers = build_hf_headers()
#             return fsspec.filesystem("https", headers=headers)

#     if not isremotepath(path):
#         # Return local reader if not remote path
#         return fsspec.filesystem("file")

#     return fs

def fs_from_path(path: str) -> fsspec.AbstractFileSystem:
    
    if path.startswith("az://"):
        return get_remote_filesystem()

    if path.startswith("https://"):
        headers = build_hf_headers()
        return fsspec.filesystem("https", headers=headers)

    if not isremotepath(path):
        # Return local reader if not remote path
        return fsspec.filesystem("file")

    raise ValueError(f"Could not determine filesystem for path: {path}")


def fs_for_path(path: str, fs: Optional[fsspec.AbstractFileSystem] = None) -> fsspec.AbstractFileSystem:
    """Return the filesystem to use for reading/writing ``path``.

    Reuses the (credentialed) ``fs`` only when it already matches the backend of
    ``path`` (i.e. ``fs`` is an Azure FS and ``path`` is on ``az://``); otherwise it
    derives the right backend from ``path`` (local for non-``az://`` paths). This lets
    inputs, outputs and finetuning weights live on independent backends — e.g. images
    on local disk while weights/outputs are on blob, or vice versa.
    """
    if not path.startswith("az://"):
        return fsspec.filesystem("file")
    if isinstance(fs, AzureBlobFileSystem):
        return fs
    return fs_from_path(path)


def pathjoin(*parts):
    if not parts:
        return ""

    # Handle the first part specially to preserve leading slash if present
    result = str(parts[0]).replace("\\", "/")
    if len(result) > 1:  # Not just a single slash
        result = result.rstrip("/")

    # Process remaining parts normally
    for part in parts[1:]:
        part_str = str(part).replace("\\", "/").rstrip("/")
        if result and part_str:
            result += "/" + part_str
        elif part_str:  # If result is empty but part isn't
            result = part_str

    return result


def round_seconds(obj: datetime) -> datetime:
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)


def setup_stream_logger(logger: Optional[logging.Logger]=None, level=logging.INFO) -> logging.Logger:
    if logger is None:
        logger = logging.getLogger(__name__)
    """Setup a stream logger for the given logger"""
    if len(logger.handlers) > 0:
        ch = logger.handlers[0]
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)
        logger.setLevel(level)

    logger_azure = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
    logger_azure.setLevel(logging.WARNING)

    logger.propagate = False
    return logger


class CustomJSONEncoder(json.JSONEncoder):

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.default(*args, **kwds)

    def default(self, obj_to_encode):
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, "to_json"):
            return obj_to_encode.to_json()

        # UUIDs are not JSON serializable, so we'll convert them to strings.
        if isinstance(obj_to_encode, uuid.UUID):
            return str(obj_to_encode)

        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            item = obj_to_encode.item()
            if np.isnan(obj_to_encode):
                return None
            return item

        if (
            isinstance(obj_to_encode, float)
            or isinstance(obj_to_encode, int)
            and math.isnan(obj_to_encode)
        ):
            return None

        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        if isinstance(obj_to_encode, base.BaseGeometry):
            return mapping(obj_to_encode)
        if isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.round("1s").isoformat()
        if isinstance(obj_to_encode, datetime):
            return round_seconds(obj_to_encode).isoformat()

        # torch or tensorflow -> list, pretty straightforward.
        if hasattr(obj_to_encode, "numpy"):
            return obj_to_encode.numpy().tolist()

        if pd.isna(obj_to_encode):
            return None
        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)


def setup_file_logger(
    logdir, namefile: str, logger: Optional[logging.Logger] = None
) -> logging.Logger:
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    log_file_name = os.path.join(
        logdir, f"{namefile}_{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M')}.log"
    )
    if logger is None:
        logger = logging.getLogger(namefile)

    logger.propagate = False

    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Set the formatter for the handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger_azure = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
    logger_azure.setLevel(logging.WARNING)

    return logger
