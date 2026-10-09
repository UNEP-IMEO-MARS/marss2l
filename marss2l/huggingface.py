import os
import re
import tempfile
from typing import Optional, Union
from urllib.parse import unquote
from uuid import UUID

import fsspec
import pandas as pd
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import HfFileSystem, constants, hf_hub_download, hf_hub_url
from loguru import logger as _logger
from loguru._logger import Logger

REPO_ID = "UNEP-IMEO/MARS-S2L"

# Tables served from the Hugging Face hub cache instead of streamed on every read.
CACHED_EXTENSIONS = (".csv", ".parquet")

_REPO_TYPES = {"datasets": "dataset", "spaces": "space", None: "model"}

# [datasets/|spaces/]<namespace>/<name>[@<revision>]/<file>, the HfFileSystem form. Parsed locally
# (no ``repo_info`` call as in ``HfFileSystem.resolve_path``) so cached tables also open offline.
# ``refs/pr/<n>`` and ``refs/convert/<name>`` are the revisions that contain slashes.
_HF_REPO_PATH = re.compile(
    r"^(?:(?P<type>datasets|spaces)/)?(?P<repo_id>[^/@]+/[^/@]+)"
    r"(?:@(?P<revision>refs/pr/\d+|refs/convert/\w+|[^/]+))?/(?P<filename>.+)$"
)

# https://huggingface.co/[datasets/|spaces/]<namespace>/<name>/resolve/<revision>/<file>
_HF_RESOLVE_URL = re.compile(
    re.escape(constants.ENDPOINT)
    + r"/(?:(?P<type>datasets|spaces)/)?(?P<repo_id>[^/]+/[^/]+)/resolve/(?P<revision>[^/]+)/(?P<filename>[^?#]+)$"
)


def _cached_repo_file(
    *,
    repo_type: str,
    repo_id: str,
    revision: str,
    path_in_repo: str,
    local_dir: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
    endpoint: Optional[str] = None,
) -> str:
    """Local path of a file of a Hugging Face repo.

    Returns ``<local_dir>/<path_in_repo>`` if it exists, otherwise the copy in the Hugging Face
    hub cache (``~/.cache/huggingface/hub`` or ``$HF_HOME/hub``). ``hf_hub_download`` makes one
    HEAD request to detect a newer version, downloads only on change and falls back to the cached
    copy offline. ``token`` and ``endpoint`` default to the global Hugging Face settings.
    """
    if local_dir is not None:
        local_path = os.path.join(local_dir, path_in_repo)
        if os.path.exists(local_path):
            _logger.info(f"reading {path_in_repo} from local_dir {local_dir}")
            return local_path

    return hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token,
        endpoint=endpoint,
    )


class HfCachedFileSystem(HfFileSystem):
    """``HfFileSystem`` that reads ``.csv`` and ``.parquet`` files through the Hugging Face hub cache.

    Other files (image chips, weights) and every write go to ``HfFileSystem`` unchanged. Table
    reads use the filesystem's ``token`` and ``endpoint``, and work offline once cached.

    Args:
        local_dir: optional folder that mirrors the repo layout. A table present there is read
            from it instead of the hub, e.g. to check new release files before they are uploaded.
    """

    def __init__(self, *args, local_dir: Optional[str] = None, **storage_options):
        super().__init__(*args, **storage_options)
        self.local_dir = local_dir

    def _open(self, path: str, mode: str = "rb", revision: Optional[str] = None, **kwargs):
        match = _HF_REPO_PATH.match(path)
        if mode != "rb" or match is None or not path.endswith(CACHED_EXTENSIONS):
            return super()._open(path, mode=mode, revision=revision, **kwargs)

        if match["revision"] is not None:
            revision = unquote(match["revision"])
        local_path = _cached_repo_file(
            repo_type=_REPO_TYPES[match["type"]],
            repo_id=match["repo_id"],
            revision=revision or constants.DEFAULT_REVISION,
            path_in_repo=match["filename"],
            local_dir=self.local_dir,
            token=self.token,
            endpoint=self.endpoint,
        )
        return open(local_path, "rb")


class HfCachedHTTPFileSystem(HTTPFileSystem):
    """``HTTPFileSystem`` that reads Hugging Face ``.csv`` and ``.parquet`` URLs through the hub cache.

    Handles ``https://huggingface.co/datasets/<repo>/resolve/<revision>/<file>`` URLs (the form
    of ``CSV_PATH_DEFAULT_HF``); every other URL goes to ``HTTPFileSystem`` unchanged.
    """

    def __init__(self, *args, local_dir: Optional[str] = None, **storage_options):
        super().__init__(*args, **storage_options)
        self.local_dir = local_dir

    def _open(self, path: str, mode: str = "rb", **kwargs):
        match = _HF_RESOLVE_URL.match(path)
        if mode != "rb" or match is None or not match["filename"].endswith(CACHED_EXTENSIONS):
            return super()._open(path, mode=mode, **kwargs)

        local_path = _cached_repo_file(
            repo_type=_REPO_TYPES[match["type"]],
            repo_id=match["repo_id"],
            revision=unquote(match["revision"]),
            path_in_repo=unquote(match["filename"]),
            local_dir=self.local_dir,
        )
        return open(local_path, "rb")


CSV_PATH_DEFAULT_HF_CONVENTION = f"datasets/{REPO_ID}/validated_images_all.csv"
PARQUET_PATH_DEFAULT_HF_CONVENTION = f"datasets/{REPO_ID}/validated_images_all.parquet"
CSV_PLUME_PATH_DEFAULT_HF_CONVENTION = f"datasets/{REPO_ID}/validated_images_plumes.csv"
PARQUET_PATH_DEFAULT_HF_CONVENTION = f"datasets/{REPO_ID}/validated_images_plumes.parquet"

CSV_PATH_DEFAULT_HF = hf_hub_url(
    repo_id=REPO_ID, filename="validated_images_all.csv", repo_type="dataset"
)
PARQUET_PATH_DEFAULT_HF = hf_hub_url(
    repo_id=REPO_ID, filename="validated_images_all.parquet", repo_type="dataset"
)

CSV_PLUME_PATH_DEFAULT_HF = hf_hub_url(
    repo_id=REPO_ID, filename="validated_images_plumes.csv", repo_type="dataset"
)
PARQUET_PLUME_PATH_DEFAULT_HF = hf_hub_url(
    repo_id=REPO_ID, filename="validated_images_plumes.parquet", repo_type="dataset"
)

COLUMNS_DF_EXPORT = [
    "s2path",
    "plumepath",
    "cloudmaskpath",
    "ch4path",
    "wind_u",
    "wind_v",
    "vza",
    "sza",
    "percentage_clear",
    "tile",
    "isplume",
    "ch4_fluxrate",
    "ch4_fluxrate_std",
    "satellite",
    "tile_date",
    "notified",
    "id_location",
    "last_update",
    "location_name",
    "country",
    "lon",
    "lat",
    "offshore",
    "sector",
    "observability",
    "background_image_tile",
    # Acquisition time and geometry of the background (reference) pass. Needed to
    # convert its reflectances to radiances, which the shot-noise error
    # propagation requires for all four radiances, not just the target's.
    "tile_date_bg",
    "sza_bg",
    "vza_bg",
    "satellite_bg",
    # Provenance of the two solar zenith angles: "db" if the stored value was
    # kept, "computed" if it was substituted (see marss2l.solar_geometry).
    "sza_source",
    "sza_bg_source",
    "crs",
    "transform_a",
    "transform_b",
    "transform_c",
    "transform_d",
    "transform_e",
    "transform_f",
    "width",
    "height",
    "window_row_off",
    "window_col_off",
    "window_height",
    "window_width",
    "footprint",
    "plume",
    "split_name",
    "case_study",
    "wind_source",
    "id_loc_image",
]


def folder_hf(id_loc_image: UUID, split: str, 
              with_reponame: bool = True, 
              data_folder:str="data",
              nfolders:int=100) -> str:
    # Hugging Face has a 10000 items per folder restriction, we'll circunvent that by creating subfolders
    if isinstance(id_loc_image, str):
        id_loc_image = UUID(id_loc_image)
    if nfolders <= 1:
        internal_folder_name = "all"
    else:
        internal_folder_name = id_loc_image.int % nfolders
    
    if with_reponame:
        return f"datasets/{REPO_ID}/{data_folder}/{split}/{internal_folder_name}/"
    else:
        return f"{data_folder}/{split}/{internal_folder_name}/"


def image_path_hf(path: str, id_loc_image: UUID, split: str, 
                  with_reponame: bool = True, 
                  data_folder:str="data",
                  nfolders:int=100) -> str:
    # Hugging Face has a 10000 items per folder restriction, we'll circunvent that by creating subfolders
    return folder_hf(id_loc_image, split, with_reponame=with_reponame, 
                     data_folder=data_folder, nfolders=nfolders) + os.path.basename(path)


# from huggingface_hub import  hf_file_system
# fs = hf_file_system.HfFileSystem()
# fs.exists(f"datasets/{REPO_ID}/validated_images_all.csv")

FIELDS_TO_COPY = ["s2path", "plumepath", "cloudmaskpath", "ch4path"]


def copy_item_image(
    row: pd.Series,
    fsread: fsspec.AbstractFileSystem,
    fswrite: fsspec.AbstractFileSystem,
    logger: Logger,
    dry_run: bool = False,
    overwrite: bool = False,
    temp_folder: Optional[str] = None,
) -> list[str]:
    """
    Export a single image's files to HuggingFace or a temporary folder.

    Args:
        row: DataFrame row with image metadata
        fsread: Source filesystem
        fswrite: Dest filesystem
        logger: Logger instance
        dry_run: If True, only log actions without uploading
        overwrite: If True, upload even if file exists
        temp_folder: If provided, copy files to this local folder instead of uploading

    Returns:
        list[str]: List of files copied/uploaded
    """
    id_loc_image = row.Index if hasattr(row, "Index") else row.id_loc_image
    files_processed = []

    for field in FIELDS_TO_COPY:
        if (not row.isplume) and (field in {"plumepath", "ch4path"}):
            # Not needed
            continue

        path = getattr(row, field)
        if path is None:
            continue

        remote_path = image_path_hf(path, id_loc_image, row.split_name)
        if temp_folder is not None:
            # Copy to temporary folder for batch upload
            target_path = os.path.join(temp_folder, os.path.basename(remote_path))
            text = "copy"
        else:
            target_path = remote_path
            text = "upload"

        try:
            # Copy file
            if overwrite or not fswrite.exists(target_path):
                if dry_run:
                    logger.info(f"[DRY RUN] Would {text} {path} to {target_path}")
                else:
                    with fsread.open(path, "rb") as fsrc:
                        with fswrite.open(target_path, "wb") as fdst:
                            fdst.write(fsrc.read())
                files_processed.append(target_path)
        except Exception as e:
            logger.opt(exception=e).error(f"Error processing {path} to {target_path}")

    return files_processed


def export_dataframe_csvs_to_hf(
    dataframe_images: pd.DataFrame,
    fswrite: fsspec.AbstractFileSystem,
    logger: Logger,
    dry_run: bool = False,
) -> None:
    """
    Export dataframe to HuggingFace CSVs (main CSV and train/val/test splits).

    This function:
    1. Converts image paths to HuggingFace paths
    2. Uploads the main CSV file
    3. Uploads train/val/test split CSV files

    Args:
        dataframe_images (pd.DataFrame): DataFrame with image metadata. Must have columns
            from FIELDS_TO_COPY, 'isplume', 'id_loc_image', 'split_name', and columns in COLUMNS_DF_EXPORT.
        fswrite (fsspec.AbstractFileSystem): Filesystem for writing (e.g., HfFileSystem).
        logger (Logger): Logger instance for logging messages.
        dry_run (bool, optional): If True, only log actions without uploading files. Defaults to False.

    Returns:
        None
    """
    # Change the path in the dataframe to point to HF paths
    dataframe_images_export = dataframe_images.copy().reset_index()

    for field in FIELDS_TO_COPY:
        dataframe_images_export[field] = dataframe_images_export.apply(
            lambda row: (
                image_path_hf(
                    getattr(row, field), row.id_loc_image, row.split_name, with_reponame=False
                )
                if row.isplume or field not in {"plumepath", "ch4path"}
                else None
            ),
            axis=1,
        )
    dataframe_images_export = dataframe_images_export[COLUMNS_DF_EXPORT]

    # TODO Export parquet files also

    # Copy CSV file
    if dry_run:
        logger.info(f"[DRY RUN] Would upload CSV file to {CSV_PATH_DEFAULT_HF_CONVENTION}")
    else:
        logger.info(f"Uploading CSV file to {CSV_PATH_DEFAULT_HF_CONVENTION}")
        with fswrite.open(CSV_PATH_DEFAULT_HF_CONVENTION, "wb") as fdst:
            dataframe_images_export.to_csv(fdst, index=False)

    # Upload train, test, val.csv files by selecting train_2023, val_2023, test_2023 from split_name
    for split_name, traintestval_split in zip(
        ["train_2023", "val_2023", "test_2023"], ["train", "val", "test"]
    ):
        dataframe_split = dataframe_images_export[
            dataframe_images_export["split_name"] == split_name
        ]
        csv_split_path_hf = f"datasets/{REPO_ID}/{traintestval_split}.csv"

        if dry_run:
            logger.info(
                f"[DRY RUN] Would upload {traintestval_split} CSV file to {csv_split_path_hf}"
            )
        else:
            logger.info(f"Uploading {traintestval_split} CSV file to {csv_split_path_hf}")
            with fswrite.open(csv_split_path_hf, "wb") as fdst:
                dataframe_split.to_csv(fdst, index=False)
