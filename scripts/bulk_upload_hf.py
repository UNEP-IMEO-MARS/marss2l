import fsspec
from huggingface_hub import hf_file_system
from huggingface_hub import HfApi
from marss2l.utils import fs_from_path, setup_file_logger
from marss2l.dataframe_image_plumes import read_csv_images
from marss2l.huggingface import (
    REPO_ID,
    folder_hf,
    copy_item_image,
    export_dataframe_csvs_to_hf,
)
from typing import Optional
from tqdm import tqdm
import argparse
import tempfile
import os


def export_images_to_hf(
    csv_path: str,
    path_prepend_data: Optional[str] = None,
    show_pbar: bool = True,
    dry_run: bool = False,
):
    
    fsread = fs_from_path(csv_path)
    fswrite = hf_file_system.HfFileSystem()
    fslocal = fsspec.filesystem("file")
    logger = setup_file_logger("log", "export_images_to_hf")

    logger.info(f"Exporting images from {csv_path} to HuggingFace repo {REPO_ID}")

    if dry_run:
        logger.info("DRY RUN MODE: No files will be uploaded")

    dataframe_images = read_csv_images(
        csv_path, fs=fsread, path_prepend_data=path_prepend_data, add_case_study=True, split="all"
    )
    dataframe_images["folder_hf"] = dataframe_images.apply(
        lambda row: folder_hf(row.id_loc_image, row.split_name), axis=1
    )
    dataframe_images = dataframe_images.set_index("id_loc_image")

    # Load HF dataframe to get existing files

    folders_hf = set(dataframe_images["folder_hf"].unique().tolist())
    api = HfApi()

    for folder_hf_path in tqdm(folders_hf, disable=not show_pbar):
        data_folder = dataframe_images[dataframe_images["folder_hf"] == folder_hf_path]

        # check if HF folder exists and it is not empty
        if fswrite.exists(folder_hf_path):
            # Check if there are files inside
            files_in_folder = fswrite.ls(folder_hf_path)
            nfiles = len(files_in_folder)
            if nfiles > 0:
                logger.info(
                    f"Folder {folder_hf_path} already exists in HF repo and is not empty ({nfiles} files), skipping upload"
                )
                continue

        logger.info(f"Processing folder {folder_hf_path} with {len(data_folder)} images")

        # Create temporary folder for this batch
        with tempfile.TemporaryDirectory() as tmp_folder:
            # Export all files in data_folder to temporary folder
            for row in data_folder.itertuples():
                try:
                    files = copy_item_image(
                        row=row,
                        fsread=fsread,
                        fswrite=fslocal,
                        logger=logger,
                        dry_run=dry_run,
                        overwrite=False,
                        temp_folder=tmp_folder,
                    )
                    if files:
                        logger.debug(f"Prepared {len(files)} files for {row.Index}")
                except Exception as e:
                    logger.opt(exception=e).error(f"Error preparing files for {row.Index}")

            # Check if temp folder has files
            files_in_temp = os.listdir(tmp_folder)
            if len(files_in_temp) == 0:
                logger.warning(f"No files prepared for folder {folder_hf_path}, skipping upload")
                continue

            logger.info(
                f"Uploading {len(files_in_temp)} files from {tmp_folder} to {folder_hf_path}"
            )

            if dry_run:
                logger.info(f"[DRY RUN] Would upload folder {tmp_folder} to {folder_hf_path}")
            else:
                # Upload all the content from the local folder to HF
                path_in_repo = folder_hf_path.replace(f"datasets/{REPO_ID}/", "")
                try:
                    api.upload_folder(
                        folder_path=tmp_folder,
                        path_in_repo=path_in_repo,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                    )
                    logger.info(f"Uploaded folder {folder_hf_path} with {len(files_in_temp)} files")
                except Exception as e:
                    logger.opt(exception=e).error(f"Error uploading folder {folder_hf_path} to HF")

    # Export CSVs to HuggingFace
    export_dataframe_csvs_to_hf(
        dataframe_images=dataframe_images,
        fswrite=fswrite,
        logger=logger,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export images from Azure to HuggingFace dataset")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file with image metadata",
    )
    parser.add_argument(
        "--path_prepend_data",
        type=str,
        default=None,
        help="Path to prepend to data paths (s2path, plumepath, cloudmaskpath, ch4path). Optional.",
    )
    parser.add_argument(
        "--no_pbar",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode: log actions without uploading files",
    )

    args = parser.parse_args()

    export_images_to_hf(
        csv_path=args.csv_path,
        path_prepend_data=args.path_prepend_data,
        show_pbar=not args.no_pbar,
        dry_run=args.dry_run,
    )
