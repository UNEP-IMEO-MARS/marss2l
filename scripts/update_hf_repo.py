from huggingface_hub import hf_file_system
from marss2l.utils import fs_from_path, setup_file_logger
from marss2l.dataframe_image_plumes import read_csv_images
from marss2l.huggingface import (
    REPO_ID,
    CSV_PATH_DEFAULT_HF,
    copy_item_image,
    export_dataframe_csvs_to_hf,
)
from typing import Optional
from tqdm import tqdm
import argparse


fields_to_copy = ["s2path", "plumepath", "cloudmaskpath", "ch4path"]


def update_images_hf(
    csv_path: str,
    path_prepend_data: Optional[str] = None,
    show_pbar: bool = True,
    dry_run: bool = False,
    csv_only: bool = False,
):

    fsread = fs_from_path(csv_path)
    fswrite = hf_file_system.HfFileSystem()
    logger = setup_file_logger("log", "update_images_hf")
    logger.info(f"Updating images from {csv_path} to HuggingFace repo {REPO_ID}")

    if dry_run:
        logger.info("DRY RUN MODE: No files will be uploaded")

    dataframe_images = read_csv_images(
        csv_path, fs=fsread, path_prepend_data=path_prepend_data, add_case_study=True, split="all"
    )
    dataframe_images = dataframe_images.set_index("id_loc_image")

    if csv_only:
        # Metadata-only update: skip the per-image existence check over the whole
        # dataset. The image paths are rebuilt from the basename and the split, so
        # re-exporting a CSV that already carries HuggingFace paths leaves them
        # unchanged.
        logger.info("CSV ONLY MODE: skipping the image upload, refreshing the CSVs only")
        export_dataframe_csvs_to_hf(
            dataframe_images=dataframe_images,
            fswrite=fswrite,
            logger=logger,
            dry_run=dry_run,
        )
        return

    # Load HF dataframe to get existing files
    dataframe_images_hf = read_csv_images(CSV_PATH_DEFAULT_HF, fs=None)
    dataframe_images_hf = dataframe_images_hf.set_index("id_loc_image")

    # Copy only plumepath and ch4path if row.isplume

    for row in tqdm(
        dataframe_images.itertuples(), total=len(dataframe_images), disable=not show_pbar
    ):
        id_loc_image = row.Index
        overwrite = False
        if id_loc_image not in dataframe_images_hf.index:
            # New image
            overwrite = True
        else:
            row_hf = dataframe_images_hf.loc[id_loc_image]
            if row_hf.last_update < row.last_update:
                overwrite = True
            if row_hf.isplume != row.isplume:
                # Should not happen, but just in case
                overwrite = True

        try:
            copy_item_image(
                row=row,
                fsread=fsread,
                fswrite=fswrite,
                logger=logger,
                dry_run=dry_run,
                overwrite=overwrite,
                temp_folder=None,  # Direct upload, no temp folder
            )
        except Exception as e:
            logger.opt(exception=e).error(f"Error processing image {id_loc_image}")

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
    parser.add_argument(
        "--csv_only",
        action="store_true",
        help="Only refresh the CSV files (main and train/val/test splits), skipping the "
        "per-image upload check. Use it after a metadata-only change, such as adding columns.",
    )

    args = parser.parse_args()

    update_images_hf(
        csv_path=args.csv_path,
        path_prepend_data=args.path_prepend_data,
        show_pbar=not args.no_pbar,
        dry_run=args.dry_run,
        csv_only=args.csv_only,
    )
