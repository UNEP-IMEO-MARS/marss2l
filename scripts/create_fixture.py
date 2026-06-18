"""Build a small, balanced images-CSV fixture for the integration tests.

``read_csv_images`` normally reads the full ``validated_images_all.csv`` (~100 MB) from
Hugging Face. The train/eval integration tests only need a handful of rows, so this script
produces a tiny balanced subset (N plume / N no-plume per split) that is committed to the repo
and used as ``csv_path`` by both ``test_train_final`` and ``test_eval_final``. Per-image rasters
are still read from Hugging Face at test time; only the heavy CSV is replaced.

The raster path columns are written **as-is** (raw relative paths, e.g. ``data/...``). The tests
read the local fixture CSV with a local filesystem and load the rasters from Hugging Face by
passing ``path_prepend_data="datasets/<repo>"`` plus an ``HfFileSystem`` (see the tests/conftest).

Usage:
    python -m scripts.create_fixture --output tests/fixtures/images_fixture.csv
"""

import argparse

import pandas as pd

from marss2l.dataframe_image_plumes import (
    CSV_PATH_DEFAULT,
    CSV_PLUME_PATH_DEFAULT,
    load_dataframe_split,
    read_csv_images,
)
from marss2l.utils import fs_from_path, setup_stream_logger


def create_fixture(
    csv_path: str = CSV_PATH_DEFAULT,
    plume_csv_path: str = CSV_PLUME_PATH_DEFAULT,
    output: str = "tests/fixtures/images_fixture.csv",
    plume_output: str = "tests/fixtures/plumes_fixture.csv",
    n_per_class: int = 100,
    n_test: int = 30,
    splits: tuple[str, ...] = ("train_2023", "val_2023", "test_2023"),
):
    logger = setup_stream_logger(level="INFO")
    fs = fs_from_path(csv_path)

    # Processed dataframe: needed for the split predicates and the isplume / id_loc_image columns.
    logger.info(f"Reading full images CSV from {csv_path} (this is the heavy read)")
    df_proc = read_csv_images(csv_path, fs=fs)

    # Select a balanced set of id_loc_image per split.
    selected_ids: set = set()
    for split in splits:
        n = n_test if split == "test_2023" else n_per_class
        df_split, _, _ = load_dataframe_split(
            split=split,
            dataframe_or_csv_path=df_proc,
            fs=fs,
            logger=logger,
            load_plumes=False,
        )
        plume = df_split[df_split.isplume].head(n)
        noplume = df_split[~df_split.isplume].head(n)
        logger.info(
            f"Split {split}: selected {len(plume)} plume + {len(noplume)} no-plume "
            f"(requested {n} each, available {df_split.isplume.sum()}/{(~df_split.isplume).sum()})"
        )
        selected_ids |= set(plume.id_loc_image) | set(noplume.id_loc_image)

    # --- Images fixture -------------------------------------------------------
    # Read the RAW CSV (original serialization: plume/footprint WKT, transform_a..f) and subset it.
    # Raster paths are kept raw/relative (e.g. "data/..."); the tests prepend "datasets/<repo>" and
    # read them from Hugging Face via HfFileSystem (see tests/conftest.py).
    with fs.open(csv_path) as f:
        raw = pd.read_csv(f)

    fixture = raw[raw.id_loc_image.isin(selected_ids)].copy()
    fixture.to_csv(output, index=False)
    logger.info(f"Wrote {len(fixture)} image rows ({fixture.isplume.sum()} plume) to {output}")

    # --- Plumes fixture (for do_simulation=True in train_final) ---------------
    # read_csv_plumes carries no raster path columns, so the raw subset round-trips as-is.
    with fs.open(plume_csv_path) as f:
        raw_plumes = pd.read_csv(f)
    plumes_fixture = raw_plumes[raw_plumes.id_loc_image.isin(selected_ids)].copy()
    plumes_fixture.to_csv(plume_output, index=False)
    logger.info(
        f"Wrote {len(plumes_fixture)} plume rows "
        f"(for {plumes_fixture.id_loc_image.nunique()} images) to {plume_output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default=CSV_PATH_DEFAULT, help="Source images CSV")
    parser.add_argument(
        "--plume-csv-path", default=CSV_PLUME_PATH_DEFAULT, help="Source plumes CSV"
    )
    parser.add_argument(
        "--output", default="tests/fixtures/images_fixture.csv", help="Output images fixture path"
    )
    parser.add_argument(
        "--plume-output",
        default="tests/fixtures/plumes_fixture.csv",
        help="Output plumes fixture path",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=100, help="Plume/no-plume rows per train & val split"
    )
    parser.add_argument(
        "--n-test", type=int, default=30, help="Plume/no-plume rows for the test split"
    )
    parser.add_argument(
        "--splits",
        default="train_2023,val_2023,test_2023",
        help="Comma-separated splits to include",
    )
    args = parser.parse_args()

    create_fixture(
        csv_path=args.csv_path,
        plume_csv_path=args.plume_csv_path,
        output=args.output,
        plume_output=args.plume_output,
        n_per_class=args.n_per_class,
        n_test=args.n_test,
        splits=tuple(s.strip() for s in args.splits.split(",")),
    )


if __name__ == "__main__":
    main()
