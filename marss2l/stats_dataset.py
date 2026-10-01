description = """
This script loads the dataset with all images and compute the stats per band.
"""

import os
from typing import Optional

import cyclopts
import fsspec
import fsspec.asyn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from marss2l import dataframe_image_plumes, loaders, shot_noise
from marss2l.mars_sentinel2 import quantification
from marss2l.utils import fs_from_path, pathjoin, setup_file_logger

#: Per-image fields the shot-noise statistics need, supplied by ``DatasetPlumes`` in
#: ``analysis_mode``. The ``_bg`` ones describe the reference pass and are empty for an
#: offshore scene, which has none.
GEOMETRY_KEYS = (
    "offshore",
    "satellite",
    "sza",
    "vza",
    "tile_date",
    "satellite_bg",
    "sza_bg",
    "tile_date_bg",
)


def _scalar(value):
    """Unwrap a single element out of whatever the default collate produced."""
    return value.item() if isinstance(value, torch.Tensor) else value


def reopen_filesystem_in_worker(_worker_id: int) -> None:
    """Give each forked data-loading worker its own connection to a remote store.

    An fsspec *async* filesystem (Azure, HTTP) carries an event loop and a thread that do
    not survive ``fork``: the first read in a worker raises ``This class is not fork-safe``.
    Dropping what the worker inherited makes it open its own. A no-op for local files.
    """
    info = torch.utils.data.get_worker_info()
    if info is None or not isinstance(
        getattr(info.dataset, "fs", None), fsspec.asyn.AsyncFileSystem
    ):
        return
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None
    fsspec.asyn.reset_lock()
    type(info.dataset.fs).clear_instance_cache()
    info.dataset.fs = None


def select_images(
    csv_path: str, fs, split: Optional[str] = None, path_prepend_data: Optional[str] = None
) -> pd.DataFrame:
    """The images to sweep.

    Args:
        csv_path: CSV with the image metadata.
        fs: Filesystem to read it through.
        split: One of ``dataframe_image_plumes.SPLITS`` (``train_2023``, ``val_2023``,
            ``test_2023``, ``no split``). None reads every image, which is also what a corpus
            with its own splits, such as CloudSEN12, wants.
        path_prepend_data: Prefix for the data paths, for a local copy of the dataset.
    """
    if split is None:
        return loaders.read_csv_images(
            csv_path, add_columns_for_analysis=False, fs=fs, path_prepend_data=path_prepend_data
        )
    dataframe, _, _ = dataframe_image_plumes.load_dataframe_split(
        split=split, dataframe_or_csv_path=csv_path, fs=fs, load_plumes=False
    )
    if path_prepend_data is not None:
        for field in ["s2path", "plumepath", "cloudmaskpath", "ch4path"]:
            dataframe[field] = dataframe[field].apply(
                lambda p: pathjoin(path_prepend_data, p) if isinstance(p, str) else p
            )
    return dataframe


def _write(stats_df: pd.DataFrame, fs, output_file: str) -> None:
    """Write the rows gathered so far, replacing the output file.

    Written to a temporary file and then moved into place, so the output is never a
    half-written CSV: a sweep killed while writing leaves the previous complete version.
    """
    partial = f"{output_file}.part"
    with fs.open(partial, "w") as f:
        stats_df.to_csv(f, index=False)
    fs.mv(partial, output_file)


def run(
    csv_path: str,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    output_file: Optional[str] = None,
    max_iter: Optional[int] = None,
    path_prepend_data: Optional[str] = None,
    split: Optional[str] = None,
    flush_every: int = 2000,
    dataset_name: Optional[str] = None,
    window_size: int = loaders.DEFAULT_WINDOW_SIZE_TRAINING,
    native_grid: bool = False,
):
    logger = setup_file_logger("logs", "stats_dataset")
    fs = fs_from_path(csv_path)
    if output_file is None:
        suffix = f"_{split.replace(' ', '_')}" if split else ""
        output_file = pathjoin(os.path.dirname(csv_path), f"stats_dataset{suffix}.csv")
    dataframe_data_traintest = select_images(
        csv_path, fs=fs, split=split, path_prepend_data=path_prepend_data
    )
    logger.info(f"Sweeping {len(dataframe_data_traintest)} images -> {output_file}")
    dataset = loaders.DatasetPlumes(
        mode="test",
        strprependlogs="no split",
        device="cpu",
        image_dataframe=dataframe_data_traintest,
        multipass=True,
        cloud_mask=True,
        wind=False,
        do_simulation=False,
        norm_wind=False,
        bands_l8=True,
        logger=logger,
        film_dict_mapping=None,
        film_train_zero_id=None,
        cat_mbmp=True,
        analysis_mode=True,
        window_size_training=window_size,
        window_size_data=window_size,
        native_grid=native_grid,
        # The images need not live where the CSV does (a local CSV pointing at the
        # Hugging Face copy, say), so the filesystem comes from an image path.
        fs=fs_from_path(str(dataframe_data_traintest["s2path"].iloc[0])),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        worker_init_fn=reopen_filesystem_in_worker,
    )

    stats: list = []
    written = 0
    with torch.no_grad():
        for task in tqdm(test_loader, desc="Eval model"):
            xbatch = task["y_context_ls0_0"]
            targetbatch = task["y_target"].squeeze(1)
            ch4batch = task["ch4"].squeeze(1)
            isplumebatch = task["isplume"].cpu().numpy()
            for batchidx in range(len(xbatch)):
                x = xbatch[batchidx]
                target = targetbatch[batchidx]
                ch4 = ch4batch[batchidx]
                location_name = task["location_name"][batchidx]
                tile = task["tile"][batchidx]
                id_loc_image = str(task["id_loc_image"][batchidx])
                wind_vector = task["wind"][batchidx].cpu().numpy()
                input_data = {
                    "location_name": location_name,
                    "tile": tile,
                    "id_loc_image": id_loc_image,
                    "isplume": isplumebatch[batchidx],
                    "wind_u": float(wind_vector[0]),
                    "wind_v": float(wind_vector[1]),
                }
                if dataset_name is not None:
                    input_data["dataset"] = dataset_name
                # bands out e.g. ['MBMP', 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B02_bg', 'B03_bg', 'B04_bg', 'B08_bg', 'B11_bg', 'B12_bg', 'U', 'V', 'cloudmask']
                stats_out = compute_stats(
                    dataset.bands_out,
                    isplume=input_data["isplume"],
                    ch4=ch4,
                    target=target,
                    x=x,
                    wind_vector=wind_vector,
                    pixel_size=float(_scalar(task["pixel_size"][batchidx])),
                )
                input_data.update(stats_out)

                geometry = {key: _scalar(task[key][batchidx]) for key in GEOMETRY_KEYS}
                input_data.update({k: v for k, v in geometry.items() if k != "tile_date"})
                input_data.update(
                    compute_shot_noise_stats(
                        dataset.bands_out,
                        x=x,
                        mask=valid_mask(dataset.bands_out, x),
                        **{k: v for k, v in geometry.items() if k != "offshore"},
                    )
                )
                stats.append(input_data)

            if len(stats) - written >= flush_every:
                _write(pd.DataFrame(stats), fs, output_file)
                written = len(stats)
                logger.info(f"{written}/{len(dataframe_data_traintest)} images written")

            if max_iter is not None and len(stats) >= max_iter:
                break

    _write(pd.DataFrame(stats), fs, output_file)
    logger.success(f"Wrote {len(stats)} rows to {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Shot-noise statistics
# ─────────────────────────────────────────────────────────────────────────────
def valid_mask(bands_out: list, x: torch.Tensor) -> torch.Tensor:
    """Pixels that enter a scene's statistics: no band is zero, and the cloud mask is clear.

    Zero is how the products encode invalid data. Dark pixels are kept on purpose: a
    brightness cut would make the reported floors depend on the threshold chosen.

    Args:
        bands_out: Band names, in the order they appear in ``x``.
        x: Image stack (C, H, W), reflectance x 2 for the spectral bands.

    Returns:
        Boolean tensor (H, W).
    """
    spectral = [i for i, b in enumerate(bands_out) if b not in {"MBMP", "U", "V", "cloudmask"}]
    mask = (x[spectral] != 0).all(dim=0)
    if "cloudmask" in bands_out:
        mask &= x[bands_out.index("cloudmask")] == 0
    return mask


def measured_noise_stats(
    bands_out: list, *, x: torch.Tensor, ch4: torch.Tensor, target: torch.Tensor
) -> dict:
    """The noise the operational retrieval shows: the standard deviation of its enhancement
    over valid pixels outside any annotated plume. On plume-free ground the retrieval should
    read zero, so what it reads is its noise.

    Args:
        bands_out: Band names, in the order they appear in ``x``.
        x: Image stack (C, H, W).
        ch4: Retrieved enhancement (H, W), ppb.
        target: Plume mask (H, W); all zero for a plume-free scene.

    Returns:
        ``npixelsvalid`` and ``ch4_valid_noplume_std`` (ppb).
    """
    valid = valid_mask(bands_out, x)
    background = ch4[valid & (target == 0)]
    return {
        "npixelsvalid": int(valid.sum()),
        "ch4_valid_noplume_std": background.std().item()
        if background.numel() > 1
        else float("nan"),
    }


def compute_shot_noise_stats(
    bands_out: list,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    satellite: str,
    sza: float,
    vza: float,
    tile_date: str,
    satellite_bg: str,
    sza_bg: float,
    tile_date_bg: str,
) -> dict:
    """Brightness of a scene and its three photon-noise floors, over its valid pixels.

    Both passes' reflectances are converted to radiance with their own geometry and date,
    the floors are computed per pixel (:func:`marss2l.shot_noise.eta_ladder`) and converted
    to ppb, and each is reported as its mean over the valid pixels.

    Args:
        bands_out: Band names, in the order they appear in ``x``. Landsat is relabelled to the
            Sentinel-2 names by the loader, so ``B11`` / ``B12`` whatever the instrument.
        x: Image stack (C, H, W), reflectance x 2.
        mask: Valid pixels, from :func:`valid_mask`.
        satellite, sza, vza, tile_date: Instrument and geometry of the target pass.
        satellite_bg, sza_bg, tile_date_bg: The same for the reference pass; empty for an
            offshore scene, whose single-pass retrieval has no L3 floor.

    Returns:
        ``radiance_B12_mean`` and ``radiance_B12_std`` (W m-2 sr-1 um-1), and for each rung
        ``sigma_ch4_{rung}_mean`` and ``epsilon_{rung}_mean`` (ppb). Empty when the scene has
        no valid pixel.
    """
    if not mask.any():
        return {}

    def radiance(band: str, background: bool) -> np.ndarray:
        reflectance = x[bands_out.index(band + ("_bg" if background else ""))][mask].numpy() / 2.0
        return shot_noise.radiance_from_reflectance(
            reflectance,
            satellite_bg if background else satellite,
            band,
            sza=sza_bg if background else sza,
            date_of_acquisition=tile_date_bg if background else tile_date,
        )

    radiance_23 = radiance(shot_noise.BAND_23, background=False)
    radiance_16 = radiance(shot_noise.BAND_16, background=False)
    has_reference = bool(satellite_bg) and bool(tile_date_bg) and not np.isnan(sza_bg)
    reference = (
        (
            radiance(shot_noise.BAND_23, background=True),
            radiance(shot_noise.BAND_16, background=True),
        )
        if has_reference
        else (None, None)
    )
    ladder = shot_noise.eta_ladder(
        radiance_23, radiance_16, *reference, satellite=satellite, satellite_bg=satellite_bg or None
    )

    stats_item = {
        "radiance_B12_mean": float(radiance_23.mean()),
        "radiance_B12_std": float(radiance_23.std(ddof=1))
        if radiance_23.size > 1
        else float("nan"),
    }
    for rung, eta in ladder.items():
        stats_item[f"sigma_ch4_{rung}_mean"] = float(
            shot_noise.sigma_delta_xch4(eta, satellite, sza, vza).mean()
        )
        stats_item[f"epsilon_{rung}_mean"] = float(
            shot_noise.epsilon(eta, satellite, sza, vza).mean()
        )
    return stats_item


def compute_stats(
    bands_out: list,
    isplume: int,
    ch4: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    wind_vector: np.ndarray,
    pixel_size: float = 10.0,
) -> dict:
    """
    Compute various statistics for given input data.

    Args:
        bands_out : list
            List of band names to compute statistics for.
        isplume : int
            Indicator whether the data contains a plume (1 if true, 0 otherwise).
        ch4 : torch.Tensor
            Tensor containing CH4 (methane) concentration data.
        target : torch.Tensor
            Tensor containing target labels indicating plume presence (1 for plume, 0 for no plume).
        x : torch.Tensor
            Tensor containing the data. Bands in this tensor are assumed to be in the same order as in `bands_out`.
        wind_vector : np.ndarray
            Numpy array representing the wind vector.
        pixel_size : float
            Side of a pixel in metres, for the flux quantification: 10 for the
            published chips, 20 for Sentinel-2 on its native grid, 30 for Landsat.


    Returns:
        dict
            Dictionary containing computed statistics. The keys include:
            - "ch4_mean": Mean of CH4 concentrations.
            - "ch4_std": Standard deviation of CH4 concentrations.
            - "ch4_min": Minimum of CH4 concentrations.
            - "ch4_max": Maximum of CH4 concentrations.
            - "ch4_mean_plume": Mean of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_std_plume": Standard deviation of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_min_plume": Minimum of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_max_plume": Maximum of CH4 concentrations within the plume (if isplume is 1).
            - "ch4_mean_noplume": Mean of CH4 concentrations outside the plume (if isplume is 1).
            - "ch4_std_noplume": Standard deviation of CH4 concentrations outside the plume (if isplume is 1).
            - "ch4_min_noplume": Minimum of CH4 concentrations outside the plume (if isplume is 1).
            - "ch4_max_noplume": Maximum of CH4 concentrations outside the plume (if isplume is 1).
            - Additional keys for flux rate quantification if isplume is 1.
            - For each band in bands_out (excluding "_bg" band, "U", "V"):
                - "{band}_mean": Mean of the band data.
                - "{band}_std": Standard deviation of the band data.
                - "{band}_min": Minimum of the band data.
                - "{band}_max": Maximum of the band data.
                - "{band}_mean_plume": Mean of the band data within the plume (if isplume is 1).
                - "{band}_std_plume": Standard deviation of the band data within the plume (if isplume is 1).
                - "{band}_min_plume": Minimum of the band data within the plume (if isplume is 1).
                - "{band}_max_plume": Maximum of the band data within the plume (if isplume is 1).
                - "{band}_mean_noplume": Mean of the band data outside the plume (if isplume is 1).
                - "{band}_std_noplume": Standard deviation of the band data outside the plume (if isplume is 1).
                - "{band}_min_noplume": Minimum of the band data outside the plume (if isplume is 1).
                - "{band}_max_noplume": Maximum of the band data outside the plume (if isplume is 1).
            - For "cloudmask" band:
                - "{cloudmask_value}": Count of each unique value in the cloudmask band.

    Notes:
    ------
    - The function assumes that the input tensors are properly aligned and have compatible shapes.
    - The function uses the `quantification` module to obtain flux rate statistics if `isplume` is 1.
    """
    stats_item = {}

    # Stats target (number of pixels 1)
    stats_item["npixelsplume"] = target.sum().item()
    stats_item["npixels"] = target.numel()

    stats_item.update(measured_noise_stats(bands_out, x=x, ch4=ch4, target=target))

    # mean, std, min, and max for CH4
    stats_item["ch4_mean"] = ch4.mean().item()
    stats_item["ch4_std"] = ch4.std().item()
    stats_item["ch4_min"] = ch4.min().item()
    stats_item["ch4_max"] = ch4.max().item()
    if isplume == 1:
        stats_item["ch4_mean_plume"] = ch4[target == 1].mean().item()
        stats_item["ch4_std_plume"] = ch4[target == 1].std().item()
        stats_item["ch4_min_plume"] = ch4[target == 1].min().item()
        stats_item["ch4_max_plume"] = ch4[target == 1].max().item()
        stats_item["ch4_mean_noplume"] = ch4[target == 0].mean().item()
        stats_item["ch4_std_noplume"] = ch4[target == 0].std().item()
        stats_item["ch4_min_noplume"] = ch4[target == 0].min().item()
        stats_item["ch4_max_noplume"] = ch4[target == 0].max().item()
        wind_speed = np.linalg.norm(wind_vector)
        stats_item.update(
            quantification.obtain_flux_rate(
                ch4.numpy(),
                target.numpy(),
                wind_speed=wind_speed,
                a_u_eff=quantification.A_UEFF_S2,
                b_u_eff=quantification.B_UEFF_S2,
                sig_xch4=quantification.SIGMA_CH4_S2_PPB,
                resolution=(pixel_size, pixel_size),
                # ch4 is in ppb; the function defaults to ppm.
                units_methane_enhancement="ppb",
                seed=42,
                return_std=True,
            )
        )

    # TODO average difference RGBNIR bands between image and background

    # TODO re-calculate MBMP masking plume?

    for bidx, b in enumerate(bands_out):
        if "_bg" in b:
            continue
        if b in {"U", "V"}:
            continue
        if b == "cloudmask":
            # Count unique values
            unique, counts = torch.unique(x[bidx], return_counts=True)
            stats_item.update({f"{b}_{u.item()}": c.item() for u, c in zip(unique, counts)})
        else:
            # Compute mean, std, min, and max. The /2 undoes the loader's reflectance x 2,
            # which MBMP, a band ratio, never carried.
            xband = x[bidx] if b == "MBMP" else x[bidx] / 2
            stats_item[f"{b}_mean"] = xband.mean().item()
            stats_item[f"{b}_std"] = xband.std().item()
            stats_item[f"{b}_min"] = xband.min().item()
            stats_item[f"{b}_max"] = xband.max().item()

            if isplume == 1:
                # Compute mean, std, min, and max inside and outside of the plume
                stats_item[f"{b}_mean_plume"] = xband[target == 1].mean().item()
                stats_item[f"{b}_std_plume"] = xband[target == 1].std().item()
                stats_item[f"{b}_min_plume"] = xband[target == 1].min().item()
                stats_item[f"{b}_max_plume"] = xband[target == 1].max().item()
                stats_item[f"{b}_mean_noplume"] = xband[target == 0].mean().item()
                stats_item[f"{b}_std_noplume"] = xband[target == 0].std().item()
                stats_item[f"{b}_min_noplume"] = xband[target == 0].min().item()
                stats_item[f"{b}_max_noplume"] = xband[target == 0].max().item()

    return stats_item


app = cyclopts.App(help=description)


@app.default
def main(
    csv_path: str = loaders.CSV_PATH_DEFAULT,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    output_file: Optional[str] = None,
    max_iter: Optional[int] = None,
    path_prepend_data: Optional[str] = None,
    split: Optional[str] = None,
    flush_every: int = 2000,
    dataset_name: Optional[str] = None,
    window_size: int = loaders.DEFAULT_WINDOW_SIZE_TRAINING,
    native_grid: bool = False,
) -> None:
    """Sweep the dataset and write one row of statistics per image.

    Args:
        csv_path: CSV with the image metadata.
        batch_size: Batch size for the data loader.
        num_workers: Worker processes for the data loader.
        output_file: Where to write. Defaults to ``stats_dataset[_<split>].csv`` beside the
            input CSV.
        max_iter: Stop after this many images.
        path_prepend_data: Prefix for the data paths, for a local copy of the dataset.
        split: Split to sweep (``train_2023``, ``val_2023``, ``test_2023``, ``no split``).
            Omit to read every image.
        flush_every: Write the rows gathered so far every this many images.
        dataset_name: Written to every row as a ``dataset`` column, so that sweeps of two
            corpora can be concatenated and still told apart.
        window_size: Chip size in pixels: 200 for the 10 m chips, 100 for Sentinel-2 under
            ``--native-grid`` and 67 for Landsat chips at 30 m (all 2 km).
        native_grid: Sweep Sentinel-2 on its native 20 m grid, recovered from the 10 m
            chips (``marss2l.resampling.chip_to_20m``). The photon-noise floors are per
            native pixel; the 10 m interpolation smooths the measured noise below them.
    """
    # The sweep is CPU-only, where fork is safe and much faster than spawn (which also
    # cannot pickle the dataset's file logger).
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn", force=True)
    run(
        csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        output_file=output_file,
        max_iter=max_iter,
        path_prepend_data=path_prepend_data,
        split=split,
        flush_every=flush_every,
        dataset_name=dataset_name,
        window_size=window_size,
        native_grid=native_grid,
    )


if __name__ == "__main__":
    app()
