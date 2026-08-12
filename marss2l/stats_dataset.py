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

from marss2l import dataframe_image_plumes, loaders
from marss2l.mars_sentinel2 import quantification
from marss2l.utils import fs_from_path, pathjoin, setup_file_logger

#: Per-scene fields the shot-noise statistics need, supplied by the dataset in
#: ``analysis_mode``. Absent from CSVs exported before the background geometry
#: columns existed, in which case those statistics are simply not computed.
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
    """Give each forked data-loading worker its own connection to the store.

    An fsspec **async** filesystem carries an event loop and the thread running
    it, and neither survives ``fork``: the first read in a worker raises
    ``RuntimeError: This class is not fork-safe``. Local sweeps never saw this
    because a local filesystem is synchronous.

    The fix is to drop everything inherited from the parent -- the module-level
    loop and its thread, the instance cache, and the dataset's own handle -- so
    that the worker builds its own. ``load_image_method`` already falls back to
    ``fs_from_path(path)`` when the dataset has no filesystem, and fsspec then
    caches one instance per worker, so this costs a single reconnection each.

    A no-op for a synchronous filesystem, which is why it can be passed
    unconditionally as the loader's ``worker_init_fn``.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return

    dataset = info.dataset
    filesystem = getattr(dataset, "fs", None)
    if not isinstance(filesystem, fsspec.asyn.AsyncFileSystem):
        return

    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None
    fsspec.asyn.reset_lock()
    type(filesystem).clear_instance_cache()
    dataset.fs = None


def select_images(
    csv_path: str,
    fs,
    split: Optional[str] = None,
    smoke_test: bool = False,
    path_prepend_data: Optional[str] = None,
) -> pd.DataFrame:
    """The images to sweep: a named split, optionally cut down to a smoke sample.

    Args:
        csv_path: CSV with the image metadata.
        fs: Filesystem to read it through.
        split: One of ``dataframe_image_plumes.SPLITS`` -- ``train_2023``,
            ``val_2023``, ``test_2023``, ``no split``. None reads every image, which
            is what the script did before the flag existed, and what the CloudSEN12
            corpus wants: its splits are the CloudSEN12 ones, unrelated to the
            MARS-S2L date and location cuts, and every scene of it is plume-free.
        smoke_test: Keep 20 images, 10 with plumes and 10 without, sampled with a
            fixed seed. Note this is **not** ``load_dataframe_split``'s own
            ``smoke_test``, which keeps 100 + 100 by ``head()`` and is depended on by
            training -- a different thing, deliberately not reused. A dataset with no
            plumes at all, such as CloudSEN12, simply yields the 10 without.
        path_prepend_data: Prefix for the data paths, needed for a HuggingFace copy.

    Returns:
        The image dataframe to sweep.
    """
    if split is None:
        dataframe = loaders.read_csv_images(
            csv_path, add_columns_for_analysis=False, fs=fs, path_prepend_data=path_prepend_data
        )
    else:
        dataframe, _, _ = dataframe_image_plumes.load_dataframe_split(
            split=split,
            dataframe_or_csv_path=csv_path,
            fs=fs,
            load_plumes=False,
        )
        if path_prepend_data is not None:
            for field in ["s2path", "plumepath", "cloudmaskpath", "ch4path"]:
                dataframe[field] = dataframe[field].apply(
                    lambda p: pathjoin(path_prepend_data, p) if isinstance(p, str) else p
                )

    if smoke_test:
        with_plume = dataframe[dataframe.isplume.astype(bool)]
        without_plume = dataframe[~dataframe.isplume.astype(bool)]
        dataframe = pd.concat(
            [
                with_plume.sample(min(len(with_plume), 10), random_state=0),
                without_plume.sample(min(len(without_plume), 10), random_state=0),
            ]
        )

    return dataframe


def _write(stats_df: pd.DataFrame, fs, output_file: str) -> None:
    """Write the rows gathered so far, replacing the file.

    Called periodically as well as at the end: a sweep over a full split takes
    hours, and losing all of it to a crash in the last batch is not acceptable.
    """
    with fs.open(output_file, "w") as f:
        stats_df.to_csv(f, index=False)


def run(
    csv_path: str,
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    output_file: Optional[str] = None,
    max_iter: Optional[int] = None,
    path_prepend_data: Optional[str] = None,
    split: Optional[str] = None,
    smoke_test: bool = False,
    flush_every: int = 2000,
    dataset_name: Optional[str] = None,
):
    logger = setup_file_logger("logs", "stats_dataset")
    fs = fs_from_path(csv_path)
    if output_file is None:
        # Derive the suffix from what was asked for rather than making the caller
        # name the file. The smoke sample goes local: csv_path is usually remote.
        if smoke_test:
            output_file = "stats_dataset_smoketest.csv"
        else:
            suffix = f"_{split.replace(' ', '_')}" if split else ""
            output_file = pathjoin(os.path.dirname(csv_path), f"stats_dataset{suffix}.csv")

    dataframe_data_traintest = select_images(
        csv_path,
        fs=fs,
        split=split,
        smoke_test=smoke_test,
        path_prepend_data=path_prepend_data,
    )
    logger.info(f"Sweeping {len(dataframe_data_traintest)} images -> {output_file}")
    fs_images = fs_from_path(str(dataframe_data_traintest["s2path"].iloc[0]))
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
        # Derived from an image path, not from the CSV path: the two need not live
        # in the same place -- a local CSV pointing at HuggingFace imagery is the
        # normal case while the backfilled CSV is unpublished. Passing None here
        # would not do it: DatasetPlumes turns None into a *local* filesystem, so
        # the per-path fallback in load_image_method never fires.
        fs=fs_images,
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
                )
                input_data.update(stats_out)

                geometry = {key: task[key][batchidx] for key in GEOMETRY_KEYS if key in task}
                if geometry:
                    input_data.update(
                        {k: _scalar(v) for k, v in geometry.items() if k != "tile_date"}
                    )
                    try:
                        input_data.update(
                            compute_shot_noise_stats(
                                dataset.bands_out,
                                x=x,
                                mask=valid_mask(dataset.bands_out, x),
                                satellite=_scalar(geometry["satellite"]),
                                sza=_scalar(geometry["sza"]),
                                vza=_scalar(geometry["vza"]),
                                tile_date=_scalar(geometry["tile_date"]),
                                satellite_bg=_scalar(geometry.get("satellite_bg", "")),
                                sza_bg=_scalar(geometry.get("sza_bg", float("nan"))),
                                tile_date_bg=_scalar(geometry.get("tile_date_bg", "")),
                            )
                        )
                    except Exception as e:
                        # One unconvertible scene must not lose the whole sweep.
                        logger.opt(exception=e).warning(
                            f"Shot-noise stats failed for {tile} ({id_loc_image})"
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


def valid_mask(bands_out: list, x: torch.Tensor) -> torch.Tensor:
    """Pixels that may enter a scene's statistics: not zero anywhere, not cloudy.

    Two exclusions and no more (the epic's first-pass masking rule). A pixel is out
    if **any** contributing band is zero -- which is how the loader encodes invalid
    data, after mapping NaN to 0 -- or if the cloud mask flags it.

    Zero matters more than it looks: a zero reflectance gives a zero radiance, hence
    an SNR of 0 and an infinite eta, so one masked pixel would take a scene mean to
    infinity without raising anything.

    **Dark surfaces stay in.** The SNR rescaling is optimistic there, which keeps the
    floors floors; a brightness cut would mean choosing a threshold, and a threshold
    chosen to flatter the figures is worse than a stated caveat.

    Args:
        bands_out: Band names, in the order they appear in ``x``.
        x: Image stack (C, H, W), reflectance x 2 for the spectral bands.

    Returns:
        Boolean tensor (H, W), True where the pixel is usable.
    """
    spectral = [i for i, b in enumerate(bands_out) if b not in {"MBMP", "U", "V", "cloudmask"}]
    mask = (x[spectral] != 0).all(dim=0)

    if "cloudmask" in bands_out:
        mask &= x[bands_out.index("cloudmask")] == 0

    return mask


def _summary(values: torch.Tensor, prefix: str) -> dict:
    """mean / meanabs / std / min / max for one quantity.

    ``meanabs`` is the addition: for a quantity that should be zero on plume-free
    ground -- delta XCH4, log MBMP -- the plain mean cancels and says nothing about
    magnitude, which is exactly what the measurement in the epic's section 4.4 needs.
    """
    if values.numel() == 0:
        return dict.fromkeys(
            [f"{prefix}_{s}" for s in ("mean", "meanabs", "std", "min", "max")], float("nan")
        )

    return {
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_meanabs": values.abs().mean().item(),
        f"{prefix}_std": values.std().item() if values.numel() > 1 else float("nan"),
        f"{prefix}_min": values.min().item(),
        f"{prefix}_max": values.max().item(),
    }


def measured_noise_stats(
    bands_out: list,
    *,
    x: torch.Tensor,
    ch4: torch.Tensor,
    target: torch.Tensor,
    isplume: int,
) -> dict:
    """O-base: the noise the operational retrieval actually exhibits.

    On plume-free ground the retrieval should read zero, so what it does read is its
    noise. Reported over **valid** pixels only -- a zero-filled or cloudy pixel is
    not a measurement -- and separately over the plume-free pixels of a scene that
    has a plume, which is the variant the figures use.

    ``meanabs`` earns its place here: for a quantity centred on zero the plain mean
    cancels and says nothing about magnitude.

    Args:
        bands_out: Band names, in the order they appear in ``x``.
        x: Image stack (C, H, W).
        ch4: Retrieved enhancement (H, W), ppb.
        target: Plume mask (H, W).
        isplume: 1 if the scene has a plume.

    Returns:
        ``npixelsvalid`` plus ``ch4_valid_*``, ``ch4_valid_noplume_*`` and
        ``log_mbmp_valid_*``.
    """
    mask = valid_mask(bands_out, x)
    stats_item = {"npixelsvalid": int(mask.sum().item())}

    stats_item.update(_summary(ch4[mask], "ch4_valid"))
    if isplume == 1:
        stats_item.update(_summary(ch4[mask & (target == 0)], "ch4_valid_noplume"))

    # log MBMP, for completeness: the natural units in which to check the variance
    # decomposition, though the figures report the gap in ppb.
    if "MBMP" in bands_out:
        mbmp = x[bands_out.index("MBMP")][mask]
        stats_item.update(_summary(torch.log(mbmp[mbmp > 0]), "log_mbmp_valid"))

    return stats_item


def compute_shot_noise_stats(
    bands_out: list,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    satellite: str,
    sza: float,
    vza: float,
    tile_date: str,
    satellite_bg: str = "",
    sza_bg: float = float("nan"),
    tile_date_bg: str = "",
) -> dict:
    """Per-scene radiance and the three shot-noise floors, in ppb.

    Converts both passes' reflectances to radiances, builds the L1/L2/L3 rungs per
    pixel, maps each to a minimum significant enhancement and a propagated standard
    deviation, and reports the average over the valid pixels of the scene.

    **Radiance is reported for the current image only.** The reference pass's
    radiances are needed *inside* L3 -- it has four terms -- but they are an
    intermediate, not a quantity we want per scene.

    L3 is skipped when there is no reference pass: an offshore scene uses the
    single-pass SBMP retrieval, so the primed terms do not exist and L3 is undefined
    rather than unknown.

    Args:
        bands_out: Band names, in the order they appear in ``x``.
        x: Image stack (C, H, W), reflectance x 2.
        mask: Valid-pixel mask (H, W) from :func:`valid_mask`.
        satellite: Instrument of the target pass.
        sza: Solar zenith angle of the target pass, degrees.
        vza: View zenith angle, degrees, for the air-mass factor.
        tile_date: Acquisition time of the target pass, ISO-8601.
        satellite_bg: Instrument of the reference pass.
        sza_bg: Solar zenith angle of the reference pass, degrees.
        tile_date_bg: Acquisition time of the reference pass, ISO-8601.

    Returns:
        Dictionary of per-scene statistics: ``radiance_{band}_*``, ``eta_{rung}_*``,
        ``epsilon_{rung}_*`` and ``sigma_ch4_{rung}_*`` in ppb.
    """
    from marss2l import shot_noise

    def radiance(band: str, background: bool) -> torch.Tensor:
        # The loader relabels Landsat to Sentinel-2 band names (bands_l8=True), so
        # bands_out is B11/B12 whatever the instrument -- unlike the exported raster
        # and the SRF tables, which keep B06/B07. Index with the S2 name here and
        # let band_irradiance do the translation where it is actually needed.
        reflectance = x[bands_out.index(band + ("_bg" if background else ""))] / 2.0
        return torch.as_tensor(
            shot_noise.radiance_from_reflectance(
                reflectance.numpy(),
                satellite_bg if background else satellite,
                band,
                sza=sza_bg if background else sza,
                date_of_acquisition=tile_date_bg if background else tile_date,
            ),
            dtype=torch.float64,
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

    stats_item = {}
    for band, values in [(shot_noise.BAND_23, radiance_23), (shot_noise.BAND_16, radiance_16)]:
        stats_item.update(_summary(values[mask].float(), f"radiance_{band}"))

    ladder = shot_noise.eta_ladder(
        radiance_23.numpy(),
        radiance_16.numpy(),
        *(r.numpy() if r is not None else None for r in reference),
        satellite=satellite,
        satellite_bg=satellite_bg or None,
    )

    for rung, eta in ladder.items():
        eta_tensor = torch.as_tensor(eta, dtype=torch.float64)
        stats_item.update(_summary(eta_tensor[mask].float(), f"eta_{rung}"))

        eta_valid = eta_tensor[mask].numpy()
        if eta_valid.size == 0:
            continue
        stats_item.update(
            _summary(
                torch.as_tensor(
                    shot_noise.epsilon(eta_valid, satellite, sza, vza, p=0.95), dtype=torch.float32
                ),
                f"epsilon_{rung}",
            )
        )
        # Evaluated at MBMP = 1, the plume-free value: ratio_IL normalises by the
        # scene mean, so that is what the retrieval reads where there is no plume.
        stats_item.update(
            _summary(
                torch.as_tensor(
                    shot_noise.sigma_delta_xch4(1.0, eta_valid, satellite, sza, vza),
                    dtype=torch.float32,
                ),
                f"sigma_ch4_{rung}",
            )
        )

    return stats_item


def compute_stats(
    bands_out: list,
    isplume: int,
    ch4: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    wind_vector: np.ndarray,
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

    stats_item.update(measured_noise_stats(bands_out, x=x, ch4=ch4, target=target, isplume=isplume))

    # mean, std, min, and max for CH4
    stats_item["ch4_mean"] = ch4.mean().item()
    stats_item["ch4_meanabs"] = ch4.abs().mean().item()
    stats_item["ch4_std"] = ch4.std().item()
    stats_item["ch4_min"] = ch4.min().item()
    stats_item["ch4_max"] = ch4.max().item()
    if isplume == 1:
        stats_item["ch4_mean_plume"] = ch4[target == 1].mean().item()
        stats_item["ch4_meanabs_plume"] = ch4[target == 1].abs().mean().item()
        stats_item["ch4_std_plume"] = ch4[target == 1].std().item()
        stats_item["ch4_min_plume"] = ch4[target == 1].min().item()
        stats_item["ch4_max_plume"] = ch4[target == 1].max().item()
        stats_item["ch4_mean_noplume"] = ch4[target == 0].mean().item()
        stats_item["ch4_meanabs_noplume"] = ch4[target == 0].abs().mean().item()
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
                resolution=(10, 10),
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
            # Compute mean, std, min, and max.
            # The /2 undoes the loader's reflectance x 2, so it belongs to the
            # spectral bands only. MBMP is a dimensionless ratio: the factor
            # cancels in its construction, so it never carried it and must not have
            # it removed. Halving it is what made MBMP_mean read ~0.5 where the
            # retrieval reads ~1 -- the values published before this fix are half
            # their true size.
            xband = x[bidx] if b == "MBMP" else x[bidx] / 2
            stats_item[f"{b}_mean"] = xband.mean().item()
            stats_item[f"{b}_meanabs"] = xband.abs().mean().item()
            stats_item[f"{b}_std"] = xband.std().item()
            stats_item[f"{b}_min"] = xband.min().item()
            stats_item[f"{b}_max"] = xband.max().item()

            if isplume == 1:
                # Compute mean, std, min, and max inside and outside of the plume
                stats_item[f"{b}_mean_plume"] = xband[target == 1].mean().item()
                stats_item[f"{b}_meanabs_plume"] = xband[target == 1].abs().mean().item()
                stats_item[f"{b}_std_plume"] = xband[target == 1].std().item()
                stats_item[f"{b}_min_plume"] = xband[target == 1].min().item()
                stats_item[f"{b}_max_plume"] = xband[target == 1].max().item()
                stats_item[f"{b}_mean_noplume"] = xband[target == 0].mean().item()
                stats_item[f"{b}_meanabs_noplume"] = xband[target == 0].abs().mean().item()
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
    smoke_test: bool = False,
    flush_every: int = 2000,
    dataset_name: Optional[str] = None,
) -> None:
    """Sweep the dataset and write one row of statistics per image.

    Args:
        csv_path: CSV with the image metadata.
        batch_size: Batch size for the data loader.
        num_workers: Worker processes for the data loader.
        output_file: Where to write. Defaults to ``stats_dataset[_<split>].csv``
            beside the input CSV, or ``stats_dataset_smoketest.csv`` locally under
            ``--smoke-test``.
        max_iter: Stop after this many images. Not stratified -- it simply stops;
            use ``--smoke-test`` for a balanced sample.
        path_prepend_data: Prefix for the data paths. Needed for a HuggingFace copy.
        split: Named split to sweep -- ``train_2023``, ``val_2023``, ``test_2023``,
            ``no split``. Omit to read every image.
        smoke_test: Sweep 20 images, 10 with plumes and 10 without, with a fixed
            seed. What makes the edit-run-look loop bearable on a dataset this size.
        flush_every: Write the rows gathered so far every this many images. A full
            split takes hours; this makes progress visible and survivable.
        dataset_name: Written to every row as a ``dataset`` column. What lets two
            sweeps be concatenated at plot time and still be told apart -- the
            CloudSEN12 scenes are worldwide and their countries would otherwise
            scatter across the MARS-S2L case studies (and mostly into "Rest"),
            when what the figures want is one box for the whole corpus. Omit for a
            single-dataset sweep, where the column adds nothing.
    """
    # spawn only where it is needed. It is required to share CUDA tensors, but it
    # also pickles the dataset for every worker, and the file logger the dataset
    # carries holds an open handle that cannot be pickled -- so on a CPU sweep spawn
    # forces num_workers to 0, and a full split would take most of a day. This sweep
    # is CPU-only, where fork is both safe and much faster.
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
        smoke_test=smoke_test,
        flush_every=flush_every,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    app()
