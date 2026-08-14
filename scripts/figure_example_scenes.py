"""Example scenes: the retrieval beside the floors that bound it.

One row per scene, four columns:

    RGB | radiance at 2.3 um | delta XCH4 | sigma(L3)

The point of the figure is that the last two columns are the same quantity --
a standard deviation of the retrieval in ppb, measured on the left and
propagated from photon statistics on the right -- so a reader can see directly
how much of what the retrieval reads its own photon floor accounts for, and how
both vary with the surface.

Rows are chosen to span regions and noise levels: the scenes are binned by their
own ``epsilon(L3)`` and one is drawn from each bin, preferring regions not yet
used, so that the sample is diverse by construction rather than by eye. Only
plume-free scenes are drawn, so that the measured noise quoted beside each row is
taken over the same pixels the panels show.

Rasters are read as ``GeoTensor`` and drawn with ``georeader.plot``; colour maps
and the ppb range follow ``notebooks/examples/download_and_inference.ipynb``.

Run::

    python -m scripts.figure_example_scenes figure \\
        --stats-csv <stats csv> --images-csv <images csv> \\
        --rows 10 --output-path <dir>/example_scenes.png
"""

import os
from typing import Optional

import cyclopts
import matplotlib as mpl
import numpy as np
import pandas as pd
from shapely import wkt

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from georeader import plot  # noqa: E402
from georeader.plot import add_shape_to_plot  # noqa: E402
from georeader.geotensor import GeoTensor  # noqa: E402

from marss2l import shot_noise  # noqa: E402
from marss2l.loaders import BANDS_S2_IN_L8  # noqa: E402
from marss2l.mars_sentinel2 import wind as wind_plot  # noqa: E402
from marss2l.mars_sentinel2.transmittance_to_ch4 import (  # noqa: E402
    TransmittanceCH4InterpolationFromDict,
    compute_xch4_retrieval,
)
from marss2l.utils import fs_from_path, pathjoin  # noqa: E402

app = cyclopts.App()

#: Following the example notebooks: reflectance x 10,000 over this is a display
#: RGB, and the retrieval is shown on plasma over 0-1500 ppb.
RGB_SCALE = 4_500.0
PPB_CMAP = "plasma"
RADIANCE_CMAP = "viridis"

#: The floors are a few hundred ppb where the retrieval's excursions are a few
#: thousand, so a single scale for all three ppb columns leaves the two floor
#: panels flat. Each column gets its own range, and its own colour bar, and the
#: labels carry the numbers that make them comparable.
DEFAULT_VMAX = {"ch4": 1_500.0, "sigma": 500.0}

#: Pixels below this many propagated standard deviations are not distinguishable
#: from photon noise, and the detection column hides them.
SIGNIFICANCE = 1.96

COLUMN_TITLES = [
    "RGB",
    r"$L_{23}$  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
    r"$\Delta$XCH$_4$  [ppb]",
    r"$\sigma(\Delta$XCH$_4)$ at $L_3$  [ppb]",
]
DETECTION_TITLE = rf"$\Delta$XCH$_4$ above ${SIGNIFICANCE}\,\sigma(L_3)$  [ppb]"


def select_scenes(
    scenes: pd.DataFrame, rows: int, seed: int = 0, plumes: bool = False
) -> pd.DataFrame:
    """A diverse sample: one scene per noise level, spread over regions.

    Bins the scenes by their own ``epsilon(L3)`` into as many quantile bins as
    there are rows and takes one from each, preferring a region that has not
    appeared yet.

    **Plume-free scenes only.** The measured noise quoted beside each row is
    taken over the plume-free pixels of the scene, which for a scene with a plume
    is a different set of pixels from the one the panels show -- so a figure
    mixing the two would put two different statistics under one heading. Every
    row here is a scene where the two coincide.

    Args:
        scenes: Output of ``figure_regional.load_scenes``, with ``case_study``.
        rows: How many scenes to return.
        plumes: Draw scenes that contain a plume instead of plume-free ones.
        seed: Unused now that each bin contributes its representative scene
            rather than a random member; kept so the caller's flag still works.

    Returns:
        The chosen rows, ordered by ``epsilon(L3)``.
    """
    frame = scenes.dropna(subset=["epsilon_L3_mean", "sigma_ch4_L3_mean", "measured"]).copy()
    if plumes:
        # A plume of a dozen pixels illustrates nothing about a detection limit.
        frame = frame[(frame.isplume == 1) & (frame.npixelsplume >= 30)]
    else:
        frame = frame[frame.isplume != 1]

    # A scene half covered by no-data or cloud makes a poor illustration: the
    # panels are then mostly blank and the eye reads the mask, not the retrieval.
    if {"npixelsvalid", "npixels"}.issubset(frame.columns):
        frame = frame[frame.npixelsvalid / frame.npixels > 0.98]

    # Binned on epsilon, which is a monotone function of the same eta as the
    # floor the figure plots -- same partition, and the rows do not move when the
    # columns do.
    frame["bin"] = pd.qcut(frame.epsilon_L3_mean, rows, labels=False, duplicates="drop")

    chosen, used = [], set()
    for value in sorted(frame.bin.unique()):
        candidates = frame[frame.bin == value]
        fresh = candidates[~candidates.case_study.isin(used)]
        candidates = fresh if len(fresh) else candidates

        # The scene nearest the bin's median floor, so each row is representative
        # of its noise level rather than a random member of it.
        target = candidates.epsilon_L3_mean.median()
        pick = candidates.loc[(candidates.epsilon_L3_mean - target).abs().idxmin()]
        used.add(pick.case_study)
        chosen.append(pick)

    return pd.DataFrame(chosen).sort_values("epsilon_L3_mean")


def scene_rasters(row: pd.Series, fs=None) -> dict:
    """Read one scene and build the five rasters, as GeoTensors.

    The file holds both passes stacked, six bands each, in
    :data:`BANDS_S2_IN_L8` order whatever the instrument. Values are ToA
    reflectance x 10,000.

    Args:
        row: Scene metadata -- paths, satellite, angles, dates.
        fs: Filesystem for the image paths.

    Returns:
        ``rgb``, ``radiance``, ``ch4``, ``sigma``, ``detected``.
    """
    nbands = len(BANDS_S2_IN_L8)
    b11, b12 = BANDS_S2_IN_L8.index("B11"), BANDS_S2_IN_L8.index("B12")

    image = GeoTensor.load_file(row.s2path, fs=fs)
    cloudmask = GeoTensor.load_file(row.cloudmaskpath, fs=fs).values
    if cloudmask.ndim == 3:
        cloudmask = cloudmask[0]

    values = image.values.astype(np.float64)
    target, background = values[:nbands], values[nbands:]
    valid = (cloudmask == 0) & (target != 0).all(axis=0) & (background != 0).all(axis=0)

    ch4 = compute_xch4_retrieval(
        target,
        background,
        offshore=bool(row.offshore),
        satellite=row.satellite,
        sza=float(row.sza),
        vza=float(row.vza),
        b11_index=b11,
        b12_index=b12,
        validmask=valid,
        transmittance_interpolator=TransmittanceCH4InterpolationFromDict(),
    )

    def radiance(
        index: int, band: str, pass_values: np.ndarray, background_pass: bool
    ) -> np.ndarray:
        return shot_noise.radiance_from_reflectance(
            pass_values[index] / 10_000.0,
            row.satellite_bg if background_pass else row.satellite,
            band,
            sza=float(row.sza_bg if background_pass else row.sza),
            date_of_acquisition=row.tile_date_bg if background_pass else row.tile_date,
        )

    radiance_23 = radiance(b12, "B12", target, False)
    radiance_16 = radiance(b11, "B11", target, False)
    reference = (radiance(b12, "B12", background, True), radiance(b11, "B11", background, True))

    ladder = shot_noise.eta_ladder(
        radiance_23,
        radiance_16,
        *reference,
        satellite=row.satellite,
        satellite_bg=row.satellite_bg or None,
    )
    eta = ladder["L3"]
    sigma = shot_noise.sigma_delta_xch4(1.0, eta, row.satellite, float(row.sza), float(row.vza))

    def masked(array: np.ndarray) -> GeoTensor:
        """Invalid pixels are not measurements; leave them blank rather than 0."""
        return GeoTensor(
            np.where(valid, array, np.nan),
            transform=image.transform,
            crs=image.crs,
            fill_value_default=np.nan,
        )

    # What survives a per-pixel significance test against that scene's own floor:
    # the retrieval, with everything photon noise could plausibly explain removed.
    detected = np.where(ch4 >= SIGNIFICANCE * sigma, ch4, np.nan)

    rgb = np.clip(target[[2, 1, 0]] / RGB_SCALE, 0, 1)
    return {
        "rgb": GeoTensor(rgb, transform=image.transform, crs=image.crs, fill_value_default=np.nan),
        "radiance": masked(radiance_23),
        "ch4": masked(ch4),
        "sigma": masked(sigma),
        "detected": masked(detected),
    }


PATH_COLUMNS = [
    "id_loc_image",
    "s2path",
    "cloudmaskpath",
    "satellite",
    "sza",
    "vza",
    "tile_date",
    "satellite_bg",
    "sza_bg",
    "tile_date_bg",
    "offshore",
    "location_name",
    "wind_u",
    "wind_v",
    "plume",
]


def corpus_with_paths(
    stats_csv: str,
    images_csv: str,
    permian_shapefile: Optional[str] = None,
    path_prepend_data: Optional[str] = None,
) -> pd.DataFrame:
    """One corpus, selected as the other figures select it, plus its image paths."""
    from scripts.figure_regional import apply_permian_labels, load_scenes

    scenes = load_scenes(stats_csv, images_csv)
    if permian_shapefile is not None:
        scenes = apply_permian_labels(scenes, permian_shapefile)

    paths = pd.read_csv(images_csv, usecols=PATH_COLUMNS)
    scenes = scenes.drop(
        columns=[c for c in paths.columns if c in scenes.columns and c != "id_loc_image"]
    ).merge(paths, on="id_loc_image", how="inner")

    if path_prepend_data is not None:
        for field in ["s2path", "cloudmaskpath"]:
            scenes[field] = scenes[field].apply(lambda path: pathjoin(path_prepend_data, path))

    return scenes


@app.command
def figure(
    stats_csv: str,
    images_csv: str,
    output_path: str = "example_scenes.png",
    rows: int = 10,
    permian_shapefile: Optional[str] = None,
    path_prepend_data: Optional[str] = None,
    extra_stats_csv: Optional[str] = None,
    extra_images_csv: Optional[str] = None,
    only: Optional[list[str]] = None,
    plumes: bool = False,
    ch4_vmax: float = DEFAULT_VMAX["ch4"],
    sigma_vmax: float = DEFAULT_VMAX["sigma"],
    seed: int = 0,
) -> None:
    """Draw the example-scene figure.

    Args:
        stats_csv: Output of ``stats_dataset.py``, for the per-scene statistics
            the rows are chosen by.
        images_csv: Image metadata CSV, for the paths and the geometry.
        output_path: Where to write the figure.
        rows: Scenes to draw, one per row.
        permian_shapefile: Optional basin polygon, so the row labels agree with
            the rest of the paper.
        path_prepend_data: Prefix for the image paths, which the export stores
            relative to the dataset root.
        extra_stats_csv: A second corpus to draw rows from as well. Worth adding:
            its scenes are the dark ones, where the floor is highest and most
            visibly structured.
        extra_images_csv: Image metadata CSV of that second corpus.
        only: ``id_loc_image`` values to draw, in order, instead of sampling.
            How a selection made by eye from a longer draft is pinned for the
            paper, so the figure is reproducible from the command line alone.
        plumes: Draw scenes containing a plume, and add a fifth column showing
            the retrieval with everything below the per-pixel detection limit
            masked out. The measured noise in the label is then taken over the
            scene's plume-free pixels, which the label says.
        ch4_vmax: Upper end of the retrieval colour scale, in ppb.
        sigma_vmax: Upper end of the floor's scale. Lower than the retrieval's,
            since the floor is a few hundred ppb where the retrieval's excursions
            are a few thousand.
        seed: Sampling seed for the selection.
    """
    vmax = {"ch4": ch4_vmax, "sigma": sigma_vmax}
    scenes = corpus_with_paths(stats_csv, images_csv, permian_shapefile, path_prepend_data)
    if extra_stats_csv is not None:
        scenes = pd.concat(
            [scenes, corpus_with_paths(extra_stats_csv, extra_images_csv, permian_shapefile)],
            ignore_index=True,
        )

    if only:
        indexed = scenes.set_index(scenes.id_loc_image.astype(str))
        missing = [key for key in only if key not in indexed.index]
        if missing:
            raise KeyError(f"not in the selection: {missing}")
        chosen = indexed.loc[list(only)].reset_index(drop=True)
    else:
        chosen = select_scenes(scenes, rows, seed=seed, plumes=plumes)
    print(
        chosen[
            ["dataset", "case_study", "location_name", "satellite", "epsilon_L3_mean", "measured"]
        ].to_string()
    )

    columns = 5 if plumes else 4
    fig, axes = plt.subplots(len(chosen), columns, figsize=(3.35 * columns, 3.3 * len(chosen)))
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)

    for row_index, (_, row) in enumerate(chosen.iterrows()):
        # Per row: the two corpora live on different filesystems, and fs_from_path
        # caches, so this costs one lookup rather than one connection.
        rasters = scene_rasters(row, fs=fs_from_path(str(row.s2path)))
        # The radiance keeps a colour bar per row, since its range is the scene's
        # own and is worth reading; the three ppb columns share one scale across
        # the whole figure, so one colour bar on the top row serves all of them.
        panels = [
            (rasters["rgb"], {}),
            (rasters["radiance"], dict(cmap=RADIANCE_CMAP, add_colorbar_next_to=True)),
            (rasters["ch4"], dict(vmin=0, vmax=vmax["ch4"], cmap=PPB_CMAP)),
            (rasters["sigma"], dict(vmin=0, vmax=vmax["sigma"], cmap=PPB_CMAP)),
        ]
        if plumes:
            panels.append((rasters["detected"], dict(vmin=0, vmax=vmax["ch4"], cmap=PPB_CMAP)))
        for column, (raster, kwargs) in enumerate(panels):
            ax = axes[row_index, column]
            plot.show(raster, ax=ax, add_scalebar=(column == 0), **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                titles = COLUMN_TITLES + ([DETECTION_TITLE] if plumes else [])
                ax.set_title(titles[column], fontsize=11, loc="left", pad=10)

        # The wind is what a reader needs to tell a plume from a surface feature,
        # and the retrieval panel is where that judgement is made.
        wind_plot.add_wind_to_plot(
            [float(row.wind_u), float(row.wind_v)], ax=axes[row_index, 2], fontsize=9
        )

        # The validated plume, outlined and unfilled, on the two retrieval panels:
        # on the third it says which part of the image is methane, and on the
        # fifth whether what survived the significance test is that same part.
        if plumes and isinstance(row.plume, str):
            geometry = wkt.loads(row.plume)
            if not geometry.is_empty:
                for column in (2, 4):
                    ax = axes[row_index, column]
                    # A plume that runs past the frame would otherwise stretch the
                    # axes to fit it, leaving the panel padded and out of step
                    # with its row.
                    limits = (ax.get_xlim(), ax.get_ylim())
                    add_shape_to_plot(
                        shape=geometry,
                        ax=ax,
                        crs_plot=rasters["ch4"].crs,
                        crs_shape="EPSG:4326",
                        polygon_no_fill=True,
                        kwargs_geopandas_plot={"color": "red", "linewidth": 1.0},
                    )
                    ax.set_xlim(*limits[0])
                    ax.set_ylim(*limits[1])

        # Horizontal, in the left margin: a rotated label at this size is a
        # smear, and the row identity is the first thing a reader looks for.
        axes[row_index, 0].set_ylabel(
            f"{row.case_study}{'  ·  plume' if row.isplume == 1 else ''}\n"
            f"{row.satellite}   {str(row.tile_date)[:10]}\n"
            f"floor $L_3$ {row.sigma_ch4_L3_mean:.0f} ppb\n"
            f"measured {row.measured:.0f} ppb{' (plume-free px)' if row.isplume == 1 else ''}\n"
            f"({row.measured / row.sigma_ch4_L3_mean:.1f}$\\times$ floor)",
            fontsize=10,
            rotation=0,
            ha="right",
            va="center",
            labelpad=12,
        )

    # One colour bar per ppb column, along the bottom: the ranges differ, and a
    # bar under its own column is where a reader looks for it. Per-panel bars
    # would repeat the scale ten times and steal width from the panels carrying
    # them, leaving the rows visibly unequal.
    fig.tight_layout()
    for column, key in enumerate(["ch4", "sigma"] + (["ch4"] if plumes else []), start=2):
        mappable = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=vmax[key]), cmap=PPB_CMAP
        )
        bar = fig.colorbar(
            mappable,
            ax=axes[:, column].tolist(),
            orientation="horizontal",
            fraction=0.013,
            pad=0.012,
            aspect=22,
        )
        bar.set_label("ppb", fontsize=10)
        bar.ax.tick_params(labelsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    app()
