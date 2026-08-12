"""The two regional figures of the shot-noise paper, from the sweep CSV.

``floors_by_region.png`` (F1)
    The three floors per case study, faceted by satellite family. What the
    instrument can resolve, and where the multi-pass construction costs most.

``gap_by_region.png`` (F2)
    The retrieval's measured noise against its own photon-noise floor, per case
    study, both as standard deviations in ppb, with the reducible share beside it.
    The crux result: how much of today's error a better background estimate could
    remove.

``scenes_by_region.png`` (F8)
    What the scenes are like: how bright, and how uniform -- the mean and the
    standard deviation of the 2.3 um radiance within a scene. The context for
    reading the gap, since brightness sets the floor and the spread is the
    structure the background estimate has to predict.

Everything is a groupby on the CSV that ``stats_dataset.py`` writes. Nothing here
recomputes a raster, and a point is always **one scene** -- pixels within a scene
are strongly correlated, so a distribution over pixels would claim a precision the
data does not have.

Scene selection, per the epic: the named split, ``observability == "clear"``,
onshore only (offshore uses the single-pass SBMP retrieval, so its noise is not
comparable and L3 does not exist for it) and no night reference pass.

A second corpus can be added as one further case study -- CloudSEN12, whose
scenes are worldwide and are not oil-and-gas infrastructure, so it says what the
floors and the gap look like away from the producing regions the rest of the
axis is made of. It enters as its own row rather than by country: its countries
would otherwise scatter across the MARS-S2L case studies and mostly into "Rest",
mixing two corpora that were selected on entirely different criteria.

Run::

    python -m scripts.figure_regional figures --stats-csv <csv> --images-csv <csv> \\
        --extra-stats-csv <cloudsen12 csv> --extra-images-csv <cloudsen12 csv> \\
        --output-dir <dir>
"""

import os
from typing import Optional

import cyclopts
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from marss2l.dataframe_image_plumes import ORDER_CASE_STUDIES, _set_case_study  # noqa: E402

app = cyclopts.App()

#: Label for the second corpus, and the name of its row on the case-study axis.
DEFAULT_EXTRA_LABEL = "CloudSEN12"

#: The United States row of the axis is, in practice, one oil and gas basin.
PERMIAN_LABEL = "Permian basin"
UNITED_STATES = "United States of America"

#: The axis order, with the Permian label sitting where the United States row
#: would be. Only one of the two is ever present -- the relabelling replaces the
#: country rather than adding to it -- so one list serves both cases.
ORDER_CASE_STUDIES_EXT = [
    label
    for case in ORDER_CASE_STUDIES
    for label in ((case, PERMIAN_LABEL) if case == UNITED_STATES else (case,))
]

#: The rungs are nested and ordered, so they take a single hue light-to-dark
#: rather than three unrelated colours.
RUNG_COLOURS = {"L1": "#9dc3ee", "L2": "#5b95dd", "L3": "#1f5fae"}
RUNG_LABELS = {
    "L1": "L1 — background known exactly",
    "L2": "L2 — band ratio known exactly",
    "L3": "L3 — background from a reference image",
}
MEASURED = "#eb6834"
INK, INK_SOFT, GRID = "#14181f", "#58606c", "#dfe3e8"


def _read_meta(images_csv: str) -> pd.DataFrame:
    """Scene metadata for the selection and the axis.

    ``case_study`` is derived from the country when the CSV does not carry it --
    the CloudSEN12 export does not, and re-deriving it is one call to the same
    mapping the MARS-S2L export used.
    """
    columns = pd.read_csv(images_csv, nrows=0).columns
    wanted = ["id_loc_image", "observability", "sza_bg_source", "lon", "lat"]
    wanted.append("case_study" if "case_study" in columns else "country")
    if "country" not in wanted and "country" in columns:
        # Carried for the per-country stratification of the supplement, which
        # needs the country even when the export already resolved a case study.
        wanted.append("country")

    meta = pd.read_csv(images_csv, usecols=[c for c in wanted if c in columns])
    if "case_study" not in meta.columns:
        meta["case_study"] = meta["country"].apply(_set_case_study)

    return meta


def apply_permian_labels(scenes: pd.DataFrame, shapefile: str) -> pd.DataFrame:
    """Split the United States row into the Permian basin and everything else.

    The country is not the unit anyone means here: essentially all of the corpus's
    United States imagery is one oil and gas basin, so a row labelled with the
    country invites the reader to generalise it to a continent. Scenes inside the
    basin take its name; the remainder join ``Rest``, which is what that row is
    for -- places with too few scenes to stand on their own.

    Args:
        scenes: Output of :func:`load_scenes`, with ``lon`` and ``lat``.
        shapefile: Polygon of the basin. The EIA publishes one as
            ``PermianBasin_Boundary_Structural_Tectonic.zip`` (layer
            ``PermianBasin_Extent``), which is the geological province rather than
            the producing area, so it is the generous definition of the two.

    Returns:
        The frame, with ``case_study`` reassigned for United States scenes.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if "lon" not in scenes.columns:
        raise KeyError("the images CSV must carry lon/lat to place scenes in the basin")

    basin = gpd.read_file(shapefile).geometry.union_all()
    is_us = scenes.case_study == UNITED_STATES
    inside = pd.Series(
        [Point(x, y).within(basin) for x, y in zip(scenes.lon, scenes.lat)], index=scenes.index
    )

    scenes = scenes.copy()
    scenes.loc[is_us & inside, "case_study"] = PERMIAN_LABEL
    scenes.loc[is_us & ~inside, "case_study"] = "Rest"

    n_us = int(is_us.sum())
    if n_us:
        n_in = int((is_us & inside).sum())
        print(
            f"Permian: {n_in:,} of {n_us:,} United States scenes ({n_in / n_us:.1%}) "
            f"inside the basin; {n_us - n_in:,} moved to Rest"
        )

    return scenes


def load_scenes(stats_csv: str, images_csv: str, label: Optional[str] = None) -> pd.DataFrame:
    """Read the sweep and apply the scene selection the figures use.

    Args:
        stats_csv: Output of ``stats_dataset.py``.
        images_csv: The image metadata CSV, for ``observability`` and ``case_study``.
        label: Name to give every scene of this corpus on the case-study axis,
            overriding the per-country grouping. What makes a second dataset one
            row of the figures rather than a re-partition of the first.

    Returns:
        One row per usable scene, with the measured noise and the ratio to each rung.
    """
    stats = pd.read_csv(stats_csv, low_memory=False)
    scenes = stats.merge(_read_meta(images_csv), on="id_loc_image", how="left")

    scenes = scenes[
        (scenes.observability == "clear")
        & (~scenes.offshore.astype(bool))
        & (scenes.sza_bg_source != "night")
        & scenes.sigma_ch4_L3_mean.notna()
    ].copy()

    if label is not None:
        scenes["case_study"] = label
    if "dataset" not in scenes.columns:
        # Sweeps run before --dataset-name existed carry no such column.
        scenes["dataset"] = label or "MARS-S2L"

    # O-base: the retrieval's own noise, on pixels where it should read zero. A
    # corpus with no plumes at all -- CloudSEN12 -- has no _noplume column, since
    # the sweep only writes one for a scene that has a plume to exclude.
    without_plume = scenes.ch4_valid_std
    with_plume = scenes.get("ch4_valid_noplume_std", np.nan)
    scenes["measured"] = np.where(scenes.isplume == 1, with_plume, without_plume)
    scenes = scenes[scenes.measured.notna() & (scenes.measured > 0)]

    scenes["family"] = np.where(scenes.satellite.str.startswith("S2"), "Sentinel-2", "Landsat")
    for rung in RUNG_COLOURS:
        scenes[f"ratio_{rung}"] = scenes.measured / scenes[f"sigma_ch4_{rung}_mean"]
    # Exact, because both sides are standard deviations of the same quantity.
    scenes["reducible"] = 1 - (scenes.sigma_ch4_L3_mean / scenes.measured) ** 2

    return scenes


def case_study_order(scenes: pd.DataFrame, min_scenes: int = 1) -> list:
    """The y axis: the MARS-S2L case studies in their usual order, extras last.

    A corpus added under its own label is not part of ``ORDER_CASE_STUDIES``, so
    it goes after it -- below "Rest", where a reader has finished with the
    producing regions.
    """
    present = scenes.case_study.value_counts()
    present = set(present[present >= min_scenes].index)
    known = [c for c in ORDER_CASE_STUDIES_EXT if c in present]
    return known + sorted(present - set(known))


def _style(ax, *, xlabel: str = "", title: str = "") -> None:
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, width=0.6)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def _boxes(ax, data, positions, colour, width=0.24):
    """Thin horizontal boxes, no outlier confetti, median emphasised."""
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        vert=False,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.3),
        whiskerprops=dict(color=colour, linewidth=0.9),
        capprops=dict(color=colour, linewidth=0.9),
        boxprops=dict(facecolor=colour, edgecolor=colour, linewidth=0),
    )
    return bp


def figure_floors(scenes: pd.DataFrame, path: str) -> None:
    """F1: the three floors per case study, faceted by satellite family."""
    order = case_study_order(scenes)
    families = ["Sentinel-2", "Landsat"]

    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 0.42 * len(order) + 2.4), sharex=True, sharey=True
    )
    fig.patch.set_facecolor("white")

    for ax, family in zip(axes, families, strict=True):
        subset = scenes[scenes.family == family]
        for offset, rung in zip([0.27, 0.0, -0.27], ["L1", "L2", "L3"], strict=True):
            data, positions = [], []
            for i, case in enumerate(order):
                values = subset.loc[subset.case_study == case, f"epsilon_{rung}_mean"].dropna()
                if len(values) >= 5:
                    data.append(values.values)
                    positions.append(i + offset)
            if data:
                _boxes(ax, data, positions, RUNG_COLOURS[rung])

        # A row can be empty in one facet and not the other -- CloudSEN12 is
        # Sentinel-2 only. Say so, rather than leaving a blank the reader has to
        # decide between "no data" and "too small to see".
        for i, case in enumerate(order):
            if (subset.case_study == case).sum() < 5:
                ax.annotate(
                    "no scenes",
                    xy=(0.015, i),
                    xycoords=("axes fraction", "data"),
                    va="center",
                    fontsize=7,
                    style="italic",
                    color=INK_SOFT,
                )

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8, color=INK)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.invert_yaxis()  # keep ORDER_CASE_STUDIES reading top to bottom
        ax.set_xscale("log")
        _style(
            ax,
            xlabel=r"$\epsilon$ at $p=0.95$  [ppb]",
            title=f"{'a' if family == 'Sentinel-2' else 'b'}  {family}  (n = {len(subset):,})",
        )

    fig.legend(
        handles=[Patch(facecolor=RUNG_COLOURS[r], label=RUNG_LABELS[r]) for r in RUNG_COLOURS],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SOFT,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def figure_scenes(scenes: pd.DataFrame, path: str) -> None:
    """F8: what the scenes themselves are like -- how bright, and how uniform.

    The context for reading F2. Brightness sets the floor, through the square
    root of the radiance in the SNR rescaling; the spread of that radiance within
    a scene is the simplest measure of the structure the background estimate has
    to predict, and it is what the gap above the floor is made of. Panel a is
    coloured like the floor in F2 and panel b like the measured noise, because
    that is the quantity each one explains.

    Both panels are the 2.3 um radiance, per scene: its mean over valid pixels,
    and its standard deviation over the same pixels. They are the two moments the
    sweep already writes, so this figure costs nothing beyond the CSVs behind F1
    and F2 -- the same scenes, the same selection, the same unit.

    Args:
        scenes: Output of :func:`load_scenes`.
        path: Where to write the figure.
    """
    order = case_study_order(scenes, min_scenes=5)

    # Shared x as well as y: the two panels are the same quantity in the same
    # unit, so putting them on one scale shows directly that the spread within a
    # scene is an order of magnitude below the level -- which a reader cannot see
    # from two independently scaled axes.
    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 0.42 * len(order) + 2.2), sharey=True, sharex=True
    )
    fig.patch.set_facecolor("white")

    panels = [
        (
            "radiance_B12_mean",
            RUNG_COLOURS["L3"],
            r"mean radiance at 2.3 $\mu$m  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
            "a  How bright the scene is",
        ),
        (
            "radiance_B12_std",
            MEASURED,
            r"std. dev. within the scene  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
            "b  How uniform it is",
        ),
    ]

    for ax, (column, colour, xlabel, title) in zip(axes, panels, strict=True):
        data = [scenes.loc[scenes.case_study == c, column].dropna().values for c in order]
        _boxes(ax, data, list(range(len(order))), colour, width=0.42)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8, color=INK)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.invert_yaxis()  # keep ORDER_CASE_STUDIES reading top to bottom
        ax.set_xscale("log")
        _style(ax, xlabel=xlabel, title=title)

    # The spread relative to the brightness, which is the two panels divided and
    # the quantity that actually tracks the gap: an absolute spread is larger over
    # bright ground for no other reason than that the ground is bright.
    relative = (
        (scenes.radiance_B12_std / scenes.radiance_B12_mean)
        .groupby(scenes.case_study)
        .median()
        .reindex(order)
    )
    for i, value in enumerate(relative.values):
        axes[1].annotate(
            f"{value:.2f}",
            xy=(1.0, i),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK_SOFT,
        )
    axes[1].annotate(
        "std/mean",
        xy=(1.0, -0.7),
        xycoords=("axes fraction", "data"),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=7.5,
        style="italic",
        color=INK_SOFT,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def figure_gap(scenes: pd.DataFrame, path: str) -> None:
    """F2: measured noise against the L3 floor, per case study, with the share."""
    order = case_study_order(scenes, min_scenes=5)

    fig, (ax, ax_share) = plt.subplots(
        1,
        2,
        figsize=(10.4, 0.42 * len(order) + 2.2),
        gridspec_kw={"width_ratios": [3.1, 1]},
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    for offset, column, colour in [
        (0.19, "sigma_ch4_L3_mean", RUNG_COLOURS["L3"]),
        (-0.19, "measured", MEASURED),
    ]:
        data = [scenes.loc[scenes.case_study == c, column].dropna().values for c in order]
        _boxes(ax, data, [i + offset for i in range(len(order))], colour, width=0.3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8, color=INK)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.invert_yaxis()  # keep ORDER_CASE_STUDIES reading top to bottom
    ax.set_xscale("log")
    _style(
        ax,
        xlabel=r"$\sigma(\Delta \mathrm{XCH}_4)$  [ppb]",
        title="a  What the retrieval reads, against its photon-noise floor",
    )
    ax.legend(
        handles=[
            Patch(facecolor=MEASURED, label="measured, plume-free pixels"),
            Patch(facecolor=RUNG_COLOURS["L3"], label="floor L3, propagated"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.06),
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SOFT,
    )

    # The single number a reader quotes: the share of variance that is not photons.
    share = scenes.groupby("case_study").reducible.median().reindex(order)
    ax_share.barh(range(len(order)), share.values, height=0.5, color=MEASURED, zorder=3)
    for i, value in enumerate(share.values):
        ax_share.annotate(
            f"{value:.0%}",
            xy=(value, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax_share.set_xlim(0, 1.18)
    ax_share.set_xticks([0, 0.5, 1.0])
    ax_share.set_xticklabels(["0", "50%", "100%"])
    _style(ax_share, xlabel="share of variance that is\nnot photon noise", title="b  Reducible")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _labels_with_n(scenes: pd.DataFrame, order: list) -> list:
    """Row labels carrying their sample size, for panels where n varies wildly."""
    counts = scenes.case_study.value_counts()
    return [f"{case}  (n = {counts.get(case, 0):,})" for case in order]


#: Above this many scenes a jitter cloud is a black smear that hides the box it
#: is meant to qualify, and the box is trustworthy anyway. Draw a sample.
MAX_JITTER = 250


def _jittered(ax, values, position: float, colour: str, rng, width: float = 0.16) -> None:
    """The scenes behind a box, so a row of eighteen cannot pass for a distribution."""
    if len(values) == 0:
        return
    if len(values) > MAX_JITTER:
        values = rng.choice(values, MAX_JITTER, replace=False)
    ax.scatter(
        values,
        position + rng.uniform(-width, width, size=len(values)),
        s=4,
        color=colour,
        alpha=0.4,
        linewidths=0,
        zorder=4,
        rasterized=True,
    )


def _robust_limits(arrays: list) -> tuple:
    """x limits that keep every box legible while showing nearly all the jitter."""
    pooled = np.concatenate([np.asarray(a) for a in arrays if len(a)])
    pooled = pooled[pooled > 0]
    return np.percentile(pooled, 0.5) / 1.6, np.percentile(pooled, 99.5) * 1.6


def figure_by_country(
    scenes: pd.DataFrame,
    path: str,
    panels: list,
    order: list,
    title_suffix: str = "",
    sharex: bool = False,
    log: bool = True,
) -> None:
    """A per-case-study figure with the individual scenes drawn behind the boxes.

    For the supplement, where a row can hold eighteen scenes or nine thousand. The
    sample size goes in the label and the scenes themselves are drawn, so a box
    built on a handful of images cannot be read as if it were built on thousands.
    Rows above :data:`MAX_JITTER` scenes show a random sample of that many, since
    beyond it the cloud hides the box it is there to qualify.

    Args:
        scenes: Output of :func:`load_scenes`, with ``case_study`` set to the row
            each scene belongs to.
        path: Where to write the figure.
        panels: One ``(series, xlabel, title)`` per panel, where ``series`` is a
            list of ``(column, colour, label)`` drawn together on that panel.
        order: Rows, top to bottom.
        title_suffix: Appended above the figure.
        sharex: Put the panels on one x scale. For panels in the same unit.
        log: Logarithmic x axis.
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.6 * len(panels), 0.52 * len(order) + 2.4),
        sharey=True,
        sharex=sharex,
    )

    fig.patch.set_facecolor("white")

    pooled: list = []
    for ax, panel in zip(np.atleast_1d(axes), panels, strict=True):
        # A panel may override the figure's x scale: a share bounded by 0 and 1
        # has no business on a logarithmic axis, where its whiskers run off the
        # left edge and every box reads as full width.
        series, xlabel, title, *rest = panel
        use_log = rest[0] if rest else log
        xlim = rest[1] if len(rest) > 1 else None

        offsets = np.linspace(0.19, -0.19, len(series)) if len(series) > 1 else [0.0]
        width = 0.3 if len(series) > 1 else 0.44
        drawn = []
        for (column, colour, _), offset in zip(series, offsets, strict=True):
            data = [scenes.loc[scenes.case_study == c, column].dropna().values for c in order]
            drawn.extend(data)
            _boxes(ax, data, [i + offset for i in range(len(order))], colour, width=width)
            for position, values in enumerate(data):
                _jittered(ax, values, position + offset, INK, rng, width=width * 0.42)

        # The boxes hide their outliers but the jitter does not, and a handful of
        # extreme scenes would otherwise squeeze every box into a decade. Under
        # sharex the limit has to come from every panel at once, or the last panel
        # drawn silently clips the first.
        if xlim is not None:
            ax.set_xlim(*xlim)
        elif use_log:
            pooled.extend(v for v in drawn if len(v))
            if not sharex:
                ax.set_xlim(*_robust_limits(pooled))
                pooled = []

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(_labels_with_n(scenes, order), fontsize=8, color=INK)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.invert_yaxis()
        if use_log:
            ax.set_xscale("log")
        _style(ax, xlabel=xlabel, title=title)
        if len(series) > 1:
            ax.legend(
                handles=[Patch(facecolor=c, label=lab) for _, c, lab in series],
                loc="upper left",
                bbox_to_anchor=(0.0, -0.08),
                ncol=len(series),
                frameon=False,
                fontsize=8,
                labelcolor=INK_SOFT,
            )

    if sharex and pooled:
        np.atleast_1d(axes)[0].set_xlim(*_robust_limits(pooled))

    if title_suffix:
        fig.suptitle(title_suffix, x=0.0, ha="left", fontsize=9, color=INK_SOFT, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


#: Corpus identity, for the figure that compares the two. Deliberately not the
#: rung blue or the measured orange, which carry a different meaning everywhere
#: else in the paper.
CORPUS_COLOURS = {"MARS-S2L": "#1f7a6f", "CloudSEN12": "#b5537f"}


def figure_corpora(main: pd.DataFrame, extra: pd.DataFrame, path: str) -> None:
    """S3: the same regions in both corpora, with no floor in sight.

    The supplement's other two figures each show one corpus. This one puts them
    side by side on the three quantities that do not depend on the noise model at
    all -- what the retrieval reads, how bright the scene is, how variegated it
    is -- so that the reader can see which differences between the corpora are
    about the surface and which are about what the images were taken of.

    The comparison is only meaningful within a row: a region samples the same
    ground in both corpora, up to where in it the images fall.

    Args:
        main: Scenes of the first corpus, with per-region ``case_study``.
        extra: Scenes of the second, stratified the same way.
        path: Where to write the figure.
    """
    scenes = pd.concat([main, extra], ignore_index=True)
    corpora = [c for c in CORPUS_COLOURS if c in set(scenes.dataset)]
    order = [c for c in ORDER_CASE_STUDIES_EXT if c in set(scenes.case_study)]

    counts = scenes.groupby(["case_study", "dataset"]).size().unstack(fill_value=0)
    labels = [
        f"{case}  ({' | '.join(f'{counts.get(c, {}).get(case, 0):,}' for c in corpora)})"
        for case in order
    ]

    panels = [
        (
            "measured",
            r"$\sigma(\Delta \mathrm{XCH}_4)$ measured  [ppb]",
            "a  What the retrieval reads",
        ),
        (
            "radiance_B12_mean",
            r"mean radiance at 2.3 $\mu$m  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
            "b  How bright the scene is",
        ),
        (
            "radiance_B12_std",
            r"std. dev. within the scene  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
            "c  How uniform it is",
        ),
    ]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 0.52 * len(order) + 2.6), sharey=True)
    fig.patch.set_facecolor("white")

    radiance_pool: list = []
    for ax, (column, xlabel, title) in zip(axes, panels, strict=True):
        pooled = []
        for corpus, offset in zip(corpora, np.linspace(0.18, -0.18, len(corpora)), strict=True):
            subset = scenes[scenes.dataset == corpus]
            data = [subset.loc[subset.case_study == c, column].dropna().values for c in order]
            pooled.extend(data)
            _boxes(ax, data, [i + offset for i in range(len(order))], CORPUS_COLOURS[corpus], 0.3)
            for position, values in enumerate(data):
                _jittered(ax, values, position + offset, INK, rng, width=0.12)

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels, fontsize=8, color=INK)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(*_robust_limits(pooled))
        _style(ax, xlabel=xlabel, title=title)
        if column.startswith("radiance"):
            radiance_pool.extend(pooled)

    # The two radiance panels are the same quantity in the same unit; one scale
    # across both is what shows that the spread sits an order of magnitude below
    # the level. The ppb panel keeps its own.
    for ax in axes[1:]:
        ax.set_xlim(*_robust_limits(radiance_pool))

    fig.legend(
        handles=[Patch(facecolor=CORPUS_COLOURS[c], label=c) for c in corpora],
        loc="lower center",
        ncol=len(corpora),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SOFT,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "n per region as (" + " | ".join(corpora) + ")",
        x=0.0,
        ha="left",
        fontsize=8.5,
        color=INK_SOFT,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def supplementary_figures(scenes: pd.DataFrame, output_dir: str, label: str) -> None:
    """The supplement: one corpus, stratified by the case studies of the other.

    CloudSEN12 sits a factor of two above every producing region in the main
    figures, and the obvious question is whether that is a different regime or a
    different mixture of places. Stratifying it by the case studies of the other
    corpus answers it against rows that already have a value to compare with.
    """
    order = [c for c in ORDER_CASE_STUDIES_EXT if c in set(scenes.case_study)]
    subtitle = f"{label}, stratified by the case studies of the other corpus"

    figure_by_country(
        scenes,
        os.path.join(output_dir, "cloudsen12_gap_by_country.png"),
        panels=[
            (
                [
                    ("measured", MEASURED, "measured, plume-free pixels"),
                    ("sigma_ch4_L3_mean", RUNG_COLOURS["L3"], "floor L3, propagated"),
                ],
                r"$\sigma(\Delta \mathrm{XCH}_4)$  [ppb]",
                "a  What the retrieval reads, against its photon-noise floor",
            ),
            (
                [("reducible", MEASURED, "reducible share")],
                "share of variance that is not photon noise",
                "b  Reducible",
                False,
                # Negative means a scene read below its own floor, which is a
                # result rather than an error; one decade of it is enough to see.
                (-1.0, 1.03),
            ),
        ],
        order=order,
        title_suffix=subtitle,
    )

    figure_by_country(
        scenes,
        os.path.join(output_dir, "cloudsen12_scenes_by_country.png"),
        panels=[
            (
                [("radiance_B12_mean", RUNG_COLOURS["L3"], "mean")],
                r"mean radiance at 2.3 $\mu$m  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
                "a  How bright the scene is",
            ),
            (
                [("radiance_B12_std", MEASURED, "std")],
                r"std. dev. within the scene  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
                "b  How uniform it is",
            ),
        ],
        order=order,
        title_suffix=subtitle,
        sharex=True,
    )


@app.command
def figures(
    stats_csv: str,
    images_csv: str,
    output_dir: str = ".",
    summary_csv: Optional[str] = None,
    extra_stats_csv: Optional[str] = None,
    extra_images_csv: Optional[str] = None,
    extra_label: str = DEFAULT_EXTRA_LABEL,
    permian_shapefile: Optional[str] = None,
    supplement: bool = True,
) -> None:
    """Draw F1, F2 and F8 from a sweep, and the supplement from the second corpus.

    Args:
        stats_csv: Output of ``stats_dataset.py``.
        images_csv: Image metadata CSV, for observability and case study.
        output_dir: Where to write the figures.
        summary_csv: Optional path to write the per-region table behind them.
        extra_stats_csv: Sweep of a second corpus, appended as one further case
            study. Requires ``extra_images_csv``.
        extra_images_csv: Image metadata CSV of that second corpus.
        extra_label: Name of its row on the case-study axis.
        permian_shapefile: Polygon of the Permian basin. Given one, the United
            States row becomes the basin and the scenes outside it join ``Rest``
            -- see :func:`apply_permian_labels`. Applied to **both** corpora, so
            that one axis label means one thing across every figure; in a
            worldwide corpus that leaves a small basin row and moves the rest of
            its United States scenes to ``Rest``, which is the honest reading of
            what those scenes are.
        supplement: Also draw the second corpus stratified by the case studies of
            the first, which is what says whether it differs in regime or only in
            composition.
    """
    scenes = load_scenes(stats_csv, images_csv)
    if permian_shapefile is not None:
        scenes = apply_permian_labels(scenes, permian_shapefile)

    if (extra_stats_csv is None) != (extra_images_csv is None):
        raise ValueError("--extra-stats-csv and --extra-images-csv go together")
    if extra_stats_csv is not None:
        extra = load_scenes(extra_stats_csv, extra_images_csv, label=extra_label)
        print(f"{len(extra):,} {extra_label} scenes after selection")
        if supplement:
            by_country = extra.copy()
            by_country["case_study"] = by_country["country"].apply(_set_case_study)
            if permian_shapefile is not None:
                by_country = apply_permian_labels(by_country, permian_shapefile)
            supplementary_figures(by_country, output_dir, extra_label)
            figure_corpora(scenes, by_country, os.path.join(output_dir, "corpora_by_region.png"))
        scenes = pd.concat([scenes, extra], ignore_index=True)

    print(f"{len(scenes):,} scenes after selection")

    figure_floors(scenes, os.path.join(output_dir, "floors_by_region.png"))
    figure_gap(scenes, os.path.join(output_dir, "gap_by_region.png"))
    figure_scenes(scenes, os.path.join(output_dir, "scenes_by_region.png"))

    aggregations = dict(
        scenes=("measured", "size"),
        epsilon_L1=("epsilon_L1_mean", "median"),
        epsilon_L3=("epsilon_L3_mean", "median"),
        floor_L3=("sigma_ch4_L3_mean", "median"),
        measured=("measured", "median"),
        ratio=("ratio_L3", "median"),
        reducible=("reducible", "median"),
        radiance_23=("radiance_B12_mean", "median"),
    )
    aggregations["radiance_23_std"] = ("radiance_B12_std", "median")

    summary = (
        scenes.groupby("case_study")
        .agg(**aggregations)
        .round(4)
        .sort_values("scenes", ascending=False)
    )
    print(summary.to_string())
    if summary_csv:
        summary.to_csv(summary_csv)
        print(f"wrote {summary_csv}")


if __name__ == "__main__":
    app()
