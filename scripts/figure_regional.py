"""The two regional figures of the shot-noise paper, from the sweep CSV.

``floors_by_region.png`` (F1)
    The three floors per case study, faceted by satellite family. What the
    instrument can resolve, and where the multi-pass construction costs most.

``gap_by_region.png`` (F2)
    The retrieval's measured noise against its own photon-noise floor, per case
    study, both as standard deviations in ppb, with the reducible share beside it.
    The crux result: how much of today's error a better background estimate could
    remove.

Everything is a groupby on the CSV that ``stats_dataset.py`` writes. Nothing here
recomputes a raster, and a point is always **one scene** -- pixels within a scene
are strongly correlated, so a distribution over pixels would claim a precision the
data does not have.

Scene selection, per the epic: the named split, ``observability == "clear"``,
onshore only (offshore uses the single-pass SBMP retrieval, so its noise is not
comparable and L3 does not exist for it) and no night reference pass.

Run::

    python -m scripts.figure_regional figures --stats-csv <csv> --output-dir <dir>
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

from marss2l.dataframe_image_plumes import ORDER_CASE_STUDIES  # noqa: E402

app = cyclopts.App()

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


def load_scenes(stats_csv: str, images_csv: str) -> pd.DataFrame:
    """Read the sweep and apply the scene selection the figures use.

    Args:
        stats_csv: Output of ``stats_dataset.py``.
        images_csv: The image metadata CSV, for ``observability`` and ``case_study``.

    Returns:
        One row per usable scene, with the measured noise and the ratio to each rung.
    """
    stats = pd.read_csv(stats_csv, low_memory=False)
    meta = pd.read_csv(
        images_csv, usecols=["id_loc_image", "observability", "case_study", "sza_bg_source"]
    )
    scenes = stats.merge(meta, on="id_loc_image", how="left")

    scenes = scenes[
        (scenes.observability == "clear")
        & (~scenes.offshore.astype(bool))
        & (scenes.sza_bg_source != "night")
        & scenes.sigma_ch4_L3_mean.notna()
    ].copy()

    # O-base: the retrieval's own noise, on pixels where it should read zero.
    scenes["measured"] = np.where(
        scenes.isplume == 1, scenes.ch4_valid_noplume_std, scenes.ch4_valid_std
    )
    scenes = scenes[scenes.measured.notna() & (scenes.measured > 0)]

    scenes["family"] = np.where(scenes.satellite.str.startswith("S2"), "Sentinel-2", "Landsat")
    for rung in RUNG_COLOURS:
        scenes[f"ratio_{rung}"] = scenes.measured / scenes[f"sigma_ch4_{rung}_mean"]
    # Exact, because both sides are standard deviations of the same quantity.
    scenes["reducible"] = 1 - (scenes.sigma_ch4_L3_mean / scenes.measured) ** 2

    return scenes


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
    order = [c for c in ORDER_CASE_STUDIES if c in set(scenes.case_study)]
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


def figure_gap(scenes: pd.DataFrame, path: str) -> None:
    """F2: measured noise against the L3 floor, per case study, with the share."""
    order = [c for c in ORDER_CASE_STUDIES if c in set(scenes.case_study)]
    order = [c for c in order if (scenes.case_study == c).sum() >= 5]

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


@app.command
def figures(
    stats_csv: str,
    images_csv: str,
    output_dir: str = ".",
    summary_csv: Optional[str] = None,
) -> None:
    """Draw F1 and F2 from a sweep.

    Args:
        stats_csv: Output of ``stats_dataset.py``.
        images_csv: Image metadata CSV, for observability and case study.
        output_dir: Where to write the figures.
        summary_csv: Optional path to write the per-region table behind them.
    """
    scenes = load_scenes(stats_csv, images_csv)
    print(f"{len(scenes):,} scenes after selection")

    figure_floors(scenes, os.path.join(output_dir, "floors_by_region.png"))
    figure_gap(scenes, os.path.join(output_dir, "gap_by_region.png"))

    summary = (
        scenes.groupby("case_study")
        .agg(
            scenes=("measured", "size"),
            epsilon_L1=("epsilon_L1_mean", "median"),
            epsilon_L3=("epsilon_L3_mean", "median"),
            floor_L3=("sigma_ch4_L3_mean", "median"),
            measured=("measured", "median"),
            ratio=("ratio_L3", "median"),
            reducible=("reducible", "median"),
        )
        .round(2)
        .sort_values("scenes", ascending=False)
    )
    print(summary.to_string())
    if summary_csv:
        summary.to_csv(summary_csv)
        print(f"wrote {summary_csv}")


if __name__ == "__main__":
    app()
