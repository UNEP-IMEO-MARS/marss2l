"""Where the scenes of the two corpora are (``corpora_map.png``).

Hexbin maps of the MARS-S2L test scenes and the CloudSEN12 scenes left after the selection of
:func:`scripts.shot_noise.figure_regional.load_scenes`, one panel per corpus. Each panel has its
own logarithmic colour scale, since the densest MARS-S2L hexagon holds about a hundred times more
scenes than the densest CloudSEN12 one.

Run::

    python -m scripts.shot_noise.figure_corpora_map figure <stats csv> <images csv> \\
        <cloudsen12 stats csv> <cloudsen12 images csv> --output-path <png>
"""

import cyclopts
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from scripts.shot_noise.figure_regional import INK, INK_SOFT, load_scenes  # noqa: E402

app = cyclopts.App()

#: Hexagons across the width of the map, about 485 km each in the Robinson projection.
HEXAGONS_ACROSS = 70


def _count_ticks(largest: float) -> list:
    """Plain-number ticks for a log colour bar: decades over a wide range, 1-2-5 over a narrow one."""
    steps = (1,) if largest > 200 else (1, 2, 5)
    return [m * 10**e for e in range(6) for m in steps if m * 10**e <= largest]


@app.command
def figure(
    stats_csv: str,
    images_csv: str,
    extra_stats_csv: str,
    extra_images_csv: str,
    *,
    output_path: str = "corpora_map.png",
    extra_label: str = "CloudSEN12",
) -> None:
    """Draw the map of the two corpora.

    Args:
        stats_csv: Output of ``marss2l.stats_dataset`` on the native grids, MARS-S2L.
        images_csv: Image metadata CSV of MARS-S2L, for the selection and the coordinates.
        extra_stats_csv: The same statistics for CloudSEN12.
        extra_images_csv: Image metadata CSV of CloudSEN12.
        output_path: Where to write the figure.
        extra_label: Name of the second corpus.
    """
    # cartopy is a dependency of the package, but only these figures need it.
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    corpora = [
        ("MARS-S2L test split", load_scenes(stats_csv, images_csv)),
        (extra_label, load_scenes(extra_stats_csv, extra_images_csv, label=extra_label)),
    ]

    proj, lonlat = ccrs.Robinson(), ccrs.PlateCarree()
    corners = proj.transform_points(
        lonlat, np.array([-180.0, 180.0, 0.0, 0.0]), np.array([0.0, 0.0, -90.0, 90.0])
    )
    extent = (corners[0, 0], corners[1, 0], corners[2, 1], corners[3, 1])
    # Regular hexagons: matplotlib's grid has sqrt(3) more rows than columns per unit length.
    rows = int(
        round(HEXAGONS_ACROSS * (extent[3] - extent[2]) / (extent[1] - extent[0]) / np.sqrt(3))
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), subplot_kw={"projection": proj})
    for k, (ax, (name, scenes)) in enumerate(zip(axes, corpora, strict=True)):
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="#ececec", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#8a8a8a", zorder=1)
        xy = proj.transform_points(lonlat, scenes.lon.values, scenes.lat.values)
        hexagons = ax.hexbin(
            xy[:, 0],
            xy[:, 1],
            gridsize=(HEXAGONS_ACROSS, rows),
            extent=extent,
            mincnt=1,
            cmap="viridis",
            linewidths=0,
            zorder=2,
        )
        counts = hexagons.get_array()
        hexagons.set_norm(LogNorm(vmin=1, vmax=counts.max()))
        print(
            f"{name}: {len(scenes):,} scenes in {len(counts):,} hexagons, at most {int(counts.max()):,}"
        )
        ax.spines["geo"].set_edgecolor("#bdbdbd")
        ax.spines["geo"].set_linewidth(0.6)
        ax.set_title(
            f"{'ab'[k]}  {name} ({len(scenes):,} scenes)",
            loc="left",
            fontsize=10.5,
            color=INK,
            pad=6,
        )
        bar = fig.colorbar(
            hexagons,
            ax=ax,
            orientation="horizontal",
            fraction=0.05,
            pad=0.04,
            aspect=35,
            shrink=0.8,
        )
        ticks = _count_ticks(counts.max())
        bar.set_ticks(ticks)
        bar.set_ticklabels([f"{t:,}" for t in ticks])
        bar.ax.minorticks_off()
        bar.set_label("scenes per hexagon", color=INK_SOFT, fontsize=9)
        bar.ax.tick_params(labelsize=8, colors=INK_SOFT)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    app()
