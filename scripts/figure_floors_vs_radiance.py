"""The three shot-noise floors against SWIR radiance.

One figure, two panels -- the minimum significant enhancement and the propagated
standard deviation -- each carrying six curves: the L1/L2/L3 rungs for a
Sentinel-2 and a Landsat instrument.

This is the closed form only; no dataset is read, so it runs anywhere in seconds.
It answers "what does the floor depend on, and by how much" in one picture: the
instrument sets the offset, the rung sets the spacing, and brightness sets the
slope. The per-region figures then show where real scenes actually land.

**Colour carries the instrument and line style carries the rung.** The rungs are
nested rather than independent, so they get an ordered channel (solid to dotted as
each term is added) rather than three more hues, which also keeps the palette to
two all-pairs-validated slots.

Run::

    python -m scripts.figure_floors_vs_radiance figure --output-dir <dir>
"""

import os

import cyclopts
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from marss2l import shot_noise as sn  # noqa: E402

app = cyclopts.App()

#: One instrument per family. The other two sit within a few percent of these:
#: S2B's reference SNR is 2-5% above S2A's, LC08's 4-6% below LC09's.
SERIES = {"S2A": "#2a78d6", "LC09": "#eb6834"}

#: Nested rungs get an ordered channel rather than three more hues.
RUNGS = {
    "L1": ("solid", "background known exactly"),
    "L2": ((0, (5, 1.6)), "band ratio known exactly"),
    "L3": ((0, (1.4, 1.5)), "background from a reference image"),
}

INK, INK_SOFT, GRID, BAND = "#14181f", "#58606c", "#dfe3e8", "#eef0f3"

SZA, VZA = 38.5, 6.1
RATIO_16_OVER_23 = 3.37
OBSERVED_RANGE_23 = (0.14, 18.5)


def _style(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, width=0.6)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


@app.command
def figure(output_dir: str = ".", n_points: int = 160) -> None:
    """Draw ``floors_vs_radiance.png``.

    Args:
        output_dir: Where to write the figure.
        n_points: Samples across the radiance sweep.
    """
    lut = sn._default_lut()
    radiance = np.logspace(np.log10(0.2), np.log10(30.0), n_points)

    fig, (ax_eps, ax_sigma) = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True)
    fig.patch.set_facecolor("white")

    for satellite, colour in SERIES.items():
        ladder = sn.eta_ladder(
            radiance,
            RATIO_16_OVER_23 * radiance,
            radiance,
            RATIO_16_OVER_23 * radiance,
            satellite=satellite,
        )
        for rung, (dash, _) in RUNGS.items():
            eta = ladder[rung]
            ax_eps.plot(
                radiance,
                sn.epsilon(eta, satellite, SZA, VZA, p=0.95, lut=lut),
                color=colour,
                linewidth=1.8,
                linestyle=dash,
                zorder=3,
            )
            ax_sigma.plot(
                radiance,
                sn.sigma_delta_xch4(1.0, eta, satellite, SZA, VZA, lut=lut),
                color=colour,
                linewidth=1.8,
                linestyle=dash,
                zorder=3,
            )

        # Label the family once, on its own top curve, rather than in a legend.
        top = sn.epsilon(ladder["L3"], satellite, SZA, VZA, p=0.95, lut=lut)
        ax_eps.annotate(
            satellite,
            xy=(radiance[0], top[0]),
            xytext=(4, 3),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            fontweight="bold",
        )

    for ax, ylabel, title in [
        (ax_eps, r"$\epsilon$ at $p=0.95$  [ppb]", "a  Minimum significant enhancement"),
        (ax_sigma, r"$\sigma(\Delta \mathrm{XCH}_4)$  [ppb]", "b  Propagated standard deviation"),
    ]:
        ax.axvspan(*OBSERVED_RANGE_23, color=BAND, zorder=0, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(
            ax,
            xlabel=r"2.3 $\mu$m radiance  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
            ylabel=ylabel,
            title=title,
        )

    ax_eps.annotate(
        "radiance observed in MARS-S2L", xy=(1.9, 2900), color=INK_SOFT, fontsize=8, ha="center"
    )

    handles = [
        plt.Line2D([], [], color=INK_SOFT, linewidth=1.8, linestyle=dash, label=f"{rung} — {what}")
        for rung, (dash, what) in RUNGS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SOFT,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()
    path = os.path.join(output_dir, "floors_vs_radiance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")

    # The spacing between rungs is the whole point, so state it rather than
    # leaving the reader to measure it off a log axis.
    for satellite in SERIES:
        ladder = sn.eta_ladder(
            5.8, RATIO_16_OVER_23 * 5.8, 5.8, RATIO_16_OVER_23 * 5.8, satellite=satellite
        )
        values = {
            rung: float(sn.epsilon(ladder[rung], satellite, SZA, VZA, p=0.95, lut=lut))
            for rung in RUNGS
        }
        print(
            f"  {satellite} at the median radiance: "
            + "  ".join(f"{k} {v:.0f} ppb" for k, v in values.items())
            + f"   (L3/L1 = {values['L3'] / values['L1']:.2f})"
        )


if __name__ == "__main__":
    app()
