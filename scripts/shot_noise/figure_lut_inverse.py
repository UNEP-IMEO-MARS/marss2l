"""The look-up-table inverse that maps a transmittance ratio to an enhancement (``lut_inverse.png``).

The enhancement against the ratio of the 2.3 and 1.6 um band transmittances,
:math:`\\Delta\\tau_{23/16}^{-1}`, at each air-mass factor of the look-up table, for Sentinel-2A and Landsat-8. Dots are the
nodes of the table; the dashed curves are the quadratics of
:func:`marss2l.shot_noise.sigma_delta_xch4`, drawn over the range they are fitted on (shaded).
Beneath, the same nodes against the relative transmittance of the 2.3 um band alone, with a
quadratic fitted over :data:`FIT_RANGE_23`.
No data needed.

Run::

    python -m scripts.shot_noise.figure_lut_inverse figure --output-path <dir>/lut_inverse.png
"""

import cyclopts
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from marss2l import shot_noise as sn  # noqa: E402

app = cyclopts.App()

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dededa"
BAND = "#eeeeea"

#: Ordinal blue ramp, light to dark with the air-mass factor (steps 250-700, one per table AMF).
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281", "#0d366b"]


def _style_axis(ax, *, xlabel: str = "", ylabel: str = "") -> None:
    """Recessive grid and axes; labels in ink."""
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=7, width=0.6)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=8)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=8)


def sza_for_amf(amf: float) -> float:
    """Solar zenith angle, in degrees, that gives ``amf`` at nadir view (AMF = 1/cos SZA + 1)."""
    return float(np.degrees(np.arccos(min(1.0, 1.0 / (amf - 1.0)))))


def lut_nodes(satellite: str, amf: float):
    """The table's enhancements (ppb) at ``amf``, with the relative transmittance of the 2.3 um
    band and the ratio of the 2.3 to the 1.6 um band transmittances they produce."""
    lut = sn.default_lut()
    deltach4 = np.asarray(lut.f_ch4_mr(amf)) - lut.background_concentration
    t23, t16 = lut.transmittance_B12_B11(satellite, None, None, deltach4, amf=amf)
    t23, t16 = np.asarray(t23), np.asarray(t16)
    return deltach4, t23, t23 / t16


#: Range over which the inverse of the 2.3 um band alone is fitted by a quadratic. Wider than the
#: ratio's, since the band alone departs further from one for the same enhancement.
FIT_RANGE_23 = (0.85, 1.05)


def inverse_23(satellite: str, amf: float, t23: np.ndarray) -> np.ndarray:
    """Enhancement (ppb) from the relative transmittance of the 2.3 um band alone, without the
    retrieval's clip of the input and keeping negative enhancements."""
    ppb = sn.default_lut().deltach4_from_B12_transmittance(
        satellite, None, None, t23, amf=amf, clip_values_retrieval=False, allow_negative_ch4=True
    )
    return np.asarray(ppb)


def quadratic_23(satellite: str, amf: float) -> np.ndarray:
    """Coefficients of the quadratic fitted to :func:`inverse_23` over :data:`FIT_RANGE_23`."""
    m = np.linspace(*FIT_RANGE_23, 101)
    return np.polyfit(m, inverse_23(satellite, amf, m), 2)


def quadratic(satellite: str, amf: float) -> np.ndarray:
    """Coefficients of the quadratic fitted over :data:`marss2l.shot_noise._FIT_RANGE`, as
    :func:`marss2l.shot_noise.sigma_delta_xch4` fits it."""
    m = np.linspace(*sn._FIT_RANGE, 101)
    ppb = sn.default_lut().deltach4_from_ratio_transmittance(
        satellite, sza=sza_for_amf(amf), vza=0.0, ratio_il=m, clip_values_retrieval=False
    )
    return np.polyfit(m, np.asarray(ppb), 2)


#: Instruments drawn, one panel each, with their panel titles.
PANELS = (("S2A", "Sentinel-2A"), ("LC08", "Landsat-8"))


def _panel_ratio(ax, satellite: str) -> None:
    """LUT nodes against the band ratio at every air-mass factor, and the quadratic fitted at each."""
    lo, hi = sn._FIT_RANGE
    ax.axvspan(lo, hi, color=BAND, zorder=0, linewidth=0)
    xf = np.linspace(lo, hi, 200)
    for amf, colour in zip(sn.default_lut().amf_arr, RAMP):
        deltach4, _, ratio = lut_nodes(satellite, float(amf))
        ax.scatter(ratio, deltach4, s=20, color=colour, edgecolor="white", linewidth=1.0,
                   zorder=5, label=f"AMF {amf:.2f}")
        ax.plot(xf, np.polyval(quadratic(satellite, float(amf)), xf), color=colour,
                linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
    ax.text(0.5 * (lo + hi), 0.97, "quadratic fit range", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color=INK_SOFT)


def _panel_band(ax, satellite: str) -> None:
    """LUT nodes against the relative transmittance of the 2.3 um band alone, and the quadratic
    fitted at each air-mass factor."""
    lo, hi = FIT_RANGE_23
    ax.axvspan(lo, hi, color=BAND, zorder=0, linewidth=0)
    xf = np.linspace(lo, hi, 200)
    for amf, colour in zip(sn.default_lut().amf_arr, RAMP):
        deltach4, t23, _ = lut_nodes(satellite, float(amf))
        ax.scatter(t23, deltach4, s=20, color=colour, edgecolor="white", linewidth=1.0, zorder=5)
        ax.plot(xf, np.polyval(quadratic_23(satellite, float(amf)), xf), color=colour,
                linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
    ax.text(0.5 * (lo + hi), 0.97, "quadratic fit range", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color=INK_SOFT)


@app.command
def figure(output_path: str = "lut_inverse.png") -> None:
    """Draw the look-up-table nodes, Sentinel-2A beside Landsat-8: against the band ratio, with
    the quadratic fits, on top; against the 2.3 um band alone, with its own fits, beneath.

    Args:
        output_path: Where to write the figure.
    """
    fig, axes = plt.subplots(2, len(PANELS), figsize=(9.0, 6.6), sharex=True, sharey=True)
    for k, (satellite, title) in enumerate(PANELS):
        _panel_ratio(axes[0, k], satellite)
        _panel_band(axes[1, k], satellite)
        axes[0, k].set_title(title, color=INK, fontsize=9, loc="left", pad=6)
    for (r, k), ax in np.ndenumerate(axes):
        ax.axhline(0, color=INK_SOFT, linewidth=0.6, zorder=1)
        ax.axvline(1, color=INK_SOFT, linewidth=0.6, linestyle=":", zorder=1)
        ax.tick_params(labelbottom=True)
        _style_axis(
            ax,
            xlabel=r"$\Delta\tau_{23/16}$" if r == 0 else r"$\Delta\tau_{23}$",
            ylabel=r"$\Delta\mathrm{XCH_4}$ (ppb)" if k == 0 else "",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [
        mpl.lines.Line2D([], [], color=INK_SOFT, marker="o", markersize=4, linestyle=""),
        mpl.lines.Line2D([], [], color=INK_SOFT, linewidth=1.4, linestyle=(0, (4, 2))),
    ]
    labels += ["look-up table", "quadratic fit"]
    fig.legend(handles, labels, fontsize=7, frameon=False, labelcolor=INK_SOFT,
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    app()
