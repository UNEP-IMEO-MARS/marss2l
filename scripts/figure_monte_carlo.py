"""Monte-Carlo validation figures for the shot-noise propagation.

Two figures, both for the shot-noise paper:

``monte_carlo.png``
    Closed-form ``sigma(MBMP)`` and ``sigma(delta XCH4)`` against Monte-Carlo
    estimates over the operating range of SWIR radiances, with the agreement ratio
    below each. This validates the two first-order expansions the whole analysis
    rests on -- if they disagreed anywhere in the operating range, every figure
    downstream would be wrong.

``monte_carlo_epsilon.png``
    Validation of the minimum significant enhancement. Left: ``epsilon`` at
    ``p=0.95`` against the empirical 95th percentile of the retrieval on plume-free
    ground. Right: the realised false-alarm rate against the nominal ``1-p``, which
    is what ``epsilon`` actually claims.

**Two instruments are drawn, not four.** S2A is the noisiest of the four in these
bands and LC09 the quietest, so they bracket S2B and LC08; adding all four would put
four hues on a scatter for no extra information.

Run::

    python -m scripts.figure_monte_carlo figures --output-dir <dir>
"""

from typing import Tuple

import cyclopts
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from marss2l import shot_noise as sn  # noqa: E402

app = cyclopts.App()

# ── palette ──────────────────────────────────────────────────────────────────
# Two categorical slots, validated for all-pairs use (CVD dE 24.7, normal 33.6).
# Identity is colour; closed-form versus Monte-Carlo is mark type, so neither
# distinction rests on colour alone.
SERIES = {"S2A": "#2a78d6", "LC09": "#eb6834"}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dededa"
BAND = "#eeeeea"

# ── the scene the sweep represents ───────────────────────────────────────────
#: Solar and view zenith angle: the medians of the MARS-S2L target images.
SZA, VZA = 38.5, 6.1

#: Median ratio of the 1.6 um to the 2.3 um radiance, measured over 13 MARS-S2L
#: case studies. The sweep is over the 2.3 um radiance; the companion band follows.
RATIO_16_OVER_23 = 3.37

#: 1st-99th percentile span of the 2.3 um radiance over those same scenes -- the
#: range the retrieval actually operates in, shaded on every panel.
OBSERVED_RANGE_23 = (0.14, 18.5)
OBSERVED_MEDIAN_23 = 5.84


def _radiances(radiance_23: float) -> Tuple[float, float, float, float]:
    """The four radiances of a plume-free pixel: both bands, both passes.

    Identical target and reference radiances make the true MBMP exactly 1, which is
    what the retrieval reads on plume-free ground after ``ratio_IL`` normalises by
    the scene mean.
    """
    radiance_16 = RATIO_16_OVER_23 * radiance_23
    return radiance_23, radiance_16, radiance_23, radiance_16


def _style_axis(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Recessive grid and axes; labels in ink, never in a series colour."""
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=7, width=0.6)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=9, loc="left", pad=6)


def _shade_observed(ax) -> None:
    ax.axvspan(*OBSERVED_RANGE_23, color=BAND, zorder=0, linewidth=0)


def compute_sweep(satellite: str, radiances_23: np.ndarray, n_samples: int, seed: int) -> dict:
    """Closed-form and Monte-Carlo noise at each radiance, for one instrument."""
    lut = sn._default_lut()
    rng = np.random.default_rng(seed)

    out = {k: [] for k in ("eta", "sigma_mbmp_mc", "sigma_ch4", "sigma_ch4_mc", "eps", "eps_mc")}
    for radiance_23 in radiances_23:
        four = _radiances(float(radiance_23))
        eta = float(sn.eta_ladder(*four, satellite=satellite)["L3"])

        mbmp_samples = sn.monte_carlo_mbmp(*four, satellite=satellite, n_samples=n_samples, rng=rng)
        ch4_samples = sn.monte_carlo_delta_xch4(
            *four, satellite=satellite, sza=SZA, vza=VZA, n_samples=n_samples, rng=rng, lut=lut
        )

        out["eta"].append(eta)
        out["sigma_mbmp_mc"].append(float(mbmp_samples.std()))
        out["sigma_ch4"].append(float(sn.sigma_delta_xch4(1.0, eta, satellite, SZA, VZA, lut=lut)))
        out["sigma_ch4_mc"].append(float(ch4_samples.std()))
        out["eps"].append(float(sn.epsilon(eta, satellite, SZA, VZA, p=0.95, lut=lut)))
        out["eps_mc"].append(float(np.percentile(ch4_samples, 95)))

    return {k: np.asarray(v) for k, v in out.items()}


def compute_calibration(
    satellite: str, radiances_23: np.ndarray, n_samples: int, seed: int
) -> dict:
    """Realised false-alarm rate against the nominal ``1-p``, per radiance."""
    lut = sn._default_lut()
    rng = np.random.default_rng(seed)
    nominal = np.array([0.32, 0.20, 0.10, 0.05, 0.02, 0.01])

    realised = []
    for radiance_23 in radiances_23:
        four = _radiances(float(radiance_23))
        eta = float(sn.eta_ladder(*four, satellite=satellite)["L3"])
        samples = sn.monte_carlo_delta_xch4(
            *four, satellite=satellite, sza=SZA, vza=VZA, n_samples=n_samples, rng=rng, lut=lut
        )
        realised.append(
            [
                float(
                    (samples > float(sn.epsilon(eta, satellite, SZA, VZA, p=1 - a, lut=lut))).mean()
                )
                for a in nominal
            ]
        )

    return {"nominal": nominal, "realised": np.asarray(realised)}


def figure_agreement(sweeps: dict, radiances_23: np.ndarray, path: str) -> None:
    """Closed form against Monte Carlo, with the agreement ratio beneath."""
    fig, axes = plt.subplots(
        2, 3, figsize=(9.6, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.0]}
    )
    fig.patch.set_facecolor("white")

    panels = [
        (
            0,
            "eta",
            "sigma_mbmp_mc",
            r"$\sigma(\mathrm{MBMP})$",
            "a  Transmittance ratio",
            (0.97, 1.03),
        ),
        (
            1,
            "sigma_ch4",
            "sigma_ch4_mc",
            r"$\sigma(\Delta \mathrm{XCH}_4)$  [ppb]",
            "b  Retrieved enhancement",
            (0.97, 1.03),
        ),
        # The detection threshold itself, against the empirical 95th percentile of
        # the retrieval on plume-free ground -- the quantity the regional figures
        # plot, so its agreement belongs beside the two standard deviations.
        (
            2,
            "eps",
            "eps_mc",
            r"$\epsilon$ at $p=0.95$  [ppb]",
            "c  Detection threshold",
            # A quantile is a harder thing to reproduce than a standard deviation,
            # so the agreement is looser at the noisy end; give it room to show.
            (0.96, 1.05),
        ),
    ]

    for column, closed_key, mc_key, ylabel, title, ratio_limits in panels:
        top, bottom = axes[0, column], axes[1, column]
        for satellite, colour in SERIES.items():
            sweep = sweeps[satellite]
            top.plot(
                radiances_23,
                sweep[closed_key],
                color=colour,
                linewidth=2.0,
                zorder=3,
                solid_capstyle="round",
            )
            top.plot(
                radiances_23,
                sweep[mc_key],
                marker="o",
                markersize=4.5,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=colour,
                markeredgewidth=1.4,
                zorder=4,
            )
            bottom.plot(
                radiances_23,
                sweep[mc_key] / sweep[closed_key],
                marker="o",
                markersize=3.5,
                linestyle="-",
                linewidth=1.0,
                color=colour,
                markerfacecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )
            # Direct label at the left end, where the curves are furthest apart.
            top.annotate(
                satellite,
                xy=(radiances_23[0], sweep[closed_key][0]),
                xytext=(3, 4),
                textcoords="offset points",
                color=colour,
                fontsize=8,
                fontweight="bold",
            )

        _shade_observed(top)
        _shade_observed(bottom)
        top.set_xscale("log")
        top.set_yscale("log")
        _style_axis(top, ylabel=ylabel, title=title)

        bottom.axhline(1.0, color=INK_SOFT, linewidth=0.8, zorder=2)
        bottom.axhspan(0.98, 1.02, color=GRID, alpha=0.6, zorder=1, linewidth=0)
        bottom.set_ylim(*ratio_limits)
        _style_axis(
            bottom,
            xlabel=r"2.3 $\mu$m radiance  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
            ylabel="Monte Carlo\n/ closed form",
        )

    # One legend for the mark types; identity is carried by the direct labels.
    handles = [
        plt.Line2D([], [], color=INK_SOFT, linewidth=2.0, label="closed form"),
        plt.Line2D(
            [],
            [],
            color=INK_SOFT,
            marker="o",
            markersize=4.5,
            linestyle="none",
            markerfacecolor="white",
            markeredgewidth=1.4,
            label="Monte Carlo",
        ),
        plt.Line2D([], [], color=BAND, linewidth=8, label="observed radiance range"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SOFT,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_epsilon(sweeps: dict, calibrations: dict, radiances_23: np.ndarray, path: str) -> None:
    """The detection threshold: its value, and the false-alarm rate it claims."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.patch.set_facecolor("white")

    left, right = axes
    for satellite, colour in SERIES.items():
        sweep = sweeps[satellite]
        left.plot(radiances_23, sweep["eps"], color=colour, linewidth=2.0, zorder=3)
        left.plot(
            radiances_23,
            sweep["eps_mc"],
            marker="o",
            markersize=4.5,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=1.4,
            zorder=4,
        )
        left.annotate(
            satellite,
            xy=(radiances_23[0], sweep["eps"][0]),
            xytext=(3, 4),
            textcoords="offset points",
            color=colour,
            fontsize=8,
            fontweight="bold",
        )

        calibration = calibrations[satellite]
        for row in calibration["realised"]:
            right.plot(
                calibration["nominal"],
                row,
                marker="o",
                markersize=4.0,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=colour,
                markeredgewidth=1.2,
                alpha=0.85,
                zorder=3,
            )

    _shade_observed(left)
    left.set_xscale("log")
    left.set_yscale("log")
    _style_axis(
        left,
        xlabel=r"2.3 $\mu$m radiance  [W m$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$]",
        ylabel=r"$\epsilon$ at $p=0.95$  [ppb]",
        title="a  Detection threshold",
    )
    left.legend(
        handles=[
            plt.Line2D([], [], color=INK_SOFT, linewidth=2.0, label="closed form"),
            plt.Line2D(
                [],
                [],
                color=INK_SOFT,
                marker="o",
                markersize=4.5,
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=1.4,
                label="Monte-Carlo 95th pct",
            ),
        ],
        loc="upper right",
        frameon=False,
        fontsize=7.5,
        labelcolor=INK_SOFT,
    )

    limits = (0.006, 0.5)
    right.plot(limits, limits, color=INK_SOFT, linewidth=1.0, zorder=2)
    right.annotate(
        "1:1",
        xy=(0.25, 0.25),
        xytext=(4, -9),
        textcoords="offset points",
        color=INK_SOFT,
        fontsize=7.5,
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(*limits)
    right.set_ylim(*limits)
    _style_axis(
        right,
        xlabel=r"nominal false-alarm rate  $1-p$",
        ylabel="realised rate",
        title="b  Calibration of the threshold",
    )
    right.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=colour,
                marker="o",
                markersize=4.5,
                linestyle="none",
                markerfacecolor="white",
                markeredgewidth=1.4,
                label=satellite,
            )
            for satellite, colour in SERIES.items()
        ],
        loc="upper left",
        frameon=False,
        fontsize=7.5,
        labelcolor=INK_SOFT,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@app.command
def figures(
    output_dir: str = ".",
    n_samples: int = 200_000,
    n_radiances: int = 12,
    seed: int = 0,
) -> None:
    """Compute the Monte-Carlo validation and write both figures.

    Args:
        output_dir: Directory to write ``monte_carlo.png`` and
            ``monte_carlo_epsilon.png`` into.
        n_samples: Draws per radiance. 200k gives a Monte-Carlo standard error on
            the estimated sigma of about 0.2%, well inside the agreement claimed.
        n_radiances: Points across the radiance sweep.
        seed: Base seed, so the figures are reproducible.
    """
    import os

    radiances_23 = np.logspace(np.log10(0.2), np.log10(30.0), n_radiances)

    sweeps, calibrations = {}, {}
    for offset, satellite in enumerate(SERIES):
        print(f"sweeping {satellite} over {n_radiances} radiances, {n_samples} samples each")
        sweeps[satellite] = compute_sweep(satellite, radiances_23, n_samples, seed + offset)
        calibrations[satellite] = compute_calibration(
            satellite, radiances_23[::4], n_samples, seed + 100 + offset
        )

    for satellite, sweep in sweeps.items():
        for label, key, mc_key in [
            ("sigma(MBMP)", "eta", "sigma_mbmp_mc"),
            ("sigma(dXCH4)", "sigma_ch4", "sigma_ch4_mc"),
            ("epsilon", "eps", "eps_mc"),
        ]:
            ratio = sweep[mc_key] / sweep[key]
            print(
                f"  {satellite} {label:12s} MC/closed form: "
                f"min {ratio.min():.4f} max {ratio.max():.4f}"
            )

    agreement_path = os.path.join(output_dir, "monte_carlo.png")
    epsilon_path = os.path.join(output_dir, "monte_carlo_epsilon.png")
    figure_agreement(sweeps, radiances_23, agreement_path)
    figure_epsilon(sweeps, calibrations, radiances_23, epsilon_path)
    print(f"wrote {agreement_path}\nwrote {epsilon_path}")


if __name__ == "__main__":
    app()
