from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def plot_ch4_background_map(
    dataframe: pd.DataFrame,
    units_out: str = "ppb",
    plot_type: str = "hexbin",
    value_column: Optional[str] = "ch4_mean_noplume",
    lon_column: str = "lon",
    lat_column: str = "lat",
    vmax: Optional[float] = 1e4,
    vmin: Optional[float] = 10,
    gridsize: int = 60,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a global map of the average background methane enhancement (``value_column``)
    on a cartopy ``PlateCarree`` map with a log-scaled colorbar.

    The values in ``value_column`` are assumed to be in ``ppb``. If ``units_out`` is
    different from ``"ppb"`` they are converted with
    :func:`marshsi.quantification.convert_units` (and so are ``vmax`` and ``vmin``).

    If ``value_column`` is ``None`` the map instead shows the number of images per
    hexbin cell (a "count" map, hexbin only); no unit conversion is applied.

    Args:
        dataframe (pd.DataFrame): must contain ``lon_column``, ``lat_column`` and
            (unless counting) ``value_column``. Values in ``value_column`` are expected in ``ppb``.
        units_out (str): output units for the color scale. One of the units accepted
            by :func:`marshsi.quantification.convert_units` (e.g. "ppb", "ppm",
            "ppm x m"). Ignored when ``value_column`` is None. Defaults to "ppb".
        plot_type (str): "hexbin" for a hexbin of the per-cell mean, or "scatterplot"
            to plot the individual points. Must be "hexbin" when counting. Defaults to "hexbin".
        value_column (str, optional): column to color by (in ppb). If None, color by the
            number of images per hexbin cell instead. Defaults to "ch4_mean_noplume".
        lon_column (str): longitude column. Defaults to "lon".
        lat_column (str): latitude column. Defaults to "lat".
        vmax (float, optional): upper clip of the (log) color scale, in ppb for a value
            map. None auto-scales. Defaults to 1e4.
        vmin (float, optional): lower clip of the (log) color scale, in ppb for a value
            map. None auto-scales. Defaults to 10.
        gridsize (int): hexbin grid size (ignored for scatterplot). Defaults to 60.
        cmap (str): matplotlib colormap. Defaults to "viridis".
        colorbar_label (str, optional): colorbar label. If None a sensible default is
            used depending on the mode. Defaults to None.
        title (str, optional): axis title. Defaults to None.
        ax (plt.Axes, optional): cartopy GeoAxes to plot on. If None a new figure/axis
            is created. Defaults to None.

    Returns:
        plt.Axes: the (cartopy) axis with the plot.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if plot_type not in ("hexbin", "scatterplot"):
        raise ValueError(f'plot_type must be "hexbin" or "scatterplot", found "{plot_type}"')

    count_mode = value_column is None
    if count_mode and plot_type != "hexbin":
        raise ValueError('counting the number of images (value_column=None) requires plot_type="hexbin"')

    columns = [lon_column, lat_column] + ([] if count_mode else [value_column])
    df = dataframe[columns].copy()

    vmax_out, vmin_out = vmax, vmin
    if not count_mode:
        # values are stored in ppb; convert both the data and the color clip if needed
        if units_out != "ppb":
            from marshsi.quantification import convert_units

            df[value_column] = convert_units(df[value_column].to_numpy(), "ppb", units_out)
            if vmax is not None:
                vmax_out = convert_units(vmax, "ppb", units_out)
            if vmin is not None:
                vmin_out = convert_units(vmin, "ppb", units_out)
        # LogNorm needs strictly positive values
        df = df[df[value_column] > 0]

    # drop rows without a valid location
    df = df[df[lon_column].notna() & df[lat_column].notna()]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    fig = ax.figure

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    norm = LogNorm(vmax=vmax_out, vmin=vmin_out)
    if plot_type == "hexbin":
        mappable = ax.hexbin(
            df[lon_column], df[lat_column],
            C=None if count_mode else df[value_column],
            reduce_C_function=np.mean,
            gridsize=gridsize, cmap=cmap, norm=norm,
            mincnt=1, linewidths=0.2, transform=ccrs.PlateCarree(),
        )
    else:
        mappable = ax.scatter(
            df[lon_column], df[lat_column],
            c=df[value_column], cmap=cmap, norm=norm,
            s=8, linewidths=0, transform=ccrs.PlateCarree(),
        )

    extend = "max" if vmax_out is not None else "neither"
    cb = fig.colorbar(mappable, ax=ax, orientation="horizontal", shrink=0.7, pad=0.05, extend=extend)
    if colorbar_label is None:
        colorbar_label = "number of images" if count_mode else rf"mean $\Delta$XCH$_4$ outside plume ({units_out})"
    cb.set_label(colorbar_label)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    if title is not None:
        ax.set_title(title)

    return ax
