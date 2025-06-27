import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Union, Tuple, Dict
import numpy as np


def plot_time_series(mlis_df:pd.DataFrame, 
                     only_validated:bool=False,
                     nsplits:int=1,
                     width:int=10,
                     height:int=2,
                     colors_satellite:Optional[Dict[str,str]]=None,
                     cluster_sentinel2_and_landsat:bool=True,
                     loc_legend:Optional[str]=None,
                     ax:Optional[plt.Axes]=None) -> Union[plt.Axes, Tuple[plt.Figure, list[plt.Axes]]]:
    """
    Plot the time series of flux rate for the given MarsLocationImage objects.


    Args:
        mlis_df (pd.DataFrame): DataFrame with observations over a particular site.
        only_validated (bool, optional): If True, it will only plot the validated plumes. 
            Defaults to False.
        nsplits (int, optional): If greater than 1, it will split the plot in nsplits 
            subplots.
        width (int, optional): Width of the plot. Defaults to 10.
        height (int, optional): Height of the plot. Defaults to 2.
        colors_satellite (Optional[Dict[str,str]], optional): Dictionary with the colors.
            If None it will use the default matplotlib colors. Defaults to None.
        cluster_sentinel2_and_landsat: If True, it will cluster S2A and S2B as S2 and 
            LC08 and LC09 as L89. Defaults to True.
        loc_legend (Optional[str], optional): Location of the legend. Defaults to None.
        ax (Optional[plt.Axes], optional): Axes to plot. If nsplits > 1, 
            it will be ignored.

    Returns:
        Union[plt.Axes, Tuple[plt.Figure, list[plt.Axes]]]: if nsplits > 1, 
            it will return a tuple with the figure and the list of axes.
            If nsplits == 1, and ax is not None, it will return the ax.
    """
    if nsplits > 1:
        assert ax is None, "ax must be None if nsplits > 1 because it will be ignored"
    
    def isplumestr(mli) -> str:
        if (mli.observability is not None) and (mli.observability != "clear"):
            return "cloudy/bad retrieval"
        
        if mli.observability is None:
            raise ValueError("Observability must be defined")
            # if mli.too_much_clouds(threshold_max_noclear=threshold_max_noclear):
            #     return "cloudy/bad retrieval"
        
        if (mli.isplume is None):
            return "not validated"
        elif mli.isplume:
            return "yes"
        else:
            return "no"

    mlis_df["isplumestr"] = mlis_df.apply(lambda row: isplumestr(row), axis=1)
    mlis_df.loc[mlis_df.isplumestr == "no", ["ch4_fluxrate", "ch4_fluxrate_std"]] = 0
    mlis_df.loc[mlis_df.isplumestr == "cloudy/bad retrieval", ["ch4_fluxrate", "ch4_fluxrate_std"]] = pd.NA

    if only_validated:
        mlis_df = mlis_df[mlis_df.isplumestr != "not validated"]
    
    if cluster_sentinel2_and_landsat:
        mlis_df.satellite = mlis_df.satellite.apply(lambda x: "S2" if x.startswith("S2") else x)
        mlis_df.satellite = mlis_df.satellite.apply(lambda x: "L8/9" if x.startswith("LC08") or x.startswith("LC09") else x)
    
    if colors_satellite is None:
        colors_satellite = {satame: color for satame, color in zip(mlis_df.satellite.unique(), 
                                                                    plt.cm.tab10.colors)}
    
    max_fluxrate = mlis_df.ch4_fluxrate.max()
    max_std = mlis_df.ch4_fluxrate_std.max()

    max_val_plot_th = max_fluxrate/1000 + max_std/1_000

    def plot_subset(df, ax:plt.Axes, plot_legend:bool=False):
        df_positive = df[(df.isplumestr == "yes") & (df.ch4_fluxrate > 0)]
        df_negative = df[(df.isplumestr == "no")]
        df_nonobservable = df[df.isplumestr == "cloudy/bad retrieval"]

        ax.scatter(df_nonobservable.tile_date, 
                    max_val_plot_th+np.zeros_like(df_nonobservable.ch4_fluxrate.values),
                    label="cloudy", color="gray")
        for i, (satame, data) in enumerate(df_positive.groupby("satellite")):
            ax.errorbar(data.tile_date, data.ch4_fluxrate/1000,
                        yerr=2*data.ch4_fluxrate_std/1000,
                        fmt="o", label=satame, 
                        color=colors_satellite[satame])
            data_neg = df_negative[df_negative.satellite == satame]
            ax.scatter(data_neg.tile_date, data_neg.ch4_fluxrate.values, 
                        color=colors_satellite[satame])
        if plot_legend:
            # from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            colors_satellite_with_cloudy = colors_satellite.copy()
            colors_satellite_with_cloudy["cloudy"] = "gray"
            legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=satame)
                                for satame, color in colors_satellite_with_cloudy.items()]
            ax.legend(handles=legend_handles, loc=loc_legend)

        ax.grid(axis="y")
        # ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_ylim(0, max_val_plot_th + 0.5)
        ax.set_ylabel("Estimated emissions (t/h)")
    
    if ax is not None:
        plot_subset(mlis_df, ax, True)
        return ax
    
    split_size = len(mlis_df) // nsplits
    
    # fig, ax = plt.subplots(2,1,figsize=(16,8),tight_layout=True)
    fig, axs = plt.subplots(nsplits, 1, figsize=(width,nsplits*height),
                            tight_layout=True)
    for i in range(nsplits):
        axiter = axs[i] if nsplits > 1 else axs
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < nsplits - 1 else len(mlis_df)
        plot_subset(mlis_df.iloc[start_idx:end_idx], axiter, i == nsplits-1)
    
    return fig, axs        
