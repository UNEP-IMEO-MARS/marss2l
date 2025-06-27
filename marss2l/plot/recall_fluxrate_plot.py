import matplotlib.gridspec as gridspec
from marss2l import loaders
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import recall_score
from marss2l.metrics import get_scenelevel_metrics
from marss2l.plot.colors import C0, C1, C2, C3, C4
import numpy as np

THRESHOLD_MARSS2L = 0.5
THRESHOLD_MBMP = -.985


from matplotlib.ticker import FuncFormatter

@FuncFormatter
def percentage_formatter(x, pos):
    return f"{x*100:.0f}%"

# fig = plt.figure(figsize=(10,2.5), layout="constrained")
# models_plot_recall = ["MBMP (baseline)", "CH4Net", "MARS-S2L (U326v309)"]

def fluxrate_to_str(fluxrate_th: float) -> str:
    """
    The string is shown without decimal places if it doesn't have decimal places and with 1 decimal place if it has.

    Args:
        fluxrate (float): fluxrate in k/h

    Returns:
        str: fluxrate in t/h as string
    """
    if abs(fluxrate_th - round(fluxrate_th)) < 0.05:
        return f"{int(fluxrate_th)}"
    else:
        return f"{fluxrate_th:.1f}"


INTERVALS_FLUXRATE_STR = []
for i, fluxrate in enumerate(loaders.INTERVALS_FLUXRATE[:-1]):
    if i == len(loaders.INTERVALS_FLUXRATE) - 2:
        INTERVALS_FLUXRATE_STR.append(f">{fluxrate_to_str(fluxrate)}")
    else:
        INTERVALS_FLUXRATE_STR.append(f"({fluxrate_to_str(fluxrate)}, {fluxrate_to_str(loaders.INTERVALS_FLUXRATE[i + 1])}]")


def plot_recall_fpr_fluxrate(detections_dataframe: pd.DataFrame, order_models: list[str] | None = None,
                             fig: plt.Figure | None = None, axs:list[plt.Axes] | None = None,
                             add_legend : bool = True, loc_legend: str = "lower right",
                             yticks_recall: list[float] | None = None,
                             cummulative:bool = True
                             ) -> tuple[plt.Figure, tuple[plt.Axes]]:
    """
    Plot the cumulative recall as a function of the flux rate and the barplot of the false positive rate (FPR) for each model.
    This is Figure 2 in the paper.

    Args:
        detections_dataframe (pd.DataFrame): Dataframe containing the detections with the following columns:
            - model_name: name of the model
            - isplume: 1 if the image has a plume, 0 otherwise
            - isplumeprednum: 1 is the model predicts a plume, 0 otherwise
            - ch4_fluxrate_th: flux rate of the plume of the image
        order_models (list[str] | None, optional): Models to plot in the order given. Defaults to None.
            If None, all models are plotted in the order they appear in the dataframe.
        fig (plt.Figure | None, optional): Figure to plot on. Defaults to None.
        cummulative (bool, optional): Whether to compute the cummulative recall or the recall for each flux rate. 
            Defaults to True (cummulative recall).
        axs (list[plt.Axes] | None, optional): Axes to plot on. Defaults to None.
            If None, a new figure and axes are created.
        add_legend (bool, optional): Whether to add the legend to the plot. Defaults to True.

    Returns:
        tuple[plt.Figure, tuple[plt.Axes]]: Figure and axes of the plot.
    """
    
    # Compute cummulative recall for each flux rate
    mets = []
    xticks = loaders.INTERVALS_FLUXRATE[1:-1]
    min_fluxrate = detections_dataframe.ch4_fluxrate_th[detections_dataframe.isplume].min()
    for fluxrateidx, flux_rate_thr in enumerate(xticks):
        for model, df_model in detections_dataframe.groupby("model_name"):
            
            fluxrate_thr_end = xticks[fluxrateidx+1] if fluxrateidx < len(xticks)-1 else loaders.INTERVALS_FLUXRATE[-1]
            if fluxrate_thr_end < min_fluxrate:
                continue
            
            if cummulative:
                dg = df_model[df_model.ch4_fluxrate_th > flux_rate_thr]
            else:
                dg = df_model[(df_model.ch4_fluxrate_th > flux_rate_thr) & (df_model.ch4_fluxrate_th <= fluxrate_thr_end)]
            
            if dg.shape[0] == 0:
                continue

            # add count of plumes in interval
            nplumes_interval = df_model.loc[(df_model.ch4_fluxrate_th > flux_rate_thr) & (df_model.ch4_fluxrate_th <= fluxrate_thr_end), "isplume"].sum()
            
            # add interval name
            flux_rate_thr_th = fluxrate_to_str(flux_rate_thr)
            if fluxrate_thr_end != loaders.INTERVALS_FLUXRATE[-1]:
                fluxrate_thr_end_th = fluxrate_to_str(fluxrate_thr_end)
                interval_str = f"({flux_rate_thr_th}, {fluxrate_thr_end_th}]"
            else:
                interval_str = f">{flux_rate_thr_th}"

            mets.append({
                    "recall": recall_score(dg.isplumenum, dg.isplumeprednum),
                    "nplumes_interval": nplumes_interval,
                    "interval_ch4_fluxrate_str": interval_str,
                    "nlocs": dg.location_name.nunique(),
                    "nplumes": dg.isplumenum.sum(),
                    "% plumes": dg.isplumenum.sum() / df_model.isplumenum.sum(),
                    "nnoplume": (1-dg.isplumenum).sum(),
                    "fluxrate": flux_rate_thr,
                    "fluxrateidx": fluxrateidx,
                    "model_name": model})

            
    mets = pd.DataFrame(mets)
    mets_recall_vs_fluxrate = mets.copy()

    # Compute overall metrics for FPR barplot
    mets = []
    for model, dg in detections_dataframe.groupby("model_name"):
        threshold = THRESHOLD_MARSS2L if not model.startswith("MBMP") else THRESHOLD_MBMP
        mets_iter = get_scenelevel_metrics(dg.scenepredcontinuous, dg.isplumenum, threshold=threshold,
                                           as_percentage=False)
        mets_iter.update({"nsamples": dg.shape[0],
                    "nlocs": dg.location_name.nunique(),
                    "nplumes": dg.isplumenum.sum(),
                    "nnoplume": (1-dg.isplumenum).sum(),
                    "model_name": model})
        mets.append(mets_iter)

    mets = pd.DataFrame(mets).sort_values(["balanced_accuracy"], ascending=False)
    overall_mets = mets[["model_name"]+[c for c in mets.columns if c != "model_name"]].copy()

    if order_models is None:
        order_models = overall_mets.model_name.unique().tolist()
    else:
        overall_mets = overall_mets[overall_mets.model_name.isin(order_models)].copy()
        # sort overall_mets by order_models
        overall_mets = overall_mets.set_index("model_name").loc[order_models].reset_index()
        mets_recall_vs_fluxrate = mets_recall_vs_fluxrate[mets_recall_vs_fluxrate.model_name.isin(order_models)].copy()

    # Create a GridSpec with 2 rows and 2 columns, specifying the width ratios

    if axs is None:
        if fig is None:
            fig = plt.figure(figsize=(10,2.5), layout="constrained")
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4], figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
    else:
        ax1, ax = axs

    # Plot the FPR
    ax1.bar(overall_mets.model_name, overall_mets.fpr, 
            color=[C0, C4, C2], 
            label=overall_mets.model_name, alpha=0.9)
    ax1.set_ylabel("FPR")
    ax1.yaxis.set_major_formatter(percentage_formatter)
    ax1.grid(axis="y")
    ax1.tick_params(axis='x', labelrotation=30)

    # Plot the histogram of flux rates
    axtwin = ax.twinx()

    df_plot_hist_fluxrate = mets_recall_vs_fluxrate.loc[mets_recall_vs_fluxrate.model_name == order_models[0],["interval_ch4_fluxrate_str", "nplumes_interval"]].copy()

    # Complement the intevals in loaders.INTERVALS_FLUXRATE[1:-1] with the intervals in df_plot_hist_fluxrate
    df_plot_bar = []
    for fluxrateinterval in INTERVALS_FLUXRATE_STR[1:]:
        if fluxrateinterval not in df_plot_hist_fluxrate.interval_ch4_fluxrate_str.values:
            df_plot_bar.append({"interval_ch4_fluxrate_str": fluxrateinterval, "nplumes_interval": 0})
        else:
            df_plot_bar.append(df_plot_hist_fluxrate[df_plot_hist_fluxrate.interval_ch4_fluxrate_str == fluxrateinterval].iloc[0].to_dict())
        
    df_plot_bar = pd.DataFrame(df_plot_bar)

    axtwin.bar(df_plot_bar.interval_ch4_fluxrate_str, 
               df_plot_bar.nplumes_interval,
               color=C1, alpha=0.7, fill=True)

    # Plot the cummulative recall    
    mets_show = mets_recall_vs_fluxrate[["interval_ch4_fluxrate_str","recall", "model_name"]].copy()
    sns.lineplot(mets_show, x="interval_ch4_fluxrate_str", 
                 y="recall", hue="model_name", ax=ax,
                 style="model_name",
                 palette=[C0, C4, C2],
                 markers=True,
                 dashes=False,
                 hue_order=order_models, 
                 legend=add_legend)
    # ax.set_ylim(0, 1.01)
    if yticks_recall is None:
        yticks_recall = np.arange(0, 1.1, 0.1)
    
    ax.set_yticks(yticks_recall,
                  labels=[f"{int(i*100)}%" for i in yticks_recall])
    
    if add_legend:
        sns.move_legend(ax, loc_legend, title="")
   
    # sns.scatterplot(mets_show, x="interval_ch4_fluxrate_str", y="recall",hue="model_name",
    #                 hue_order=order_models, legend=False,
    #                 ax=ax)
    
    # ax.yaxis.set_major_formatter(percentage_formatter)

    ax.tick_params(axis='x', labelrotation=30)

    ax.grid(axis="y")

    ax.set_xlabel("Flux rate (t/h)")
    if cummulative:
        ax.set_ylabel("Cumulative recall")
    else:
        ax.set_ylabel("Recall")
    
    axtwin.set_ylabel("# plumes")

    return fig, (ax1, ax) 