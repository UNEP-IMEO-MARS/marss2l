import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from .colors import PALETTE_ALL, C0, C1, C2, C4


def plot_prob_vs_emission_rate(df_plot: pd.DataFrame, figsize:tuple[int,int]=(13, 7)) -> Tuple[plt.Figure, list[plt.Axes]]:
    """
    Creates a 2x2 grid of subplots to visualize how models probability scores 
    vary with plume detection status (isplume) and emission rate intervals.

    The figure comprises four subplots arranged as follows:
    1) ax1 (top-left): A boxplot of 'scenepredcontinuous' grouped by 'isplume'.
       - Shows how the models probability score differs for plumes vs no plumes.
    2) ax2 (top-right): A boxplot of 'scenepredcontinuous' grouped by emission-rate intervals ('interval_ch4_fluxrate_str').
       - Helps compare prediction score distributions across different emission-rate ranges.
    3) ax3 (bottom-left): A count plot of the number of images grouped by 'isplume'.
       - Displays how many images are labeled as plumes vs no plumes.
    4) ax4 (bottom-right): A count plot of the number of plumes grouped by emission-rate intervals ('interval_ch4_fluxrate_str').
       - Shows how many plumes fall into each emission-rate range.

    Expected columns in df_plot:
        - 'model_name': Name of the models making the predictions.
        - 'interval_ch4_fluxrate_str': Categorical string identifying the emission-rate interval.
        - 'scenepredcontinuous': Continuous probability (float) output by the models.
        - 'isplume': Boolean or equivalent indicating whether the image is labeled as a plume (True) or not (False).

    Args:
        df_plot (pd.DataFrame): The dataframe containing at least the columns described above.

    Returns:
        Tuple[plt.Figure, list[plt.Axes]]:
            - plt.Figure: The created figure.
            - list[plt.Axes]: A list of the four subplot axes [ax1, ax2, ax3, ax4].
    """
    import seaborn as sns
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=figsize, layout="constrained")

    # Create a GridSpec with 2 rows and 2 columns, specifying the width ratios
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], figure=fig)

    # Create subplots using the GridSpec and share the x-axis between specific subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Narrower subplot
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)  # Wider subplot
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Narrower subplot, sharing x-axis with ax1
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax2)  # Wider subplot, sharing x-axis with ax2

    # df_plot = outs_same_period_with_fluxrate.loc[outs_same_period_with_fluxrate.model_name.isin(model_names_plot)]
    model_names_plot = df_plot.model_name.unique().tolist()

    data = df_plot.loc[df_plot.interval_ch4_fluxrate_str != "[0, 0]"].copy()
    data["interval_ch4_fluxrate_str"] = data["interval_ch4_fluxrate_str"].cat.remove_categories("[0, 0]")
    df_plot_hist_fluxrate = data.loc[data.model_name == model_names_plot[0]].copy()

    sns.boxplot(data=data, x="interval_ch4_fluxrate_str", y="scenepredcontinuous", hue="model_name", ax=ax2,
                palette=PALETTE_ALL[:len(model_names_plot)])
    ax2.legend(loc="lower right")
    sns.countplot(data=df_plot_hist_fluxrate, 
                x="interval_ch4_fluxrate_str", ax=ax4,color=C1)

    df_plot_hist_fluxrate = df_plot.loc[df_plot.model_name == model_names_plot[0]].copy()

    sns.boxplot(data=df_plot, x="isplume", y="scenepredcontinuous", hue="model_name", ax=ax1,
                palette=PALETTE_ALL[:len(model_names_plot)], legend=False)
    # sns.violinplot(data=df_plot, x="isplume", y="scenepredcontinuous", hue="model_name", ax=ax1,palette=[C0,C2,C4,C1], legend=False,cut=0)
    sns.countplot(data=df_plot_hist_fluxrate, x="isplume", ax=ax3,color=C1)
    plt.xticks(rotation=30)# ax2.set_ylabel("model probability score")
    ax3.set_ylabel("# images")
    ax4.set_ylabel("# plumes")
    ax1.set_ylabel("model probability score")

    ax2.yaxis.set_visible(False)
    # ax4.yaxis.set_visible(False)
    ax1.xaxis.set_visible(False)
    ax2.xaxis.set_visible(False)

    ax4.set_xlabel("Flux rate (t/h)")
    ax3.set_xlabel("")
    # ax3.set_yticks(range(0, 13_100, 1_000))
    ax3.set_xticks([False, True],["no plume","plume"])

    return fig, [ax1, ax2, ax3, ax4]
