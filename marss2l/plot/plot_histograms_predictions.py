import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_row(data, set_ylabel:bool=False, model_name:str="MARS S2L", 
             max_plumes_hist:int=None, 
             max_images_hist:int=None, axs:list[plt.Axes]=None,
             thrreshold_pred:float=0.5,
             add_text:bool=True) -> list[plt.Axes]:
    """
    Histogram plots for the case studies.

    Args:
        data: pandas DataFrame containing the predictions and targets with columns 'scene_pred' and 'target'
        set_ylabel: bool, set to True to set the y-labels
        model_name: str, name of the model
        max_plumes_hist: int, maximum number of plumes to show in the histogram
        max_images_hist: int, maximum number of images to show in the histogram
        axs: plt.Axes, axes to plot the histograms on. If None, a new figure is created.
        thrreshold_pred: float, threshold for the predictions. Default is 0.5.
        add_text: bool, set to True to add text to the plot
    

    """
    if axs is None:
        fig, axs = plt.subplots(2,1, figsize=(6,8), sharex=True, sharey='row')
    else:
        assert len(axs)==2, "axs must be a list of two axes"
    
    for ax in axs.ravel():
        ax.xaxis.set_major_locator(plt.MaxNLocator(8)) # Fewer major gridlines on x-axis
        ax.yaxis.set_major_locator(plt.MaxNLocator(8)) # Fewer major gridlines on y-axis
    
    axs[1].grid("on", zorder=1)
    axs[0].grid("on", zorder=1)
    nimages, bins, patches = axs[0].hist(data.scene_pred[data.target==0], zorder=2, alpha=0.8, linewidth=1.5, edgecolor="black", bins=10)
    for p in patches[:len(bins)//2]:
        p.set_facecolor('#648FFF')
    for p in patches[len(bins)//2:]:
        p.set_facecolor('#DC267F')
    
    #axs[0][row].set_xlabel("Predicted probability")
    axs[0].set_title(f"{model_name} predictions (no plume)", fontsize=18)
    nplumes, bins, patches = axs[1].hist(data.scene_pred[data.target==1], zorder=2, alpha=0.8, linewidth=1.5, edgecolor="black", bins=10)
    for p in patches[len(bins)//2:]:
        p.set_facecolor('#648FFF')
    for p in patches[:len(bins)//2]:
        p.set_facecolor('#DC267F')
    if set_ylabel:
        axs[0].set_ylabel("Number of images", fontsize=16)
        axs[1].set_ylabel("Number of images", fontsize=16)
    axs[1].set_xlabel("Predicted probability", fontsize=16)
    axs[1].set_title(f"{model_name} predictions (plume)", fontsize=18)
    
    if max_images_hist is None:
        max_images_hist = max(nimages)
    if max_plumes_hist is None:
        max_plumes_hist = max(nplumes)

    axs[0].set_ylim([0, max_images_hist])
    axs[1].set_ylim([0, max_plumes_hist])

    axs[0].axvline(x=thrreshold_pred, color='black', linestyle='--')
    axs[1].axvline(x=thrreshold_pred, color='black', linestyle='--')

    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()

    # Add the text 
    if add_text:
        TP = len(data[np.logical_and(data.target==1, data.scene_pred>=thrreshold_pred)])
        TN = len(data[np.logical_and(data.target==0, data.scene_pred<thrreshold_pred)])
        FP = len(data[np.logical_and(data.target==0, data.scene_pred>=thrreshold_pred)])
        FN = len(data[np.logical_and(data.target==1, data.scene_pred<thrreshold_pred)])

        axs[0].text(0.12, max_images_hist-np.rint(max_images_hist*0.1), f"True negatives = {TN}", fontsize=12)
        axs[0].text(0.55, max_images_hist-np.rint(max_images_hist*0.1), f"False positives = {FP}", fontsize=12)

        axs[1].text(0.12, max_plumes_hist-np.rint(max_plumes_hist*0.1), f"False negatives = {FN}", fontsize=12)
        axs[1].text(0.55, max_plumes_hist-np.rint(max_plumes_hist*0.1), f"True positives = {TP}", fontsize=12)

    return axs
