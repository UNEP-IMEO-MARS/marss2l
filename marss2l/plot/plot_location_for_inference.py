import matplotlib.pyplot as plt
import numpy as np
from marss2l.loaders import DatasetPlumes
from marss2l.mars_sentinel2 import plume_detection, wind
from typing import Callable, Tuple, Dict
from numpy.typing import NDArray
import torch
from .geographical_location import plot_geographical_location
from georeader import plot
from datetime import datetime, timezone
from typing import Optional
from matplotlib.gridspec import GridSpec

def plot_location(dataset:DatasetPlumes, 
                  inference_function:Callable[[Dict[str, torch.Tensor]], NDArray], 
                  location_name:str, 
                  start_ind=0, n_panels:int=7, 
                  start_date:Optional[datetime]=None,
                  add_mbmp:bool=True,
                  add_geographical_location:bool=False,
                  add_colorbar_next_to:bool=False,
                  binary_pred:bool=False,
                  pred_with_rgb:bool=True, 
                  threshold_prediction:float=0.5) -> Tuple[plt.Figure, NDArray]:
    """
    Plot the images for a given location.
    The function will plot the DeltaXCH4, MBMP, label and predicted images for a given location.
    The function will also plot the wind vector in the MBMP image.

    Args:
        dataset (DatasetPlumes): dataset containing the images.
        inference_function (Callable[[Dict[str, torch.Tensor]], NDArray]): Inference function to get the predictions.
        location_name (str): Name of the location. It will fetch the images of the dataset 
            that are in the location.
        start_ind (int, optional): First index of the location to plot if `start_date` is not provided. 
            Defaults to 0.
        n_panels (int, optional): Last index of the location to plot. Defaults to 7.
            0 <= n_panels <= len(indexes_loc)
        start_date: (datetime, optional): Start date for the plot. 
            If None, it will use the `start_ind`.
        add_mbmp (bool, optional): If True, it will plot the MBMP image. Defaults to True.
        add_geographical_location (bool, optional): If True, it will plot the geographical location of the image
            in the first panel in the top left corner. Defaults to False.
        add_colorbar_next_to (bool, optional): If True, it will add a colorbar next to the last panel.
            Defaults to False.
        binary_pred (bool, optional): If True, it will plot the binary prediction. Defaults to False.
        pred_with_rgb (bool, optional): If True, it will plot the RGB image as background of the
            prediction. Defaults to True.
        threshold_prediction (float, optional): Threshold for the prediction. Defaults to 0.5.

    Returns:
        Tuple[plt.Figure, NDArray]: figure and axes with the plot.
        The axes will be a 4xN array, where N is the number of panels.
        The first row will be the DeltaXCH4 image, the second row will be the MBMP image,
        the third row will be the label image and the fourth row will be the predicted image.
    """
    # Compute indexes_plot
    if start_date is not None:
        # set tz to timezone.utc of start_date if None
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)

        bool_date = (dataset.dataframe.tile_date >= start_date) & \
                    (dataset.dataframe.location_name == location_name)
        if not bool_date.any():
            raise ValueError(f"Start date {start_date} not found in the dataset")
        
        #Items to plot
        items_plot = dataset.dataframe.loc[bool_date].sort_values("tile_date")

        indexes_plot = items_plot.index.tolist()
    else:
        items_plot = dataset.dataframe.loc[dataset.dataframe.location_name == location_name].sort_values("tile_date")
        if start_ind >= len(items_plot):
            raise ValueError(f"Start index {start_ind} out of range. Max index is {len(items_plot)-1}")
        indexes_plot = items_plot.index[start_ind:].tolist()
    
    if len(indexes_plot) < n_panels:
        n_panels = len(indexes_plot)
    
    nrows = 4 if add_mbmp else 3
    height = nrows * 2
    
    # Create figure with GridSpec for better control of layout
    fig = plt.figure(figsize=(n_panels*1.8 + (0.4 if add_colorbar_next_to else 0), height))
    
    # Set up GridSpec with an extra narrow column for colorbars if needed
    if add_colorbar_next_to:
        gs = GridSpec(nrows, n_panels+1, figure=fig, width_ratios=[1]*n_panels + [0.05])
    else:
        gs = GridSpec(nrows, n_panels, figure=fig)
    
    # Create the axes manually
    axs = np.empty((nrows, n_panels), dtype=object)
    for r in range(nrows):
        for c in range(n_panels):
            axs[r, c] = fig.add_subplot(gs[r, c])
            
    # Create colorbar axes if needed
    if add_colorbar_next_to:
        cbar_axs = [fig.add_subplot(gs[r, -1]) for r in range(nrows)]
    
    # Set up axes formatting
    for ax in axs.ravel():
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        
        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0][0].set_ylabel(r"$\Delta$XCH$_4$ (ppb)", fontsize=18)
    idxplot = 1
    if add_mbmp:
        axs[idxplot][0].set_ylabel("MBMP", fontsize=18)
        idxplot += 1

    axs[idxplot][0].set_ylabel("Label", fontsize=18)
    idxplot += 1
    axs[idxplot][0].set_ylabel("Predicted", fontsize=18)
    
    for count in range(n_panels):
        index = indexes_plot[count]
        item = dataset[index]
        date = dataset.dataframe.loc[index, "tile_date"]
        satellite = dataset.dataframe.loc[index, "satellite"]
        satellite = satellite.replace("LC0","L")
        preds = inference_function(item)
        
        pred_binary = plume_detection.binary_connected_prediction(preds, threshold_prediction=threshold_prediction)

        if binary_pred:
            preds = pred_binary
        else:
            preds[pred_binary == 0] = 0

        axs[0][count].set_title(f"{date.strftime("%Y-%m-%d")} {satellite}",  fontsize=15)
        axs[0][count].imshow(item['ch4'][0], cmap="plasma", vmin=0, vmax=2000)#, vmin=0, vmax=1)

        idxplot = 1
        if add_mbmp:
            axs[idxplot][count].imshow(item['mbmp'], cmap="magma_r") #, vmin=0, vmax=1)
            wind.add_wind_to_plot(item["wind"], ax=axs[idxplot][count])
            idxplot += 1
        else:
            # Add wind in CH4 image
            wind.add_wind_to_plot(item["wind"], ax=axs[0][count])
        
        axs[idxplot][count].imshow(item['y_target'], vmin=0, vmax=1, cmap="magma", interpolation="nearest")
        idxplot += 1
        
        if pred_with_rgb:
            input_data = item["y_context_ls0_0"]
            rgb = input_data[(3,2,1),...]
            rgb = torch.permute(rgb, (1, 2, 0))
            axs[idxplot][count].imshow(rgb.clip(0,1))

            # preds: convert to masked_array and mask values < threshold_prediction
            preds = np.ma.masked_where(preds < threshold_prediction, preds)
        
        axs[idxplot][count].imshow(preds, vmin=0, vmax=1, cmap="magma")

    # Add colorbar next to the last panel
    if add_colorbar_next_to:
        for row in range(nrows):
            # Create colorbar for the last image in each row
            im = axs[row, -1].images[-1]
            plt.colorbar(im, cax=cbar_axs[row])
        
    plt.tight_layout()
    return fig, axs
