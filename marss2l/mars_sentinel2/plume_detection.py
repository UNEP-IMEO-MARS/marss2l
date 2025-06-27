from typing import Union
from georeader.geotensor import GeoTensor
from numpy.typing import NDArray
import numpy as np
from skimage import measure

MINIMUM_NUMBER_PIXELS_PLUME = 150


def binary_connected_prediction(pred_continuous:Union[GeoTensor,NDArray],
                                threshold_prediction:float,
                                threshold_pixels:float=MINIMUM_NUMBER_PIXELS_PLUME) -> Union[GeoTensor,NDArray]:
    """
    Returns a binary prediction where the connected components with less than `threshold_pixels` pixels are removed.

    Args:
        pred_continuous (Union[GeoTensor,NDArray]): (H, W) or GeoTensor with float values (not necessarily between 0 and 1)
        threshold_prediction (float): threshold value for the prediction
        threshold_pixels (float, optional): Minimum number of pixels in the scene. Defaults to MINIMUM_NUMBER_PIXELS_PLUME.

    Returns:
        Union[GeoTensor,NDArray]: binary prediction of type uint8 where the connected components with less than `threshold_pixels` pixels are removed.
    """
    if isinstance(pred_continuous, GeoTensor):
        pred_continuous_values = pred_continuous.values
    else:
        pred_continuous_values = pred_continuous

    pred_discrete = (pred_continuous_values > threshold_prediction).astype(np.uint8)
    labels, nclusters = measure.label(pred_discrete, 
                                      connectivity=2, return_num=True)  # detect clusters and store their properties
    
    for cluster in range(1,nclusters+1):
        labels_cluster = labels==cluster
        if np.sum(labels_cluster) < threshold_pixels:
            labels[labels_cluster] = 0
        
    pred_values = (labels > 0).astype(np.uint8)
    if isinstance(pred_continuous, GeoTensor):
        pred = GeoTensor(pred_values, 
                         transform=pred_continuous.transform, 
                         crs=pred_continuous.crs,
                         fill_value_default=0)
    else:
        pred = pred_values
    
    return pred


def count_connected_pixels(pred_continuous:Union[GeoTensor,NDArray], 
                           threshold_prediction:float,
                           threshold_pixels:float=MINIMUM_NUMBER_PIXELS_PLUME) -> int:
    """
    Counts the number of connected components in the scene with values above `threshold_prediction`.    

    Args:
        pred_continuous (Union[GeoTensor,NDArray]): (H, W) or GeoTensor with float values (not necessarily between 0 and 1)
        threshold_prediction (float): threshold value for the prediction
        threshold_pixels (float, optional): Minimum number of pixels in the scene. Defaults to MINIMUM_NUMBER_PIXELS_PLUME.

    Returns:
        int: number of connected components in the scene with values above `threshold_prediction`
    """
    binary_pred = binary_connected_prediction(pred_continuous, threshold_prediction, threshold_pixels)
    if isinstance(binary_pred, GeoTensor):
        binary_pred_values = binary_pred.values
    else:
        binary_pred_values = binary_pred
    
    return int(np.sum(binary_pred_values))


def threshold_cutoff_connected_components(pred_continuous:Union[GeoTensor,NDArray], 
                                          threshold_pixels:float=MINIMUM_NUMBER_PIXELS_PLUME,
                                          tol:float=1e-3) -> float:
    """
    Implements binary search to find the continuous value that produces more than `threshold_pixels` pixels connected 
    in the scene.

    Args:
        pred_continuous (Union[GeoTensor,NDArray]): (H, W) or GeoTensor with float values (not necessarily between 0 and 1)
        threshold_pixels (float, optional): Minimum number of pixels in the scene. Defaults to MINIMUM_NUMBER_PIXELS_PLUME.
        tol (float, optional): Tolerance for the binary search. Defaults to 1e-3.

    Returns:
        scene_prob (float): minimum value such that sum(connected_components(pred_continuous >= scene_prob)) >= threshold_pixels
    """
    if isinstance(pred_continuous, GeoTensor):
        pred_continuous_values = pred_continuous.values
    else:
        pred_continuous_values = pred_continuous
    
    min_value = np.min(pred_continuous_values)
    max_value = np.max(pred_continuous_values)
    
    # binary search
    threshold = (min_value + max_value) / 2
    while (max_value - min_value) > tol:
        npixels_connected = count_connected_pixels(pred_continuous_values, threshold, 
                                                   threshold_pixels=threshold_pixels)
        if npixels_connected >= threshold_pixels:
            min_value = threshold
        else:
            max_value = threshold
        threshold = (min_value + max_value) / 2
    
    return threshold