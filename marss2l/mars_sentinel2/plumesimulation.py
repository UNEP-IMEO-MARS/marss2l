import numpy as np
from georeader.geotensor import GeoTensor

from . import transmittance_to_ch4
from typing import Union, Tuple, Optional, Dict
import scipy.ndimage as ndi
from skimage.transform import rotate
import numpy as np
from numpy.typing import NDArray, ArrayLike
import math


def rotate_wind_vector(wind_vector:ArrayLike, angle:float) -> NDArray:
    """
    Rotate a wind vector by a certain angle.

    Args:
        wind_vector (ArrayLike): Wind vector [U, V]
        angle (float): Angle in degrees.

    Returns:
        NDArray: Rotated wind vector
    """
    angle = math.radians(angle)
    wind_vector = np.array(wind_vector)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation_matrix.dot(wind_vector)

def counterclockwise_wind_angle(wind_vector_ch4:Tuple[float,float], wind_vector_image:Tuple[float,float]) -> float:
    """
    Calculate the angle in degrees from the wind vector of the plume to the wind vector of the image.

    https://stackoverflow.com/questions/14066933/direct-way-of-computing-the-clockwise-angle-between-two-vectors

    Args:
        wind_vector_ch4 (Tuple[float,float]): wind vector of the plume. [U, V]
        wind_vector_image (Tuple[float,float]): wind vector of the image. [U, V]

    Returns:
        float: Angle in degrees.
    """
    assert len(wind_vector_ch4) == 2, f"wind_vector_ch4 must be a 2D array, got {wind_vector_ch4.shape}"
    assert len(wind_vector_image) == 2, f"wind_vector_image must be a 2D array, got {wind_vector_image.shape}"
    wind_vector_ch4 = np.array(wind_vector_ch4)
    wind_vector_image = np.array(wind_vector_image)
    dot = np.dot(wind_vector_ch4, wind_vector_image)     # Dot product between [x1, y1] and [x2, y2]
    det = wind_vector_ch4[0]*wind_vector_image[1] - wind_vector_ch4[1]*wind_vector_image[0]      # Determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    # angle = math.acos(np.clip(wind_vector_ch4.dot(wind_vector_image) / np.linalg.norm(wind_vector_ch4) / np.linalg.norm(wind_vector_image), -1, 1))
    angle = angle / math.pi * 180
    return angle


def simulate_from_transmittance(image:Union[GeoTensor, NDArray], transmittance_b12_full:Union[GeoTensor, NDArray], 
                                transmittance_b11_full:Union[GeoTensor, NDArray], b11_index:int, b12_index:int) -> GeoTensor:
    """
    Simulate a plume in a multispectral image (S2 or Landsat) from the transmittance of the plume.

    Args:
        image (Union[GeoTensor, NDArray]): Multispectral image. 
            Expected to be a 3D array with the bands in the first dimension, dtype np.uint16 and units reflectance * 10000.
            (C, H', W').
        transmittance_b12_full (Union[GeoTensor, NDArray]): Transmittance of the B12 band in the simulated image.
        transmittance_b11_full (Union[GeoTensor, NDArray]): Transmittance of the B11 band in the simulated image.
        b11_index (int): Index of the B11 band in the image. 0 <= b11_index < C
        b12_index (int): Index of the B12 band in the image. 0 <= b11_index < C
    
    Returns:
        GeoTensor: Simulated image. a GeoTensor or NDArray with dtype numpy.uint16, units: reflectance * 10000
    """

    assert len(image.shape) == 3, f"image must be a 3D array, got {image.shape}"
    assert len(transmittance_b12_full.shape) == 2, f"transmittance_b12_full must be a 2D array, got {transmittance_b12_full.shape}"
    assert len(transmittance_b11_full.shape) == 2, f"transmittance_b11_full must be a 2D array, got {transmittance_b11_full.shape}"
    assert image.shape[-2:] == transmittance_b12_full.shape, f"transmittance_b12_full must have the same shape as the image, got {transmittance_b12_full.shape} and {image.shape}"
    assert image.shape[-2:] == transmittance_b11_full.shape, f"transmittance_b11_full must have the same shape as the image, got {transmittance_b11_full.shape} and {image.shape}"

    if isinstance(transmittance_b12_full, GeoTensor):
        transmittance_b12_full_values = transmittance_b12_full.values
        transmittance_b12_full_values[transmittance_b12_full_values == transmittance_b12_full.fill_value_default] = 1
    else:
        transmittance_b12_full_values = transmittance_b12_full

    if isinstance(transmittance_b11_full, GeoTensor):
        transmittance_b11_full_values = transmittance_b11_full.values
        transmittance_b11_full_values[transmittance_b11_full_values == transmittance_b11_full.fill_value_default] = 1
    else:
        transmittance_b11_full_values = transmittance_b11_full

    simulated_image = image.copy()
    if isinstance(image, GeoTensor):
        simulated_image.values[b12_index] = np.round(simulated_image.values[b12_index] *  transmittance_b12_full_values).astype(np.uint16)
        simulated_image.values[b11_index] = np.round(simulated_image.values[b11_index] *  transmittance_b11_full_values).astype(np.uint16)
    else:
        simulated_image[b12_index] = np.round(simulated_image[b12_index] *  transmittance_b12_full_values).astype(np.uint16)
        simulated_image[b11_index] = np.round(simulated_image[b11_index] *  transmittance_b11_full_values).astype(np.uint16)
    
    return simulated_image
       
                   
class PlumeSimulator:
    def __init__(self, transmittance_simulator:Optional[transmittance_to_ch4.TransmittanceCH4Interpolation]=None,
                 max_val_ch4_ppb:float=32_000):
        if transmittance_simulator is None:
            transmittance_simulator = transmittance_to_ch4.TransmittanceCH4InterpolationFromDict()
        self.transmittance_simulator = transmittance_simulator
        self.max_val_ch4_ppb = max_val_ch4_ppb
    
    def simulate_plume(self, ch4:NDArray, plume_mask:NDArray, wind_vector_ch4:Tuple[float,float], 
                        image:Union[GeoTensor, NDArray], b11_index:int, b12_index:int, satellite:str,
                        vza:float, sza:float, wind_vector_image:Tuple[float,float],
                        loc_injection:Optional[Tuple[int, int]]=None,
                        return_transmittance_and_ch4:bool=False) -> Dict[str, Union[NDArray, GeoTensor]]:
        """
        Simulate a plume in a multispectral image (S2 or Landsat)

        Args:
            ch4 (NDArray): Array with the methane concentration in ppb. (H, W) 
            plume_mask (NDArray): Binary mask with the plume. (H, W)
            wind_vector_ch4 (NDArray): [U, V] wind vector of the plume. (2, )
            image (Union[GeoTensor, NDArray]): Multispectral image. 
                Expected to be a 3D array with the bands in the first dimension, dtype np.uint16 and units reflectance * 10000.
                (C, H', W').
            b11_index (int): Index of the B11 band in the image. 0 <= b11_index < C
            b12_index (int): Index of the B12 band in the image. 0 <= b11_index < C
            satellite (str): Satellite name. Name of the satellite of the image. ["S2A", "S2B", "LC08", "LC09"]
            vza (float): view zenith angle of the image.
            sza (float): solar zenith angle of the image.
            wind_vector_image (NDArray): [U, V] wind vector of the image. (2, )
            loc_injection (Optional[Tuple[int, int]], optional): Location to inject the plume in the image (row, col). If None, it will be randomly selected. Defaults to None.
            return_transmittance_and_ch4 (bool, optional): If True, the transmittance and the ch4 simulated will be returned. Defaults to False.

        Returns:
            Dict[str, Union[NDArray, GeoTensor]]: Dictionary with:
                - image: Simulated image. GeoTensor or NDArray with dtype numpy.uint16, units: reflectance * 10000)
                - label: Plume mask in the simulated image.
                - window_row_off: Row offset of the plume in the simulated image.
                - window_col_off: Column offset of the plume in the simulated image.
                - window_width: Width of the plume in the simulated image.
                - window_height: Height of the plume in the simulated image
                - transmittance_b12 (optional): Transmittance of the B12 band in the simulated image.
                - transmittance_b11 (optional): Transmittance of the B11 band in the simulated image.
                - ch4 (optional): Simulated ch4 in the simulated image.
        
        """    
        assert ch4.shape == plume_mask.shape, f"ch4 and plume_mask must have the same shape, got {ch4.shape} and {plume_mask.shape}"
        assert len(ch4.shape) == 2, f"ch4 and plume_mask must be 2D arrays, got {ch4.shape} and {plume_mask.shape}"
        assert len(image.shape) == 3, f"image must be a 3D array, got {image.shape}"
        assert plume_mask.any(), "plume_mask must have at least one True value"
        
        # Smooth the ch4 values and restrinct them to the plume mask
        distances_plume = ndi.distance_transform_edt(plume_mask)
        mean_div = np.mean(distances_plume[plume_mask])
        if mean_div <= 1e-6:
            mean_div = 1
        distances_plume = np.clip(distances_plume/mean_div,0,1)
        
        ch4_simulate = ch4 * distances_plume

        # Set NaN values to 0
        ch4_simulate = np.nan_to_num(ch4_simulate, nan=0, posinf=self.max_val_ch4_ppb, 
                                     copy=False,
                                     neginf=0)
        # Clip values to 0-8000
        ch4_simulate = np.clip(ch4_simulate, 0, self.max_val_ch4_ppb)

        # Rotate the plume according to the wind
        angle = counterclockwise_wind_angle(wind_vector_ch4, wind_vector_image)
        
        if abs(angle)  > 1:
            ch4_simulate = rotate(ch4_simulate, angle=angle, resize=True, cval=0)
            plume_mask = rotate(plume_mask, angle=angle, resize=True, cval=False)
        
        transmittance_b12, transmittance_b11 = self.transmittance_simulator.transmittance_B12_B11(satellite, vza, sza, deltach4=ch4_simulate)
        
        plume_mask_full = np.zeros(image.shape[-2:],dtype=bool)

        if loc_injection is None:
            y_max = image.shape[-2] - transmittance_b11.shape[-2]
            if y_max <= 0:
                loc_injection_y = 0
            else:
                loc_injection_y = np.random.randint(0, y_max)
            
            x_max = image.shape[-1] - transmittance_b11.shape[-1]
            if x_max <= 0:
                loc_injection_x = 0
            else:
                loc_injection_x = np.random.randint(0, x_max)
        else:
            loc_injection_y, loc_injection_x = loc_injection
        
        loc_injection_y_end = min(loc_injection_y + transmittance_b11.shape[-2], image.shape[-2])
        loc_injection_x_end = min(loc_injection_x + transmittance_b11.shape[-1], image.shape[-1])
        if (loc_injection_y_end - loc_injection_y) != transmittance_b11.shape[-2]:
            slice_transmittance_y = slice(0, loc_injection_y_end - loc_injection_y)
        else:
            slice_transmittance_y = slice(None)
        
        if (loc_injection_x_end - loc_injection_x) != transmittance_b11.shape[-1]:
            slice_transmittance_x = slice(0, loc_injection_x_end - loc_injection_x)
        else:
            slice_transmittance_x = slice(None)
        
        plume_mask_full[loc_injection_y:loc_injection_y_end, 
                        loc_injection_x:loc_injection_x_end] =  plume_mask[slice_transmittance_y, slice_transmittance_x]
        
        transmittance_b12_full = np.ones(image.shape[-2:], dtype=transmittance_b12.dtype)
        transmittance_b11_full = np.ones(image.shape[-2:], dtype=transmittance_b11.dtype)
        
        transmittance_b12_full[loc_injection_y:loc_injection_y_end, 
                            loc_injection_x:loc_injection_x_end] =  transmittance_b12[slice_transmittance_y, slice_transmittance_x]
        transmittance_b11_full[loc_injection_y:loc_injection_y_end, 
                            loc_injection_x:loc_injection_x_end] =  transmittance_b11[slice_transmittance_y, slice_transmittance_x]
        
        if return_transmittance_and_ch4:
            ch4_simulate_full = np.zeros(image.shape[-2:],dtype=ch4_simulate.dtype)
            ch4_simulate_full[loc_injection_y:loc_injection_y_end, 
                            loc_injection_x:loc_injection_x_end] =  ch4_simulate[slice_transmittance_y, slice_transmittance_x]
        
        simulated_image = simulate_from_transmittance(image, transmittance_b12_full, transmittance_b11_full, b11_index, b12_index)

        if isinstance(image, GeoTensor):
            plume_mask_full = GeoTensor(plume_mask_full, image.transform, image.crs, 
                                        fill_value_default=False)
        
        out = {"image": simulated_image, 
            "label": plume_mask_full,
            "window_row_off": loc_injection_y, 
            "window_col_off": loc_injection_x,
            "window_width": loc_injection_x_end - loc_injection_x,
            "window_height": loc_injection_y_end - loc_injection_y}
        
        if return_transmittance_and_ch4:
            if isinstance(image, GeoTensor):
                ch4_simulate_full = GeoTensor(ch4_simulate_full, image.transform, image.crs, fill_value_default=0)
                transmittance_b12_full = GeoTensor(transmittance_b12_full, image.transform, image.crs, fill_value_default=1)
                transmittance_b11_full = GeoTensor(transmittance_b11_full, image.transform, image.crs, fill_value_default=1)
        
            out["transmittance_b12"] = transmittance_b12_full
            out["transmittance_b11"] = transmittance_b11_full
            out["ch4"] = ch4_simulate_full

        return out