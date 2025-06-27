import torch
import numpy as np
from typing import Optional
from marss2l.mars_sentinel2.mixing_ratio_methane import FILL_VALUE_RATIO_IL

def to_mbmp(s2_data:torch.Tensor, b11_index:Optional[int]=None, 
            b12_index:Optional[int]=None, b11_index_prev:Optional[int]=None,
            b12_index_prev:Optional[int]=None) -> torch.Tensor:
    """
    Ratio to enhance methane of Irakullis-Loritxate et al 2022:
    (b12ch4 / b11ch4) / (b12noch4 / b11noch4) * np.median(b12noch4 / b11noch4) / np.median(b12ch4 / b11ch4)

    Args:
        s2_data (torch.Tensor): Sentinel-2 data with shape (n_bands, height, width).
        b11_index (Optional[int], optional): index of band 11. Defaults to None.
        b12_index (Optional[int], optional): index of band 12. Defaults to None.
        b11_index_prev (Optional[int], optional): index of band 11 of reference image. Defaults to None.
        b12_index_prev (Optional[int], optional): index of band 12 of reference image. Defaults to None.


    Returns:
        torch.Tensor: MBMP image with shape (height, width).
    """
    
    if b11_index is None or b12_index is None or b12_index_prev is None or b12_index_prev is None:
        raise ValueError("b11_index, b12_index, b11_index_prev and b12_index_prev must be provided")
        # if s2_data.shape[0]==26:
        #     b11_index = 11
        #     b12_index = 12
        #     b11_index_prev = b11_index+13
        #     b12_index_prev = b12_index+13
        # elif s2_data.shape[0]==16:
        #     b11_index = 6
        #     b12_index = 7
        #     b11_index_prev = b11_index+8
        #     b12_index_prev = b12_index+8
        # else:
        #     raise ValueError(f"Unknown number of bands {s2_data.shape[0]}")

    b11_prev = s2_data[b11_index_prev,...]
    b12_prev = s2_data[b12_index_prev,...]
    b11 = s2_data[b11_index,...]
    b12 = s2_data[b12_index,...]

    ratio = torch.full_like(b11, 1.0)
    b11_not_zero = b11 != 0
    ratio[b11_not_zero] = (b12[b11  != 0]/b11[b11_not_zero])
    ratio[b11_not_zero] = ratio[b11_not_zero] / (ratio[b11_not_zero].median())
    ratio = ratio.clamp(0, 10)

    ratio_prev = torch.full_like(b11_prev, 1.0)
    b11_prev_not_zero = b11_prev != 0
    ratio_prev[b11_prev_not_zero] = (b12_prev[b11_prev_not_zero]/b11_prev[b11_prev_not_zero])
    ratio_prev[b11_prev_not_zero] = ratio_prev[b11_prev_not_zero] / (ratio_prev[b11_prev_not_zero].median())
    ratio_prev = ratio_prev.clamp(0, 10)

    mbmp = ratio / ratio_prev
    mbmp[torch.isnan(mbmp)] = FILL_VALUE_RATIO_IL
    mbmp[torch.isinf(mbmp)] = FILL_VALUE_RATIO_IL
    mbmp[torch.abs(mbmp)>10] = 10

    return mbmp
