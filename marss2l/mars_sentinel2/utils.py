import numpy as np
from georeader.geotensor import GeoTensor
from georeader import read
from typing import List, Tuple, Union, Optional
    

def align_images(image:GeoTensor, bg:GeoTensor, 
                 validmask_bg:Optional[GeoTensor]=None,
                 corregister:bool=True,
                 rgb_bands:List[int]=[3,2,1],
                 max_translations:float=5) -> Union[GeoTensor, Tuple[GeoTensor, GeoTensor]]:
    """
    Aligns the background image to the input image.

    Args:
        image_s2 (GeoTensor): S2 image
        background_s2 (GeoTensor): S2 image to align
        validmask_bg (Optional[GeoTensor], optional): mask with valid values. Defaults to None.
        corregister (bool, optional): If True, it uses satalign to do corregister the images. Defaults to True.
        max_translations (float, optional): max translation allowed in pixels when corregistering. Defaults to 5.

    Returns:
        GeoTensor: aligned background image or a tuple with the aligned background image and the validmask_bg
    """
    if validmask_bg is not None:
        assert validmask_bg.same_extent(bg), "background_s2 and validmask must have the same shape"

    if not bg.same_extent(image):
        bg = read.read_reproject_like(bg, image)
        if validmask_bg is not None:
            validmask_bg = read.read_reproject_like(validmask_bg, image)
    
    if corregister:
        import satalign
        # Crop center is 80% of the minimum of the width or height
        crop_center = round(min(bg.shape[1], bg.shape[2]) * 0.8)
        syncmodel = satalign.PCC(
                datacube=bg.values[np.newaxis],
                reference=image.values.astype(np.float32),
                channel="mean", # mean of RGB
                rgb_bands=rgb_bands,
                crop_center=crop_center,
                max_translations=max_translations,
                # upsample_factor=10,
                num_threads=2)
        news2cube, warps = syncmodel.run_multicore()
        if validmask_bg is not None:
            validmask_bg_warped = syncmodel.warp_feature(img=validmask_bg.values[np.newaxis].astype(np.float32), 
                                                        warp_matrix=warps[0])[0] > 0.5
            validmask_bg = GeoTensor(validmask_bg_warped, validmask_bg.transform, validmask_bg.crs,
                                    fill_value_default=False)
        bg = GeoTensor(news2cube[0], bg.transform, bg.crs, fill_value_default=bg.fill_value_default)
    
    if validmask_bg is not None:
        bg.values[... ,~validmask_bg.values] = bg.fill_value_default
        return bg, validmask_bg
    
    return bg