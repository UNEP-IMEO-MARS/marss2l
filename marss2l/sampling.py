from typing import Tuple, Union

import numpy as np
import pandas as pd
from rasterio.windows import Window

WINDOW_SIZE_TRAINING = 200
WINDOW_SIZE_DATA = 200


def get_window_from_item(
    item: Union[dict[str, int], pd.Series], include_source: bool = False
) -> Window:
    """
    Extract Window object from item dictionary containing window parameters.

    Args:
        item: Dictionary or Series containing window_col_off, window_row_off,
              window_width, and window_height keys.

    Returns:
        Window: rasterio Window object with the extracted parameters.
    """
    wd = Window(
        col_off=int(round(item["window_col_off"])),
        row_off=int(round(item["window_row_off"])),
        width=int(round(item["window_width"])),
        height=int(round(item["window_height"])),
    )
    if not include_source:
        return wd

    # Expand the window if item["pixel_row"] or item["pixel_col"] is outside the window
    pixel_row = int(round(item["pixel_row"]))
    pixel_col = int(round(item["pixel_col"]))

    if pixel_row < wd.row_off:
        wd = Window(col_off=wd.col_off, row_off=pixel_row, width=wd.width, height=wd.height)

    if pixel_col < wd.col_off:
        wd = Window(col_off=pixel_col, row_off=wd.row_off, width=wd.width, height=wd.height)

    if pixel_row >= wd.row_off + wd.height:
        wd = Window(
            col_off=wd.col_off,
            row_off=wd.row_off,
            width=wd.width,
            height=pixel_row - wd.row_off + 1,
        )
    if pixel_col >= wd.col_off + wd.width:
        wd = Window(
            col_off=wd.col_off,
            row_off=wd.row_off,
            width=pixel_col - wd.col_off + 1,
            height=wd.height,
        )
    return wd


def sample_window(
    plume_window: Window,
    window_size_training: int,
    window_size_data: int,
    add_jitter: bool = True,
) -> Tuple[int, int]:
    """
    Sample a window of size `window_size_training` surrounding the plume center. If add_jitter is True, the center
    of the window will be jittered by a random value between -window_size_training//4 and window_size_training//4

    Args:
        plume_window (Window): Window object containing plume location information (row_off, col_off, width, height)
        window_size_training (int, optional): Size of the training window. Defaults to WINDOW_SIZE_TRAINING.
        window_size_data (int, optional): Size of the data window. Defaults to WINDOW_SIZE_DATA.
        add_jitter (bool, optional): If True, the center of the window will be jittered. Defaults to True.

    Returns:
        Tuple[int, int]: row_off, col_off to sample the window
    """
    window_col_center_jitter = plume_window.col_off + plume_window.width // 2
    window_row_center_jitter = plume_window.row_off + plume_window.height // 2

    if add_jitter:
        window_col_center_jitter += np.random.randint(
            -plume_window.width // 4, plume_window.width // 4
        )
        window_row_center_jitter += np.random.randint(
            -plume_window.height // 4, plume_window.height // 4
        )

    window_row_off_jitter = window_row_center_jitter - window_size_training // 2
    window_col_off_jitter = window_col_center_jitter - window_size_training // 2

    # Make sure the window is inside the image
    window_row_off_jitter = min(
        max(0, window_row_off_jitter), window_size_data - window_size_training
    )
    window_col_off_jitter = min(
        max(0, window_col_off_jitter), window_size_data - window_size_training
    )

    return window_row_off_jitter, window_col_off_jitter
