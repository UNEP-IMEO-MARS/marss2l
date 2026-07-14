import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
from georeader import get_utm_epsg, rasterio_crs, read
from georeader.geotensor import GeoTensor
from numpy.typing import ArrayLike, NDArray
from rasterio.warp import Resampling, transform_geom
from shapely.geometry import Point, mapping, shape
from skimage.transform import SimilarityTransform, warp

from marss2l.mars_sentinel2 import transmittance_to_ch4


def rotate(
    image: NDArray,
    angle: float,
    resize: bool = False,
    center: Optional[Tuple[int, int]] = None,
    order=None,
    mode="constant",
    cval=0,
    clip=True,
    preserve_range=False,
):
    """Rotate image by a certain angle around its center.

    This function is copied from `skimage.transform.rotate` and
    modified to return the new center of the image after the rotation.

    Parameters
    ----------
    image : ndarray
        Input image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    resize : bool, optional
        Determine whether the shape of the output image will be automatically
        calculated, so the complete rotated image exactly fits. Default is
        False.
    center : iterable of length 2
        The rotation center. If ``center=None``, the image is rotated around
        its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
        note that this parameter is (cols, rows), contrary to normal skimage
        ordering.

    Returns
    -------
    rotated : ndarray
        Rotated version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import rotate
    >>> image = data.camera()
    >>> rotate(image, 2).shape
    (512, 512)
    >>> rotate(image, 2, resize=True).shape
    (530, 530)
    >>> rotate(image, 90, resize=True).shape
    (512, 512)

    """

    rows, cols = image.shape[0], image.shape[1]

    if image.dtype == np.float16:
        image = image.astype(np.float32)

    # rotation around center
    if center is None:
        center = np.array((cols, rows)) / 2.0 - 0.5
    else:
        center = np.asarray(center)
    tform1 = SimilarityTransform(translation=center)
    tform2 = SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    output_shape = None
    if resize:
        # determine shape of output image
        corners = np.array([[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform

    # Make sure the transform is exactly affine, to ensure fast warping.
    tform.params[2] = (0, 0, 1)

    # New location of the center of the image
    center_new = tform.inverse(np.array(center).reshape((1, 2)))[0]

    return (
        warp(
            image,
            tform,
            output_shape=output_shape,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
        ),
        center_new,
    )


def rotate_wind_vector(wind_vector: ArrayLike, angle: float) -> NDArray:
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


def counterclockwise_wind_angle(
    wind_vector_ch4: Tuple[float, float], wind_vector_image: Tuple[float, float]
) -> float:
    """
    Calculate the angle in degrees from the wind vector of the plume to the wind vector of the image.

    https://stackoverflow.com/questions/14066933/direct-way-of-computing-the-clockwise-angle-between-two-vectors

    Args:
        wind_vector_ch4 (Tuple[float,float]): wind vector of the plume. [U, V]
        wind_vector_image (Tuple[float,float]): wind vector of the image. [U, V]

    Returns:
        float: Angle in degrees.
    """
    assert (
        len(wind_vector_ch4) == 2
    ), f"wind_vector_ch4 must be a 2D array, got {wind_vector_ch4.shape}"
    assert (
        len(wind_vector_image) == 2
    ), f"wind_vector_image must be a 2D array, got {wind_vector_image.shape}"
    wind_vector_ch4 = np.array(wind_vector_ch4)
    wind_vector_image = np.array(wind_vector_image)
    dot = np.dot(wind_vector_ch4, wind_vector_image)  # Dot product between [x1, y1] and [x2, y2]
    det = (
        wind_vector_ch4[0] * wind_vector_image[1] - wind_vector_ch4[1] * wind_vector_image[0]
    )  # Determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    # angle = math.acos(np.clip(wind_vector_ch4.dot(wind_vector_image) / np.linalg.norm(wind_vector_ch4) / np.linalg.norm(wind_vector_image), -1, 1))
    angle = angle / math.pi * 180
    return angle


def rotate_pixel_coordinates(
    row: Union[int, float, NDArray],
    col: Union[int, float, NDArray],
    center: tuple[int, int],
    angle: float,
) -> Union[Tuple[int, int], Tuple[NDArray, NDArray]]:
    """
    Rotate pixel coordinates around a center by a certain angle (counter-clockwise).

    This function applies a 2D rotation transformation to pixel coordinates. The rotation
    is performed counter-clockwise around the specified center point.

    The transformation follows these steps:
    1. Translate coordinates so center becomes origin: (col', row') = (col - center_col, row - center_row)
    2. Apply rotation matrix:
       [col_rotated]   [cos(θ)  -sin(θ)] [col']
       [row_rotated] = [sin(θ)   cos(θ)] [row']
    3. Translate back: (col_final, row_final) = (col_rotated + center_col, row_rotated + center_row)

    Note: In image coordinates, rows increase downward and columns increase rightward.
    The rotation is counter-clockwise in the mathematical sense (standard rotation matrix).

    Args:
        row (Union[int, float, NDArray]): Row coordinate(s). Can be a single value or array.
        col (Union[int, float, NDArray]): Column coordinate(s). Can be a single value or array.
        center (tuple[int, int]): Center of rotation (row_center, col_center).
        angle (float): Rotation angle in degrees (counter-clockwise).

    Returns:
        Union[Tuple[int, int], Tuple[NDArray, NDArray]]: Rotated pixel coordinates.
            - If inputs are scalars: (row_rotated, col_rotated) as integers
            - If inputs are arrays: (row_rotated_array, col_rotated_array) as NDArrays

    Examples:
        >>> # Rotate a single point 90 degrees counter-clockwise around (100, 100)
        >>> rotate_pixel_coordinates(100, 150, (100, 100), 90)
        (150, 100)

        >>> # Rotate multiple points
        >>> rows = np.array([100, 110, 120])
        >>> cols = np.array([150, 150, 150])
        >>> rotate_pixel_coordinates(rows, cols, (100, 100), 90)
        (array([150, 150, 150]), array([100, 90, 80]))
    """
    angle_rad = math.radians(angle)
    row_center, col_center = center

    # Convert to numpy arrays for vectorized operations
    row_arr = np.atleast_1d(row)
    col_arr = np.atleast_1d(col)

    # Step 1: Translate so center becomes origin
    col_centered = col_arr - col_center
    row_centered = row_arr - row_center

    # Step 2: Apply 2D rotation matrix
    # Standard rotation matrix (counter-clockwise):
    # [cos(θ)  -sin(θ)]
    # [sin(θ)   cos(θ)]
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    col_rotated = cos_angle * col_centered - sin_angle * row_centered
    row_rotated = sin_angle * col_centered + cos_angle * row_centered

    # Step 3: Translate back to original coordinate system
    col_final = col_rotated + col_center
    row_final = row_rotated + row_center

    # Return appropriate type based on input
    if np.isscalar(row) and np.isscalar(col):
        return int(np.round(row_final[0])), int(np.round(col_final[0]))
    else:
        return row_final, col_final


def simulate_from_transmittance(
    image: Union[GeoTensor, NDArray],
    transmittance_b12_full: Union[GeoTensor, NDArray],
    transmittance_b11_full: Union[GeoTensor, NDArray],
    b11_index: int,
    b12_index: int,
) -> GeoTensor:
    r"""
    Simulate a plume in a multispectral image (S2 or Landsat) from the transmittance of the plume.

    Args:
        image (Union[GeoTensor, NDArray]): Multispectral image.
            Expected to be a 3D array with the bands in the first dimension, dtype np.uint16 and units reflectance * 10000.
            (C, H', W').
        transmittance_b12_full (Union[GeoTensor, NDArray]): Transmittance of the B12 band in the simulated image.
        transmittance_b11_full (Union[GeoTensor, NDArray]): Transmittance of the B11 band in the simulated image.
        b11_index (int): Index of the B11 band in the image. 0 \leq b11_index < C
        b12_index (int): Index of the B12 band in the image. 0 \leq b11_index < C

    Returns:
        GeoTensor: Simulated image. a GeoTensor or NDArray with dtype numpy.uint16, units: reflectance * 10000
    """

    assert len(image.shape) == 3, f"image must be a 3D array, got {image.shape}"
    assert (
        len(transmittance_b12_full.shape) == 2
    ), f"transmittance_b12_full must be a 2D array, got {transmittance_b12_full.shape}"
    assert (
        len(transmittance_b11_full.shape) == 2
    ), f"transmittance_b11_full must be a 2D array, got {transmittance_b11_full.shape}"
    assert (
        image.shape[-2:] == transmittance_b12_full.shape
    ), f"transmittance_b12_full must have the same shape as the image, got {transmittance_b12_full.shape} and {image.shape}"
    assert (
        image.shape[-2:] == transmittance_b11_full.shape
    ), f"transmittance_b11_full must have the same shape as the image, got {transmittance_b11_full.shape} and {image.shape}"

    if isinstance(transmittance_b12_full, GeoTensor):
        transmittance_b12_full_values = transmittance_b12_full.values
        transmittance_b12_full_values[
            transmittance_b12_full_values == transmittance_b12_full.fill_value_default
        ] = 1
    else:
        transmittance_b12_full_values = transmittance_b12_full

    if isinstance(transmittance_b11_full, GeoTensor):
        transmittance_b11_full_values = transmittance_b11_full.values
        transmittance_b11_full_values[
            transmittance_b11_full_values == transmittance_b11_full.fill_value_default
        ] = 1
    else:
        transmittance_b11_full_values = transmittance_b11_full

    simulated_image = image.copy()
    if isinstance(image, GeoTensor):
        simulated_image.values[b12_index] = np.round(
            simulated_image.values[b12_index] * transmittance_b12_full_values
        ).astype(np.uint16)
        simulated_image.values[b11_index] = np.round(
            simulated_image.values[b11_index] * transmittance_b11_full_values
        ).astype(np.uint16)
    else:
        simulated_image[b12_index] = np.round(
            simulated_image[b12_index] * transmittance_b12_full_values
        ).astype(np.uint16)
        simulated_image[b11_index] = np.round(
            simulated_image[b11_index] * transmittance_b11_full_values
        ).astype(np.uint16)

    return simulated_image


def calculate_injection_slices(
    image_shape: Tuple[int, int],
    plume_shape: Tuple[int, int],
    upper_left_coord: Tuple[int, int],
) -> Tuple[Tuple[slice, slice], Tuple[slice, slice]]:
    """
    I want to inject the plume image into the image at the location upper_left_coord. This function
    returns the slices for the plume and the image that will be used to inject the plume.

    Args:
        image_shape (Tuple[int, int]): Shape of the image (H, W).
        plume_shape (Tuple[int, int]): Shape of the plume (H', W').
        upper_left_coord (Tuple[int, int]): top left coordinate for the image injection (row, col). It could be
            outside the image bounds.

    Returns:
        Tuple[Tuple[slice, slice], Tuple[slice, slice]]: Tuple with the slices for the plume and the image.
    """
    assert (
        upper_left_coord[0] > -plume_shape[0] and upper_left_coord[1] > -plume_shape[1]
    ), f"upper_left_coord must be within the plume bounds, got {upper_left_coord} and plume shape {plume_shape}"

    assert (
        upper_left_coord[0] < image_shape[0] and upper_left_coord[1] < image_shape[1]
    ), f"upper_left_coord must be within the image bounds, got {upper_left_coord} and image shape {image_shape}"

    # Calculate the start and end positions for the image slice
    start_row_img = max(upper_left_coord[0], 0)
    start_col_img = max(upper_left_coord[1], 0)
    end_row_img = min(upper_left_coord[0] + plume_shape[0], image_shape[0])
    end_col_img = min(upper_left_coord[1] + plume_shape[1], image_shape[1])

    # Calculate the start and end positions for the plume slice
    start_row_plume = max(0, -upper_left_coord[0])
    start_col_plume = max(0, -upper_left_coord[1])
    end_row_plume = start_row_plume + (end_row_img - start_row_img)
    end_col_plume = start_col_plume + (end_col_img - start_col_img)

    # Create the slices for the image and the plume
    image_slice = (slice(start_row_img, end_row_img), slice(start_col_img, end_col_img))
    plume_slice = (
        slice(start_row_plume, end_row_plume),
        slice(start_col_plume, end_col_plume),
    )

    return image_slice, plume_slice


def sample_loc_injection(
    image_shape: Tuple[int, int], padding_margin: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Sample a random location to inject the plume in the image.

    Args:
        image_shape (Tuple[int, int]): Shape of the image (height, width).
        padding_margin (Tuple[int, int]): Margin to avoid the borders of the image. (padding_row, padding_col)

    Returns:
        Tuple[int, int]: Location to inject the plume in the image (row, col).
    """

    return (
        np.random.randint(padding_margin[0], image_shape[0] - padding_margin[0]),
        np.random.randint(padding_margin[1], image_shape[1] - padding_margin[1]),
    )


def compute_upper_left_injection_coordinate(
    loc_injection: Tuple[int, int],
    plume_source: Optional[Tuple[int, int]] = None,
    plume_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Compute the upper left coordinate to inject the plume in the image.

    Args:
        image_shape (Tuple[int, int]): Shape of the image (height, width).
        plume_shape (Tuple[int, int]): Shape of the plume (height', width').
        loc_injection (Optional[Tuple[int, int]]): Location to inject the plume in the image (row, col).
            If None, it will be randomly selected.
        plume_source (Optional[Tuple[int, int]], optional): Location of the plume source relative to plume_shape (row, col).
            If provided the plume source will be aligned with the loc_injection. Defaults to None.

    Returns:
      Tuple[int, int]: top left coordinate for the image injection (row, col).
    """
    # The upper left coordinate for the injection is the loc_injection minus the plume source
    if plume_source is None:
        assert plume_shape is not None, "If plume_source is None, plume_shape must be provided"
        plume_source = plume_shape[0] // 2, plume_shape[1] // 2

    return loc_injection[0] - plume_source[0], loc_injection[1] - plume_source[1]


def point_to_pixel_coords(point: Point, geotensor: GeoTensor) -> Tuple[int, int]:
    """
    Convert a point in EPSG:4326 to pixel coordinates in the geotensor.

    Args:
        point (Point): Point in EPSG:4326.
        geotensor (GeoTensor): GeoTensor with the `crs`and `transform` attributes.

    Returns:
        Tuple[int, int]: Pixel coordinates in the geotensor (row, col).
    """
    point_crs = shape(transform_geom("EPSG:4326", geotensor.crs, mapping(point)))
    point_crs_coords = point_crs.coords[0]  # (x, y)
    transform_inv = ~geotensor.transform
    pixel_coords = transform_inv * point_crs_coords
    return int(round(pixel_coords[1])), int(round(pixel_coords[0]))


class PlumeSimulator:
    def __init__(
        self,
        transmittance_simulator: Optional[
            transmittance_to_ch4.TransmittanceCH4Interpolation
        ] = None,
        padding_sample_loc_injection: Tuple[int, int] = (20, 20),
        max_val_ch4_ppb: Optional[float] = 16_000,
    ):
        if transmittance_simulator is None:
            transmittance_simulator = transmittance_to_ch4.TransmittanceCH4InterpolationFromDict()
        self.transmittance_simulator = transmittance_simulator
        self.padding_sample_loc_injection = padding_sample_loc_injection
        self.max_val_ch4_ppb = max_val_ch4_ppb

    def simulate_plume(
        self,
        ch4: Union[GeoTensor, NDArray],
        plume_mask: Union[GeoTensor, NDArray],
        wind_vector_ch4: Tuple[float, float],
        image: Union[GeoTensor, NDArray],
        b11_index: int,
        b12_index: int,
        satellite: str,
        vza: float,
        sza: float,
        wind_vector_image: Tuple[float, float],
        loc_injection: Optional[Union[Tuple[int, int], Point]] = None,
        plume_source: Optional[Union[Tuple[int, int], Point]] = None,
        smooth_ch4: bool = True,
        units_ch4: str = "ppb",
        return_transmittance_and_ch4: bool = False,
    ) -> Dict[str, Union[NDArray, GeoTensor]]:
        r"""
        Simulate a plume in a multispectral image (S2 or Landsat)

        Args:
            ch4 (Union[GeoTensor, NDArray]): Array with the methane concentration in ppb. (H, W) If GeoTensor
                it will resize to the same spatial resolution as the image (and it will assert that the
                image is a GeoTensor too).
            plume_mask (Union[GeoTensor, NDArray]): Binary mask with the plume. (H, W). If GeoTensor
                it will resize to the same spatial resolution as the image (and it will assert that the
                image is a GeoTensor too).
            wind_vector_ch4 (NDArray): [U, V] wind vector of the plume. (2, )
            image (Union[GeoTensor, NDArray]): Multispectral image.
                Expected to be a 3D array with the bands in the first dimension, dtype np.uint16 and units reflectance * 10000.
                (C, H', W').
            b11_index (int): Index of the B11 band in the image. 0 \leq b11_index < C
            b12_index (int): Index of the B12 band in the image. 0 \leq b11_index < C
            satellite (str): Satellite name. Name of the satellite of the image. ["S2A", "S2B", "LC08", "LC09", "LE07", "LT05", "LT04"]
            vza (float): view zenith angle of the image.
            sza (float): solar zenith angle of the image.
            wind_vector_image (NDArray): [U, V] wind vector of the image. (2, )
            loc_injection (Optional[Union[Tuple[int, int], Point]], optional):
                Location to inject the source of plume in the image. If Point it assumes the coordinates are EPSG:4326. If Tuple
                it assumes the coordinates are in pixel coordinates (row, col).
                If None, it will be randomly selected. Defaults to None.
            plume_source (Optional[Union[Tuple[int, int], Point]], optional):
                Location of the plume source in the `ch4` and `plume_mask` images. If Point it assumes the coordinates are EPSG:4326. If Tuple
                it assumes the coordinates are in pixel coordinates (row, col) from the plume mask.
                If provided, the plume source will be aligned with the `loc_injection` in the simulated image. Defaults to None.
            smooth_ch4 (bool, optional): If True, the ch4 values will be smoothed on the edges. Defaults to True.
                This is useful to avoid sharp edges in the plume, specially if the ch4 values are noisy. Set to False
                if the ch4 values come from a simulation or from a sensor with low noise.
            units_ch4 (str, optional): Units of the delta xch4 values. Defaults to "ppb".
            return_transmittance_and_ch4 (bool, optional): If True, the transmittance and the ch4 simulated will be returned. Defaults to False.

        Returns:
            Dict[str, Union[NDArray, GeoTensor]]: Dictionary with:
                - image: Simulated image. GeoTensor or NDArray with dtype numpy.uint16, units: reflectance * 10000)
                - label: Plume mask in the simulated image.
                - window_row_off: Row offset of the plume in the simulated image.
                - window_col_off: Column offset of the plume in the simulated image.
                - window_width: Width of the plume in the simulated image.
                - window_height: Height of the plume in the simulated image
                - source_row (optional): Row of the plume source in the simulated image.
                - source_col (optional): Column of the plume source in the simulated image.
                - transmittance_b12 (optional): Transmittance of the B12 band in the simulated image.
                - transmittance_b11 (optional): Transmittance of the B11 band in the simulated image.
                - ch4 (optional): Simulated ch4 in the simulated image.

        """
        assert (
            ch4.shape == plume_mask.shape
        ), f"ch4 and plume_mask must have the same shape, got {ch4.shape} and {plume_mask.shape}"
        assert (
            len(ch4.shape) == 2
        ), f"ch4 and plume_mask must be 2D arrays, got {ch4.shape} and {plume_mask.shape}"
        assert len(image.shape) == 3, f"image must be a 3D array, got {image.shape}"
        assert (
            b11_index >= 0 and b11_index < image.shape[0]
        ), f"b11_index must be between 0 and {image.shape[0]}, got {b11_index}"
        assert (
            b12_index >= 0 and b12_index < image.shape[0]
        ), f"b12_index must be between 0 and {image.shape[0]}, got {b12_index}"

        # If the ch4 and plume_mask are GeoTensors make sure they have the same spatial resolution as the image
        if isinstance(ch4, GeoTensor) or isinstance(plume_mask, GeoTensor):
            # Assert both ch4 and plume_mask are GeoTensors
            assert isinstance(
                ch4, GeoTensor
            ), "If ch4 is a GeoTensor, plume_mask must be a GeoTensor too"
            assert isinstance(
                plume_mask, GeoTensor
            ), "If plume_mask is a GeoTensor, ch4 must be a GeoTensor too"
            assert isinstance(
                image, GeoTensor
            ), "If ch4 is a GeoTensor, image must be a GeoTensor too"
            assert ch4.same_extent(plume_mask), "ch4 and plume_mask must have the same extent"

            # Make sure the image is in a projected CRS (UTM)
            crs_image = rasterio_crs(image.crs)
            if not crs_image.is_projected:
                raise ValueError(f"Image must be in a projected CRS (i.e. UTM) found {crs_image}")

            # reproject CH4 to UTM with same spatial resolution as the image
            crs = rasterio_crs(ch4.crs)

            if not crs.is_projected:
                crs_dst = get_utm_epsg(ch4.footprint(crs="EPSG:4326").centroid.coords[0])
                ch4 = read.read_to_crs(ch4, crs_dst, resolution_dst_crs=image.res)
                plume_mask = read.read_reproject_like(
                    plume_mask, ch4, resampling=Resampling.nearest
                )
            elif ch4.res != image.res:
                # Resize ch4 and plume_mask to the same resolution as the image
                # Do bilinear resampling for ch4.
                # TODO Does this make sense for concentrations? I think that we should consider two cases separately:
                # 1. If the resolution of the ch4 is higher than the image, we should do a average/sum resampling (right?).
                # 2. If the resolution of the ch4 is lower than the image, we should do bilinear???
                ch4 = read.resize(
                    ch4,
                    resolution_dst=image.res,
                    anti_aliasing=True,
                    resampling=Resampling.bilinear,
                )
                plume_mask = read.read_reproject_like(
                    plume_mask, ch4, resampling=Resampling.nearest
                )

            if plume_source is not None and isinstance(plume_source, Point):
                # plume_source = plume_source.coords[0]
                plume_source = point_to_pixel_coords(plume_source, ch4)

            ch4 = ch4.values
            plume_mask = plume_mask.values.astype(bool)

        if plume_source is not None and isinstance(plume_source, Point):
            raise ValueError("If plume_source is a Point it the plume_mask must be a GeoTensor")

        # From here ch4 and plume_mask are NDArrays and plume_source is a Tuple[int, int] or None
        assert plume_mask.any(), "plume_mask must have at least one True value"

        if loc_injection is not None and isinstance(loc_injection, Point):
            assert isinstance(
                image, GeoTensor
            ), "If loc_injection is a Point, image must be a GeoTensor too"
            loc_injection = point_to_pixel_coords(loc_injection, image)

        # Convert units to ppb if needed
        if units_ch4 != "ppb":
            from marss2l.mars_sentinel2 import quantification

            ch4 = quantification.convert_units(ch4, units_ch4, "ppb")

        # Smooth the ch4 values and restrinct them to the plume mask
        if smooth_ch4:
            distances_plume = ndi.distance_transform_edt(plume_mask)
            mean_div = np.mean(distances_plume[plume_mask])
            if mean_div <= 1e-6:
                mean_div = 1
            distances_plume = np.clip(distances_plume / mean_div, 0, 1)

            ch4_simulate = ch4 * distances_plume
        else:
            ch4_simulate = ch4 * plume_mask

        # Set NaN values to 0
        ch4_simulate = np.nan_to_num(
            ch4_simulate, nan=0, posinf=self.max_val_ch4_ppb, copy=False, neginf=0
        )
        # Clip values to 0-max_val_ch4_ppb
        ch4_simulate = np.clip(ch4_simulate, 0, self.max_val_ch4_ppb)

        # Rotate the plume according to the wind
        angle = counterclockwise_wind_angle(wind_vector_ch4, wind_vector_image)

        if abs(angle) > 1:
            if plume_source is None:
                # If the plume source is not provided, the center of the plume will
                # be the center of the plume mask.
                center = None
            else:
                # The plume source shall be the center for the rotation.
                # The center parameter in the rotate function is (cols, rows)
                center = plume_source[1], plume_source[0]

            ch4_simulate, new_center = rotate(
                ch4_simulate, angle=angle, center=center, resize=True, cval=0
            )
            plume_mask, new_center = rotate(
                plume_mask, angle=angle, center=center, resize=True, cval=False
            )
            plume_source = int(round(new_center[1])), int(round(new_center[0]))

            # Sanity check, assert that the plume source is within the plume mask
            assert (
                plume_source[0] >= 0
                and plume_source[1] >= 0
                and plume_source[0] < plume_mask.shape[0]
                and plume_source[1] < plume_mask.shape[1]
            ), f"Plume source is outside the plume mask. Plume source: {plume_source}, plume mask shape: {plume_mask.shape}"

        transmittance_b12, transmittance_b11 = self.transmittance_simulator.transmittance_B12_B11(
            satellite, vza, sza, deltach4=ch4_simulate
        )

        plume_mask_full = np.zeros(image.shape[-2:], dtype=bool)

        if loc_injection is None:
            loc_injection = sample_loc_injection(
                image.shape[-2:], self.padding_sample_loc_injection
            )

        upper_left_injection = compute_upper_left_injection_coordinate(
            loc_injection, plume_source, plume_mask.shape
        )

        image_slice, plume_slice = calculate_injection_slices(
            image.shape[-2:], plume_mask.shape, upper_left_injection
        )

        plume_mask_full[image_slice[0], image_slice[1]] = plume_mask[plume_slice[0], plume_slice[1]]

        transmittance_b12_full = np.ones(image.shape[-2:], dtype=transmittance_b12.dtype)
        transmittance_b11_full = np.ones(image.shape[-2:], dtype=transmittance_b11.dtype)

        transmittance_b12_full[image_slice[0], image_slice[1]] = transmittance_b12[
            plume_slice[0], plume_slice[1]
        ]
        transmittance_b11_full[image_slice[0], image_slice[1]] = transmittance_b11[
            plume_slice[0], plume_slice[1]
        ]

        if return_transmittance_and_ch4:
            ch4_simulate_full = np.zeros(image.shape[-2:], dtype=ch4_simulate.dtype)
            ch4_simulate_full[image_slice[0], image_slice[1]] = ch4_simulate[
                plume_slice[0], plume_slice[1]
            ]

        simulated_image = simulate_from_transmittance(
            image, transmittance_b12_full, transmittance_b11_full, b11_index, b12_index
        )

        if isinstance(image, GeoTensor):
            plume_mask_full = GeoTensor(
                plume_mask_full, image.transform, image.crs, fill_value_default=False
            )

        out = {
            "image": simulated_image,
            "label": plume_mask_full,
            "window_row_off": image_slice[0].start,
            "window_col_off": image_slice[1].start,
            "window_width": image_slice[1].stop - image_slice[1].start,
            "window_height": image_slice[0].stop - image_slice[0].start,
        }
        if plume_source is not None:
            plume_source_new_coords = (
                plume_source[0] - plume_slice[0].start,
                plume_source[1] - plume_slice[1].start,
            )
            out["source_row"] = plume_source_new_coords[0] + image_slice[0].start
            out["source_col"] = plume_source_new_coords[1] + image_slice[1].start

        if return_transmittance_and_ch4:
            if isinstance(image, GeoTensor):
                ch4_simulate_full = GeoTensor(
                    ch4_simulate_full, image.transform, image.crs, fill_value_default=0
                )
                transmittance_b12_full = GeoTensor(
                    transmittance_b12_full,
                    image.transform,
                    image.crs,
                    fill_value_default=1,
                )
                transmittance_b11_full = GeoTensor(
                    transmittance_b11_full,
                    image.transform,
                    image.crs,
                    fill_value_default=1,
                )

            out["transmittance_b12"] = transmittance_b12_full
            out["transmittance_b11"] = transmittance_b11_full
            out["ch4"] = ch4_simulate_full

        return out
