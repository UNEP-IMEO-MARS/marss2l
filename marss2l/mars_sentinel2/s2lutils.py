import logging
from typing import Any, Optional

import ee
import numpy as np
import rasterio.warp
import torch
from cloudsen12_models import cloudsen12
from georeader import get_utm_epsg, read
from georeader.geotensor import GeoTensor
from georeader.readers import S2_SAFE_reader, ee_query
from georeader.readers.ee_image import export_image, interpolate_20mbands_s2ee
from rasterio.transform import Affine
from shapely.geometry import Polygon, box, mapping, shape

Resampling = rasterio.warp.Resampling

import logging
from datetime import datetime, timedelta, timezone

from marss2l.mars_sentinel2 import query_images

RELATION_CHANNELS_S2_L89 = {
    "B01": "B01",
    "B02": "B02",
    "B03": "B03",
    "B04": "B04",
    "B08": "B05",  # OR B08A
    "B10": "B09",
    "B11": "B06",
    "B12": "B07",
}


def bands_in_l89(channels_query_s2: list[str]) -> list[str]:
    """This is basically all the channels in the RELATION_CHANNELS_S2_L89 but to make sure they're consistently ordered"""
    return [
        RELATION_CHANNELS_S2_L89[c]
        for c in S2_SAFE_reader.normalize_band_names(channels_query_s2)
        if c in RELATION_CHANNELS_S2_L89
    ]


RELATION_CHANNELS_L89_S2 = {v: k for k, v in RELATION_CHANNELS_S2_L89.items()}

BANDS_S2 = S2_SAFE_reader.BANDS_S2_L1C
BANDS_L89 = bands_in_l89(BANDS_S2)


def gee_info_to_download(
    tile: str,
    band_for_crs_transform: Optional[str] = "B02",
    geometry: Optional[Polygon] = None,
    tile_date: Optional[datetime] = None,
    logger: Optional[logging.Logger] = None,
    delta_hours_search: int = 1,
) -> dict[str, Any]:
    """
    From the name of the tile, figures out the image to download from the Google Earth Engine.

    Args:
        tile (str): tile name
        satellite (str): satellite name
        band_for_crs_transform (Optional[str], optional): Band to get the crs_transform. Defaults to "B02".
        geometry (Optional[Polygon], optional): Geometry to query images if direct tile lookup fails. Defaults to None.
        tile_date (Optional[datetime], optional): Date of the tile for temporal filtering if direct lookup fails. Defaults to None.
        logger (Optional[logging.Logger], optional): Logger instance for logging messages. Defaults to None.
        delta_hours_search (int, optional): Time window in hours for searching images if direct lookup fails. Defaults to 1.

    Returns:
        Dict[str, Any]: dictionary with keys collection_name, gee_id and proj
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    satellite = tile.split("_")[0]

    band_for_crs_transform = S2_SAFE_reader.normalize_band_names([band_for_crs_transform])[0]
    if satellite.startswith("S2"):
        collection_name = "COPERNICUS/S2_HARMONIZED"
        key_filter = "PRODUCT_ID"
        band_idx = BANDS_S2.index(band_for_crs_transform)
    else:
        collection_name = ee_query.figure_out_collection_landsat(tile)
        key_filter = "LANDSAT_PRODUCT_ID"
        band_idx = BANDS_L89.index(band_for_crs_transform)

    img_col = ee.ImageCollection(collection_name)
    image = img_col.filter(ee.Filter.eq(key_filter, tile)).first()
    try:
        info_img = image.getInfo()

        if info_img is None:
            raise ValueError(f"Image {tile} not found in collection {collection_name}")

        crs = info_img["bands"][band_idx]["crs"]
        transform = info_img["bands"][band_idx]["crs_transform"]
        projgee = {"crs": crs, "transform": transform}
        # img_local = ee_image.export_image_fast(image=image, geometry=aoi)
        gee_id = info_img["id"].split("/")[-1]

        angles_keys = [
            "SUN_ELEVATION",
            "MEAN_INCIDENCE_ZENITH_ANGLE_B12",
            "MEAN_SOLAR_ZENITH_ANGLE",
        ]
        # 'EARTH_SUN_DISTANCE', 'REFLECTANCE_CONVERSION_CORRECTION'
        out_dict = info_img["properties"].copy()
        for k in angles_keys:
            if k in out_dict:
                out_dict[k] = float(out_dict[k])

        out_dict["utcdatetime"] = datetime.fromtimestamp(out_dict['system:time_start']/1000,timezone.utc)
        if satellite.startswith("S2"):
            out_dict["tile"] = out_dict["PRODUCT_ID"]
        else:
            out_dict["tile"] = out_dict["LANDSAT_PRODUCT_ID"]

        asset_id = f"{collection_name}/{gee_id}"

        out_dict.update(
            {
                "collection_name": collection_name,
                "gee_id": gee_id,
                "proj": projgee,
                "asset_id": asset_id,
                "crs": crs,
                "transform": Affine(*transform),
            }
        )

        return out_dict
    except ValueError as e:
        logger.warning(
            f"Error figuring out info image to download {tile} from GEE. Trying with query_images.query_gee",
        )
        if satellite.startswith("S2"):
            producttype = "S2"
        elif satellite.startswith("LC08"):
            producttype = "L8"
        elif satellite.startswith("LC09"):
            producttype = "L9"
        else:
            raise NotImplementedError(f"Satellite {satellite} not supported for GEE download")

        images_available_gee = query_images.query_gee(
            geometry,
            date_start=tile_date - timedelta(hours=delta_hours_search),
            date_end=tile_date + timedelta(hours=delta_hours_search),
            producttype=producttype,
            with_wind=False,
            logger=logger,
        )
        images_available_gee = images_available_gee.reset_index()

        # Rename column title by tile
        images_available_gee = images_available_gee.rename(columns={'title': "tile"})

        if (images_available_gee is None) or (len(images_available_gee) == 0):
            logger.error(f"No images found in GEE for tile {tile} and date {tile_date}")
            return

        images_available_gee = images_available_gee[images_available_gee.satellite == satellite]

        if len(images_available_gee) == 0:
            logger.error(
                f"No images found in GEE for tile {tile}, satellite {satellite} and date {tile_date}"
            )
            return

        return images_available_gee.iloc[0].to_dict()


def download_image_and_angles(
    geometry: Polygon,
    image_to_download: dict[str, Any] | None = None,
    tile: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[GeoTensor, GeoTensor, float, float, list[str]]:
    """
    Downloads the image, cloudmask and angles from the Google Earth Engine and process the images
    according to marss2l requirements.

    Args:
        satellite (str): name of the satellite
        image_to_download (dict[str, Any]): dictionary with keys asset_id, crs, transform
    Returns:

        Tuple[GeoTensor, GeoTensor, float, float, list[str]]:
            GeoTensor object with the image reflectances (either S2 or Landsat 8/9),
            GeoTensor object with the cloud mask,
            float with the solar zenith angle,
            float with the view zenith angle,
            Name of the bands in the image (`BANDS_S2` or `BANDS_L89`).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if image_to_download is None:
        try:
            image_to_download = gee_info_to_download(tile, geometry=geometry, logger=logger)
        except Exception as e:
            logger.error(
                f"Error figuring out info image to download {tile} from GEE",
                exc_info=e,
            )
            return

    if any([k not in image_to_download for k in ["asset_id", "crs", "transform"]]):
        missing_keys = [k for k in ["asset_id", "crs", "transform"] if k not in image_to_download]
        raise ValueError(
            f"image_to_insert must have collection_name, gee_id and proj keys. Missing keys: {missing_keys} \nCurrent dict: {image_to_download}"
        )

    tile = image_to_download["tile"]

    satellite = tile.split("_")[0]
    islandsat = not satellite.startswith("S2")
    channels_query_original = BANDS_L89 if islandsat else BANDS_S2
    channels_query = [b.replace("B0", "B") for b in channels_query_original]

    if islandsat:
        channels_query += ["SZA", "VZA"]

    geotensor = export_image(
        image_to_download["asset_id"],
        crs=image_to_download["crs"],
        transform=image_to_download["transform"],
        bands_gee=channels_query,
        geometry=geometry,
    )

    invalids = ~np.all(np.isfinite(geotensor.values), axis=0)
    if geotensor.fill_value_default is not None:
        invalids |= np.any(geotensor.values == geotensor.fill_value_default, axis=0)

    if islandsat:
        # Landsat 8 and 9
        sza_image = geotensor.isel({"band": channels_query.index("SZA")}) / 100
        vza_image = geotensor.isel({"band": channels_query.index("VZA")}) / 100

        # Compute cloud mask
        geotensor = geotensor.isel({"band": slice(0, -2)})
        geotensor.fill_value_default = 0
        tmp = (geotensor.values * 10_000).astype(np.uint16)
        geotensor = GeoTensor(tmp, transform=geotensor.transform, crs=geotensor.crs, fill_value_default=geotensor.fill_value_default)
        geotensor.values[:, invalids] = geotensor.fill_value_default
        cloudmask = compute_cloud_mask(geotensor, channels_query_original, satellite=satellite)

        # Interpolate to 10 meters
        invalids_geotensor = GeoTensor(
            invalids, transform=geotensor.transform, crs=geotensor.crs, fill_value_default=True
        )
        geotensor = read.resize(
            geotensor, resolution_dst=(10, 10), resampling=Resampling.cubic_spline
        )
        invalids_geotensor = read.resize(
            invalids_geotensor, resolution_dst=(10, 10), resampling=Resampling.nearest
        )
        invalids = invalids_geotensor.values
        geotensor.values[:, invalids] = geotensor.fill_value_default
        cloudmask = read.resize(cloudmask, resolution_dst=(10, 10), resampling=Resampling.nearest)

        # Extract angles
        window = read.window_from_polygon(vza_image, geometry, crs_polygon="EPSG:4326")
        point = (
            int(round(window.row_off + window.height / 2)),
            int(round(window.col_off + window.width / 2)),
        )

        # clip the point to 0, width/height
        point = (
            max(0, min(point[0], sza_image.shape[0] - 1)),
            max(0, min(point[1], sza_image.shape[1] - 1)),
        )
        sza, vza = float(sza_image.values[point]), float(vza_image.values[point])
    else:
        vza = image_to_download.get("MEAN_INCIDENCE_ZENITH_ANGLE_B12", None)
        sza = image_to_download.get("MEAN_SOLAR_ZENITH_ANGLE", None)

        # Run cloud detection model before interpolating 20m bands to 10m
        # This is because the model expects the 20m bands to be in the original resolution
        # See https://github.com/IPL-UV/cloudsen12_models/blob/main/notebooks/problem_interp_cloudsen12.ipynb
        geotensor.fill_value_default = 0
        geotensor.values[:, invalids] = geotensor.fill_value_default
        cloudmask = compute_cloud_mask(geotensor, channels_query_original, satellite=satellite)

        geotensor = interpolate_20mbands_s2ee(geotensor, channels_query_original, inplace=False)

    return geotensor, cloudmask, sza, vza, channels_query_original


MODEL_NAME_CLOUDS_S2: str = "UNetMobV2_V2"
MODEL_NAME_CLOUDS_L89: str = "landsat30"

MODEL_CLOUD_DETECTION_S2: cloudsen12.CDModel = None
MODEL_CLOUD_DETECTION_L89: cloudsen12.CDModel = None


def load_model_cloud_detection(
    satellite: str, device: str = torch.device("cpu")
) -> cloudsen12.CDModel:

    global MODEL_CLOUD_DETECTION_S2, MODEL_CLOUD_DETECTION_L89

    islandsat = not satellite.startswith("S2")
    if islandsat:
        if MODEL_CLOUD_DETECTION_L89 is None:
            MODEL_CLOUD_DETECTION_L89 = cloudsen12.load_model_by_name(
                MODEL_NAME_CLOUDS_L89, device=device
            )
            MODEL_CLOUD_DETECTION_L89.bands = bands_in_l89(MODEL_CLOUD_DETECTION_L89.bands)

        return MODEL_CLOUD_DETECTION_L89
    else:
        if MODEL_CLOUD_DETECTION_S2 is None:
            MODEL_CLOUD_DETECTION_S2 = cloudsen12.load_model_by_name(
                MODEL_NAME_CLOUDS_S2, device=device
            )

        return MODEL_CLOUD_DETECTION_S2


def compute_cloud_mask(image: GeoTensor, band_names: list[str], satellite: str) -> GeoTensor:
    """
    Compute the cloud mask for the given image.

    Args:
        image (GeoTensor): image to process
        band_names (List[str]): list of band names of the image
        satellite (str): satellite of the image

    Returns:
        GeoTensor: cloud mask with values cloudsen12.INTERPRETATION_CLOUDSEN12 + ["invalid"]
    """
    invalids = np.any(np.isnan(image.values), axis=0) | np.any(
        image.values == image.fill_value_default, axis=0
    )

    model_cloud_detection: cloudsen12.CDModel = load_model_cloud_detection(satellite)

    image_input = get_channels_to_pred(image, band_names, model_cloud_detection.bands)
    image_input = image_input.astype(np.float32) / 10_000
    cloudmask = model_cloud_detection.predict(image_input)
    cloudmask.values[invalids] = len(
        cloudsen12.INTERPRETATION_CLOUDSEN12
    )  # Set invalids to last value
    cloudmask.fill_value_default = len(cloudsen12.INTERPRETATION_CLOUDSEN12)

    return cloudmask


def get_channels_to_pred(
    img: GeoTensor, channels: list[str], channels_model: list[str]
) -> GeoTensor:
    if channels != channels_model:
        try:
            indexes = [channels.index(band) for band in channels_model]
            image_input = img.isel({"band": indexes})
        except ValueError:
            raise ValueError(
                "Image doesn't have bands compatible with the model: "
                f"\n channels image: {channels} \n channels model: {channels_model}"
            )
    else:
        image_input = img
    return image_input


CRS_LATLONG = "EPSG:4326"


def center_polygon_meters(lon: float, lat: float, margin_meters: float) -> Polygon:
    """
    This function returns for a given lon/lat coordinate a square Polygon with `margin_meters` size centered in this location.
    Output polygon is provided in lon/lat

    Args:
        lon: longitude of the center of the polygon
        lat: latitude of the center of the polygon
        margin_meters: size of the polygon in meters


    Returns:
        Polygon in lon/lat coordinates

    """
    crs = get_utm_epsg((lon, lat))
    coords_transformed = rasterio.warp.transform(CRS_LATLONG, crs, [lon], [lat])

    x, y = coords_transformed[0][0], coords_transformed[1][0]
    # x, y
    pol_crs = box(
        x - margin_meters / 2,
        y - margin_meters / 2,
        x + margin_meters / 2,
        y + margin_meters / 2,
    )
    pol_lat_lng = shape(rasterio.warp.transform_geom(crs, CRS_LATLONG, mapping(pol_crs)))
    return pol_lat_lng
