import json
import logging
import os
import re
import warnings
from datetime import datetime, timezone
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon

MAX_DATE_ERA5 = None

# Sentinel-2 product names embed the datatake sensing-start timestamp. We use it (rather than
# the granule ``system:time_start``) as the canonical ``utcdatetime`` so that candidates queried
# here and a target built via ``S2LLocationImage.from_tile`` agree on ``tile_date`` for the same
# scene. This is the convention marsml uses too, and it is what lets the same-acquisition filter
# discard the target's own scene.
_S2_OPER_RE = re.compile(r"S2[A|B]_OPER_")


def utcdatetime_from_s2_title(title: str) -> datetime:
    """Parse the datatake sensing-start timestamp embedded in a Sentinel-2 product name.

    Modern names (``S2A_MSIL1C_20250529T172859_...``) carry it at characters 11:26; legacy OPER
    names (``S2A_OPER_..._20250529T172859_...``) at characters 25:40. Returns a tz-aware UTC
    ``datetime``.
    """
    substr = title[25 : 25 + 15] if _S2_OPER_RE.match(title) else title[11:26]
    return pd.to_datetime(substr, utc=True).to_pydatetime()


def datetime_with_timezone(date_of_acquisition: datetime) -> datetime:
    """
    Add timezone to a datetime object if it doesn't have it.

    Args:
        date_of_acquisition (datetime): Date of acquisition

    Returns:
        datetime: Date of acquisition with timezone.
    """
    if date_of_acquisition.tzinfo is None:
        return date_of_acquisition.replace(tzinfo=timezone.utc)
    else:
        return date_of_acquisition.astimezone(timezone.utc)


def max_date_era5() -> datetime:
    """
    Get the maximum date available for ERA5 data.
    """
    global MAX_DATE_ERA5
    if MAX_DATE_ERA5 is not None:
        return MAX_DATE_ERA5

    home_dir = os.path.join(os.path.expanduser("~"), ".georeader")
    json_file = os.path.join(home_dir, "marsml.json")

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            json_dict = json.load(f)
        last_update_time_era5 = json_dict["last_update_time_era5"]
        last_update_time_era5 = datetime.fromisoformat(last_update_time_era5).replace(
            tzinfo=timezone.utc
        )

        # if there is less than 1 day with now return last_update_time_era5
        if (datetime.now(tz=timezone.utc) - last_update_time_era5).days < 1:
            MAX_DATE_ERA5 = json_dict["max_date_era5"]
            MAX_DATE_ERA5 = datetime.fromisoformat(MAX_DATE_ERA5).replace(tzinfo=timezone.utc)
            return MAX_DATE_ERA5

    import ee

    image_id = "ECMWF/ERA5_LAND/HOURLY"
    max_date_ms = ee.ImageCollection(image_id).aggregate_max("system:time_start")
    MAX_DATE_ERA5 = pd.to_datetime(max_date_ms.getInfo(), unit="ms", utc=True)

    # cache file in json_file
    os.makedirs(home_dir, exist_ok=True)
    json_dict = {
        "last_update_time_era5": datetime.now(tz=timezone.utc).isoformat(),
        "max_date_era5": MAX_DATE_ERA5.isoformat(),
    }
    with open(json_file, "w") as f:
        json.dump(json_dict, f)

    return MAX_DATE_ERA5


GEE_CHUNK_SIZE = 4500


def download_from_gee(
    locations_dates: gpd.GeoDataFrame,
    datetime_column: str = "date_of_acquisition",
    collection_name: str = "ECMWF/ERA5_LAND/HOURLY",
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Download wind data from GEE. From either of these collections:

    * [NASA/GEOS-CF/v1/rpl/htf](https://developers.google.com/earth-engine/datasets/catalog/NASA_GEOS-CF_v1_rpl_htf#bands)
    * [ECMWF/ERA5_LAND/HOURLY](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY]).
        ERA5-Land data is available from 1981 to **three months from real-time**. The fuction will
        warn if the date of acquisition is less than 3 months from real-time. For ERA5 we query the wind variables
        `u_component_of_wind_10m` and `v_component_of_wind_10m`.

    Args:
        locations_dates (gpd.GeoDataFrame): Locations to download wind data for. Must have a geometry column.
        datetime_column (str, optional): name of the datetime column in locations_dates. Defaults to "date_of_acquisition".
        collection_name (str, optional): Name of the collection to download from GEE. One of:
            "ECMWF/ERA5_LAND/HOURLY", 'NASA/GEOS-CF/v1/rpl/htf'. Defaults to  "ECMWF/ERA5_LAND/HOURLY".

    Returns:
        gpd.GeoDataFrame: Wind data for the locations with columns
            ["geometry", "U", "V", "wind_index", "collection_name_wind"].
            It will have the same index and crs as locations_dates GeoDataFrame.
    """

    assert collection_name in [
        "ECMWF/ERA5_LAND/HOURLY",
        "NASA/GEOS-CF/v1/rpl/htf",
    ], f"collection_name must be one of ['ECMWF/ERA5_LAND/HOURLY', 'NASA/GEOS-CF/v1/rpl/htf'], got {collection_name}"

    assert (
        datetime_column in locations_dates.columns
    ), f"datetime_column must be in locations_dates.columns, got {datetime_column}"

    locations_dates_copy = locations_dates.copy()
    dates_of_acquisition = locations_dates_copy[datetime_column]

    # set timezone to UTC if not already
    locations_dates[datetime_column] = locations_dates[datetime_column].apply(
        datetime_with_timezone
    )

    if collection_name == "ECMWF/ERA5_LAND/HOURLY":
        wind_variables = ["u_component_of_wind_10m", "v_component_of_wind_10m"]
        locations_dates_copy["wind_index"] = dates_of_acquisition.round("60min").dt.strftime(
            "%Y%m%dT%H"
        )

        # warn if date_of_acquisition is in three months from real-time
        if dates_of_acquisition.max() > max_date_era5():
            filter_dates = dates_of_acquisition <= max_date_era5()
            if not filter_dates.any():
                msg = f"All dates queried are posterior to latest available ERA55_LAND data: {max_date_era5()} no wind information will be returned."
                if logger is None:
                    warnings.warn(msg, stacklevel=2)
                else:
                    logger.warning(msg)

                return locations_dates_copy[filter_dates]
            else:
                n_images_with_wind = filter_dates.sum()
                msg = (
                    f"Warning: max date_of_acquisition is posterior to {max_date_era5()}.  Only {n_images_with_wind} out of {len(filter_dates)} have wind available."
                    " ERA5-Land data is available 5 days from real time (see https://climate.copernicus.eu/climate-reanalysis)"
                )
                if logger is None:
                    warnings.warn(msg, stacklevel=2)
                else:
                    logger.warning(msg)

                locations_dates_copy = locations_dates_copy[filter_dates].copy()
    else:
        wind_variables = ["U", "V"]
        locations_dates_copy["wind_index"] = dates_of_acquisition.round("15min").dt.strftime(
            "%Y%m%d_%H%Mz"
        )

    locations_dates_index_name = locations_dates_copy.index.name
    if locations_dates_index_name is None:
        locations_dates_index_name = "location_dates_index"
        locations_dates_copy.index.name = locations_dates_index_name

    if locations_dates_copy.crs is None:
        locations_dates_copy = locations_dates_copy.set_crs("EPSG:4326")

    locations_dates_query = (
        locations_dates_copy[["geometry", "wind_index"]].reset_index().to_crs("EPSG:4326")
    )

    def query_gee(locations_dates_query_iter):
        import ee

        locations_dates_json = eval(locations_dates_query_iter.to_json(drop_id=True))
        locations_dates_ee = ee.FeatureCollection(locations_dates_json)

        image_collection = ee.ImageCollection(collection_name).select(wind_variables)

        def map_fun(feature: ee.Feature) -> ee.Feature:
            image_col = image_collection.filter(
                ee.Filter.eq("system:index", feature.get("wind_index"))
            )

            return ee.Algorithms.If(
                image_col.size().eq(0),
                feature,
                feature.set(image_col.first().reduceRegion(ee.Reducer.mean(), feature.geometry())),
            )

        wind_data_iter = locations_dates_ee.map(map_fun)
        wind_data_iter = gpd.GeoDataFrame.from_features(wind_data_iter.getInfo())
        return wind_data_iter

    if locations_dates_query.shape[0] > GEE_CHUNK_SIZE:
        # Query GEE in chunks of 4500
        wind_data = pd.concat(
            [
                query_gee(locations_dates_query.iloc[i : i + GEE_CHUNK_SIZE])
                for i in range(0, locations_dates_query.shape[0], GEE_CHUNK_SIZE)
            ],
            ignore_index=True,
        )
    else:
        wind_data = query_gee(locations_dates_query)

    if wind_data.shape[0] == 0:
        return wind_data

    wind_data = wind_data.set_index(locations_dates_index_name).set_crs("EPSG:4326")
    wind_data.to_crs(locations_dates_copy.crs, inplace=True)

    wind_data["collection_name_wind"] = collection_name
    wind_data.loc[locations_dates_copy.index, datetime_column] = dates_of_acquisition

    if collection_name == "ECMWF/ERA5_LAND/HOURLY":
        # rename wind columns to "U", "V"
        wind_data = wind_data.rename(
            dict(
                zip(
                    ["u_component_of_wind_10m", "v_component_of_wind_10m"],
                    ["U", "V"],
                    strict=False,
                )
            ),
            axis=1,
        )

    return wind_data


def query_gee(
    area: Union[MultiPolygon, Polygon],
    date_start: datetime,
    date_end: datetime,
    producttype: str = "both",
    with_wind: bool = True,
    add_landsat457: bool = False,
    filter_night_images: bool = True,
    wind_collection="ECMWF/ERA5_LAND/HOURLY",
    logger: Optional[logging.Logger] = None,
) -> gpd.GeoDataFrame:
    """
    Query GEE for S2 and/or Landsat images available for a given area and time period.

    Args:
        area (Union[MultiPolygon,Polygon]): area to query images in EPSG:4326
        date_start (datetime): datetime in a given timezone. If tz not provided UTC will be assumed.
        date_end (datetime): datetime in UTC. If tz not provided UTC will be assumed.
        producttype (str, optional): 'S2', "Landsat"-> {"L8", "L9"}, "both" -> {"S2", "L8", "L9"}, "S2_SR", "L8", "L9". Defaults to 'S2'.
        with_wind (bool, optional): If True, will also download wind data from `wind_collection`. Defaults to True.
        add_landsat457 (bool, optional): If True, will also query from Landsat 4, 5 and 7. Defaults to False.
        filter_night_images (bool, optional): If True, will filter out images with sun elevation lower than 0. Defaults to True.
        wind_collection (str, optional): Name of the collection to download wind data from. Defaults to "ECMWF/ERA5_LAND/HOURLY".
        logger (Optional[logging.Logger], optional): Logger to use. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Images available for the given area and time period.
            Columns:
            - "solarday".
            - "satellite". S2A, S2B, LC08, LC09, LE07, LT05, LT04
            - "overlappercentage".
            - "cloudcoverpercentage".
            - "utcdatetime".
            - "U" (optional). Component U of the wind.
            - "V" (optional). Component V of the wind.
            - "collection_name_wind" (optional). Name of the collection where the wind data was downloaded.
            - 'geometry'
            - 'gee_id': GEE image id
            - 'system:time_start'
            - 'collection_name': Name of the GEE collection
            - 'solardatetime'
            - 'localdatetime'
            - 'wind_index'
            - 'date_of_acquisition'
            - 'proj' Dict with crs and transform
            - 'asset_id': Concatenation of the collection_name and the gee_id
            Index: 'title' # Name of the S2/Landsat image without the extension

    """

    from georeader.readers import ee_query

    if producttype in ["LT04", "LT05", "LE07"]:
        # empty df
        images_available_gee = pd.DataFrame()
        add_landsat457 = True
    else:
        images_available_gee = ee_query.query(
            area,
            date_start,
            date_end,
            producttype=producttype,
            return_collection=False,
            extra_metadata_keys=[
                "SUN_ELEVATION",
                "MEAN_INCIDENCE_ZENITH_ANGLE_B12",
                "MEAN_SOLAR_ZENITH_ANGLE",
                "EARTH_SUN_DISTANCE",
                "REFLECTANCE_CONVERSION_CORRECTION",
            ],
        )

    if (images_available_gee.shape[0] > 0) and (images_available_gee.index.duplicated().sum() > 0):
        # reset index
        images_available_gee_tofix = images_available_gee.reset_index()
        for title, images_grouped in images_available_gee_tofix.groupby("title"):
            if images_grouped.shape[0] > 1:
                for i, idx in enumerate(images_grouped.index):
                    images_available_gee_tofix.loc[idx, "title"] = f"{title}_{i}"

        images_available_gee = images_available_gee_tofix.set_index("title")

    if add_landsat457:
        images_available_gee_landsat457 = ee_query.query_landsat_457(
            area,
            date_start,
            date_end,
            producttype="all",
            return_collection=False,
            extra_metadata_keys=[
                "SUN_ELEVATION",
                "MEAN_INCIDENCE_ZENITH_ANGLE_B12",
                "MEAN_SOLAR_ZENITH_ANGLE",
                "EARTH_SUN_DISTANCE",
                "REFLECTANCE_CONVERSION_CORRECTION",
            ],
        )
        if images_available_gee_landsat457.shape[0] > 0:
            if images_available_gee.shape[0] == 0:
                images_available_gee = images_available_gee_landsat457
            else:
                images_available_gee = pd.concat(
                    [images_available_gee, images_available_gee_landsat457], axis=0
                )

    if images_available_gee.shape[0] == 0:
        return images_available_gee

    # Take MEAN_SOLAR_ZENITH_ANGLE or corresponding SUN_ELEVATION depending if it is S2 or Landsat
    images_available_gee["mean_solar_zenith_angle"] = images_available_gee.apply(
        lambda x: (
            x["MEAN_SOLAR_ZENITH_ANGLE"]
            if x["satellite"].startswith("S2")
            else 90 - x["SUN_ELEVATION"]
        ),
        axis=1,
    )

    if filter_night_images:
        images_available_gee = images_available_gee[
            images_available_gee["mean_solar_zenith_angle"] < 90
        ]

    # If producttype S2 set up the utcdatetime attribute from the tile name (datatake stamp).
    if producttype in ["S2", "S2_SR", "both"]:
        s2_images_idx = images_available_gee.satellite.apply(lambda x: x.startswith("S2"))
        images_available_gee.loc[s2_images_idx, "utcdatetime"] = pd.to_datetime(
            images_available_gee[s2_images_idx].index.map(utcdatetime_from_s2_title),
            utc=True,
        )

    # Replace LO09 with LC09 and LO08 with LC08
    images_available_gee.satellite = images_available_gee.satellite.apply(
        lambda x: x.replace("LO09", "LC09").replace("LO08", "LC08")
    )

    # Add asset_id column
    images_available_gee["asset_id"] = images_available_gee.apply(
        lambda x: f"{x['collection_name']}/{x['gee_id']}", axis=1
    )
    # Add crs and transform from "proj" column
    # crs=proj["crs"]
    # transform=Affine(*proj["transform"])
    images_available_gee["crs"] = images_available_gee["proj"].apply(lambda x: x["crs"])
    images_available_gee["transform"] = images_available_gee["proj"].apply(
        lambda x: Affine(*x["transform"])
    )

    if with_wind:
        if logger is not None:
            logger.info(f"Downloading wind data from {wind_collection}")

        df_for_query = images_available_gee[["utcdatetime"]]
        df_for_query = gpd.GeoDataFrame(
            df_for_query,
            geometry=[area.centroid for _ in range(df_for_query.shape[0])],
            crs="EPSG:4326",
        )
        wind_info = download_from_gee(
            df_for_query,
            datetime_column="utcdatetime",
            collection_name=wind_collection,
            logger=logger,
        )
        if wind_info.shape[0] == 0:
            return images_available_gee
        images_available_gee_merged = pd.merge(
            images_available_gee,
            wind_info[
                [
                    c
                    for c in wind_info.columns
                    if c in ["U", "V", "wind_index", "collection_name_wind"]
                ]
            ],
            how="left",
            right_index=True,
            left_index=True,
        )

        return images_available_gee_merged

    return images_available_gee
